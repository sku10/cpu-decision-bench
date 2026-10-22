# Decision models (System One / Jev) — usage reference

A **decision model** takes a state plus one or more *typed questions* and returns
a typed value with a calibrated probability. It generates no text: no prose, no
reasoning trace, no JSON to parse. Code branches on the value directly.

This file is the reference other projects read before building on one. It covers
the wire format, the four question types, the confidence math, the limits, and
what we actually measured locally.

- **Scope note.** The category is new (first model shipped 2026-09-15) and moving
  fast. Facts below are dated; re-check anything load-bearing against the
  upstream docs, linked per section.
- **Not an LLM.** Do not send it a prompt and expect text. Do not ask it to
  write, summarise, extract free-form, or call tools. It answers closed
  questions about state you supply.

---

## 1. The model class

| | LLM | Decision model |
|---|---|---|
| Output | tokens (text) | a typed value + probability |
| Parse step | yes (regex/JSON/schema coax) | none — it's already typed |
| Reasoning trace | often | never |
| Latency | 10² – 10⁴ ms | ~100 ms typical |
| Billing | input + output tokens | **input tokens only** |
| Failure mode | plausible wrong text | wrong value, *with* a low probability |

The design claim is that the probability is **calibrated** — a stated 0.8 is
right about 80% of the time — because training optimises calibration rather
than rater preference (RLCD, "Reinforcement Learning for Calibrated
Decisions"). That is the property worth verifying on your own data, and the one
that makes threshold-based routing sound. See §6 for our local measurement.

> **Name collision.** "RLCD" also means *Reinforcement Learning from Contrastive
> Distillation* (arXiv 2307.12950, ICLR 2024) — an unrelated RLAIF variant. The
> decision-model sense is the 2026 TypeSafe one.

---

## 2. Wire format

One endpoint. One state. N independent questions, answered in parallel.

```
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer $TYPESAFE_API_KEY
Content-Type: application/json
```

```jsonc
{
  "model": "jev-latest",
  "state":  { ... },          // the content to judge: string | object | array
  "questions": {
    "<your_question_id>": {   // you choose the id; the model never sees it
      "type": "noul" | "choice" | "score" | "box",
      "instructions": "...",  // string | object | array
      "criteria": ...         // shape depends on type (see §3)
    }
  }
}
```

Response — one entry in `answers` per question id, plus `usage`:

```jsonc
{
  "model": "jev-1.13.0",      // the dated snapshot that actually served it
  "answers": { "<your_question_id>": { "type": "...", ... } },
  "usage": { "input_tokens": 360, "output_tokens": 39 }
}
```

**Questions cannot see each other.** They are evaluated independently against the
same state. There is no chaining inside a request — compose in code, or make a
second call.

### State

`state` is the material to judge — "what you would hand a panel of experts
before asking them to judge." String, JSON object, or array.

```jsonc
// string — simplest
"state": "My card was charged twice."

// object — preferred: every part is named, relationships stay legible
"state": {
  "ticket":  { "subject": "Duplicate charge", "messages": [ ... ] },
  "order":   { "id": "A-104", "charges": [ {"amount_usd": 49}, {"amount_usd": 49} ] },
  "refund_policy": "Duplicate charges are eligible for a refund."
}
```

Questions can point at a nested path using **backticks**: `` "Does `ticket.message`
request a refund?" ``. Use that whenever it removes ambiguity.

No audio, no video. Images: text-only on 1.13; 1.14 adds image input (§3.4).
Convert everything else to text or structured fields first.

---

## 3. The four question types

### 3.1 `noul` — yes/no

Returns a **single probability that the answer is yes**. 0 = no, 1 = yes,
~0.5 = genuinely undecided.

```jsonc
"is_human_escalation": {
  "type": "noul",
  "instructions": "Is the customer asking for a human agent?"
}
// optional: say what yes and no mean
"is_repeat_contact": {
  "type": "noul",
  "instructions": "Has the customer contacted support about this before?",
  "criteria": {
    "true":  "Mentions a prior attempt, ticket, or that they have asked before",
    "false": "No sign of any previous contact"
  }
}
```

```jsonc
"is_human_escalation": { "type": "noul", "noul": 0.99 }
```

**There is no `confidence` field on a noul** — and that is not an omission. The
distribution has only two outcomes, so the one number describes it completely.
Read certainty as distance from 0.5.

Calibration in practice, same question across inputs (recorded `jev-1.13.0`):

| state | `noul` |
|---|---|
| Thanks, that fixed it! | 0.02 |
| How do I reset my password? | 0.07 |
| I need this sorted today, whatever it takes. | 0.26 |
| Are you a bot? | 0.40 |
| Is there any way to speak to someone about my invoice? | 0.84 |
| I have asked three times now. Can I please just talk to a real person? | 0.99 |

Rules that matter:
- **One proposition per noul.** "Is the customer angry *and* asking for a
  refund?" forces two judgments into one number. Ask two, combine in code.
- **Phrase so high = yes.** "Is the message free of personal data?" inverts the
  polarity and the code reading it later will get it backwards.
- **A noul is not a scale.** It is P(proposition true), not the degree of the
  thing. "Is the candidate strong in Python?" → 0.81 does not mean "81% skilled".
  If you want degree, use a `score`, whose levels you actually wrote.

### 3.2 `choice` — one of N

`criteria` is a map of option name → description. **Up to 255 options.**
Descriptions may be `null` when the names speak for themselves.

```jsonc
"department": {
  "type": "choice",
  "instructions": "Which team should handle this?",
  "criteria": {
    "returns":  "Exchanges, wrong or damaged items",
    "shipping": "Delivery status, delays, lost packages",
    "billing":  "Charges, invoices, payment problems"
  }
}
```

```jsonc
"department": {
  "type": "choice",
  "choice": "returns",            // argmax
  "confidence": 0.42,             // derived from the spread (§4)
  "probabilities": { "shipping": 0.04, "billing": 0.35, "returns": 0.61 }
}
```

`probabilities` sums to 1 and is the richer signal — a 0.35 runner-up is
actionable information (notify the second team) that `choice` alone hides.

- Give the **full** option list, not a shortlist; options cost a few tokens each.
- Add `other` / `none_of_the_above` when the list may not cover the input — but a
  catch-all option often works poorly. Better: a companion `noul` ("does
  anything here fit at all?") and read the choice only when it says yes.
- Deep taxonomies: chain choices level by level (beam search over
  probabilities), don't flatten to 255.

### 3.3 `score` — ordered scale

`criteria` is an **ordered array** of level descriptions, low → high. 2–10 levels.
Level numbers are array indices starting at 0.

```jsonc
"bug_severity": {
  "type": "score",
  "instructions": "How severe is the reported issue?",
  "criteria": [
    "Cosmetic; no impact to functionality",
    "Broken or degraded feature, but workaround exists",
    "Blocking issue; no workaround exists"
  ]
}
```

```jsonc
"bug_severity": {
  "type": "score",
  "score": 1.43,                  // Σ level × P(level) — can land between levels
  "confidence": 0.35,
  "legend": { "0": "Cosmetic...", "1": "Broken...", "2": "Blocking..." },
  "probabilities": { "0": 0.0, "1": 0.57, "2": 0.43 }
}
```

- `score` is a **probability-weighted mean**, so different distributions give the
  same number: 1.0 can be "all on level 1" or "half on 0, half on 2". Always read
  `probabilities`/`confidence` alongside it.
- **Describe situations, not degrees.** "Moderately severe" gives the model
  nothing to match. Each level is judged independently — the model never sees a
  level's number or its neighbours, so "worse than the previous level" is
  meaningless, and numeric-only levels (`["0","1","2"]`) measurably degrade the
  answer.
- **One dimension per score.** A level reading "punctual and smart and
  experienced" cannot place an input that is high on one and low on another.
- Give a rare extreme its own level ("abusive or threatening" above "very
  angry"), or it collapses into the top level.
- Combining several scores: **normalise first** — divide by `len(criteria) - 1`,
  since a 4-level scale tops out at 3 and a 3-level at 2. Then weight in code.

### 3.4 `box` — locate in an image (1.14+, early feature)

Takes `instructions` only (no `criteria`), returns edges as fractions of image
width/height.

```jsonc
"product": { "type": "box", "instructions": "the product in the listing photo" }
```
```jsonc
"product": { "type": "box",
  "box": { "top": 0.49, "bottom": 0.89, "left": 0.58, "right": 0.96 },
  "confidence": 0.93 }
```

**Box has no "not found" answer** — it returns four edges even when the target is
absent. Pair it with a `noul` ("is the thing present at all?") and gate on that.

Image input and Box are flagged by the vendor as early features whose format may
change. Requesting `box` on a text-only model returns `400`.

---

## 4. Confidence — the exact math

`confidence` is **not** a separate model output. It is a statistic computed from
the `probabilities` the answer already contains, returned for convenience so you
can threshold from the first call. 1 = all mass on one outcome, 0 = flat.

Because the formulas are exact and you always get the full distribution, you can
compute a different statistic if it suits your problem better.

Runnable, with a self-test against every published vendor value:
[`docs/decision_confidence.py`](decision_confidence.py) (`python3 -I
docs/decision_confidence.py` → `ok - 12 published values reproduced`). It also
has `top_probability` and `top_to_second_ratio`, two simpler measures the vendor
notes are often more effective in practice.

**noul** — none returned. For a comparable number, use distance from 0.5:
```python
confidence = abs(2 * p - 1)          # 0 at p=0.5, 1 at p=0 or p=1
```

**choice** — how far the top option sits above an even split:
```python
def choice_confidence(probs):        # probs: list[float]
    n = len(probs)
    return (max(probs) - 1/n) / (1 - 1/n)
```
Only the top probability counts: `(0.6,0.3,0.1)` and `(0.6,0.2,0.2)` both → 0.4.

**score** — distance-weighted, because levels are ordered:
```python
def score_confidence(probs):         # probs: list[float], index = level
    n = len(probs)
    m = probs.index(max(probs))
    spread      = sum(p * abs(i - m) for i, p in enumerate(probs))
    even_spread = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1 - spread / even_spread)
```
Mass on a *neighbouring* level costs less than mass far away: with 3 levels
`(0, 0.5, 0.5)` → 0.25 (torn between adjacent), `(0.5, 0, 0.5)` → 0.0 (torn
between opposite ends). The choice formula would score both 0.25.

> **Local harness caveat.** Our `run_jevbench.py` derives noul confidence as
> `p if p >= 0.5 else 1 - p`, which maps to **0.5–1.0**, not the official
> **0–1** of `|2p-1|`. Both rank identically, so accuracy and bucketing by
> "≥0.90" are unaffected, but the two are not interchangeable in an ECE
> number. Convert before comparing our ECE to a published one.

### Thresholds

The vendor's own framing, which we endorse: **one threshold is wrong.** Gate each
action at a level set by the cost of that action being wrong.

```python
if confidence < 0.5:                 # genuinely unsure — don't guess
    route_to_human(msg)
elif action == "check_balance":      # read-only; a wrong screen is recoverable
    show_balance(acct)
elif action == "approve_transfer":   # destructive
    confirm_then_execute(acct) if confidence > 0.9 else ask_user_to_confirm(acct)
```

Three bands as a starting shape: **high** → act; **medium** → act with
confirmation / flag for review; **low** → don't act, escalate. Pick the
boundaries by plotting confidence against accuracy on *your* labelled data, not
from round numbers. `confidence` describes the spread of the alternatives — it
is **not** a statement that the workflow is safe to run.

---

## 5. Build pattern

The architectural claim: this is for **AI-powered software**, not agents. Code
owns control flow; the model supplies narrow judgments over unstructured data.

1. **Use code where you can.** Deterministic rules are cheaper and reliable.
   Short-circuit before calling the model (`if ticket.status == "closed": return`).
2. **Send only relevant context.** Avoid context rot; don't rely on weights for
   facts your own database holds.
3. **Structure the state.** Nested JSON, and point questions at paths with
   backticks.
4. **Decompose the questions** — the single most important rule. Atomic, explicit,
   one property each. Broad questions hide several judgments behind one number;
   atomic ones expose them so you can inspect, tune and weight them separately.
5. **Structure the questions when it helps.** `instructions` and each `criteria`
   value may be an object — put the question in one field and its guiding data in
   others (`what` / `not_for` / `examples` / `question` / `focus`). Field names are
   yours, none are reserved, and the model sees the names too. Reach for this when
   two options keep getting confused, or part of the question comes from code.
6. **Ask a lot of questions in one request.** Parallel evaluation; adding
   questions barely moves latency and costs only their tokens. Speculative
   fan-out — ask questions that only *some* code paths need, then ignore the rest —
   is the intended pattern, not waste.
7. **Compose in code**: weighted sums, deterministic rules, or feed the
   probabilities as features to a classical ML model.
8. **Route on uncertainty**, per §4.

Worked composition — independent signals, weights owned by code:
```python
spam_risk = (0.45 * a["requests_credentials"].noul
           + 0.30 * a["sender_identity_mismatch"].noul
           + 0.25 * a["unexpected_reward"].noul)

if 0.4 < spam_risk < 0.6 or a["topic"].confidence < 0.75:
    return route_to_human_review(ticket)     # uncertain → person
if spam_risk >= 0.6:
    return quarantine_as_spam(ticket)
```
The weights live in your code, visible and tunable — that is the point of
decomposing. Structured criteria for the three nouls above would each carry
`true`/`false` objects with `what`, `not_for` and `examples`.

---

## 6. What we measured locally (2026-10-21)

Not vendor numbers. Open-weights **H2O-Lightning-4B** (a Qwen3.5-4B fine-tune,
Apache-2.0) on the **231 public JevBench decisions**, native `POST /v1/systemone`
in llama.cpp (build 200, commit `0a1ef30`, containing PR #30361), **RTX 3060 12 GB**.
All three dataset SHA256s matched the manifest. Harness and per-item results:
the benchmark host.

| precision | file size | VRAM | correct | ECE | p50 | p95 |
|---|---|---|---|---|---|---|
| BF16   | 7.9 GiB | 8690 MiB | **205/231** | 0.0558 | 94 ms | 1208 ms |
| Q8_0   | 4.2 GiB | 4934 MiB | **204/231** | 0.0597 | 65 ms | 858 ms |
| Q5_K_M | 2.9 GiB | 3592 MiB | 202/231 | 0.0514 | 67 ms | 921 ms |
| Q4_K_M | 2.6 GiB | 3242 MiB | 202/231 | 0.0372 | 65 ms | 900 ms |

By type (Q8_0): choice 123/139, noul 66/74, score 15/18.

### Same runs with no GPU at all (2026-10-22)

Identical binary and weights, `-ngl 0 -t 16`, so every layer runs on the
**Ryzen 7 5800X (8C/16T)**. Nothing else changed.

| | correct | ECE | p50 | p95 | mean | wall (231) |
|---|---|---|---|---|---|---|
| Q8_0 GPU | 204/231 | 0.0597 | 64 ms | 859 ms | 219 ms | 50 s |
| **Q8_0 CPU** | **204/231** | 0.0558 | **424 ms** | **2768 ms** | 838 ms | **194 s** |
| Q4_K_M GPU | 202/231 | 0.0372 | 65 ms | 900 ms | 228 ms | 53 s |
| **Q4_K_M CPU** | **202/231** | 0.0399 | **300 ms** | **2154 ms** | 633 ms | **146 s** |

- **Accuracy is unchanged on CPU** at both precisions. Q8_0 differs on **0 of 231
  answers**; per-type and per-family breakdowns match exactly.
- **Not bit-identical**, unlike a GPU re-run after a kernel update (which was
  0.000e+00 drift). 222 of 231 Q8_0 confidences move, max 0.046 — different BLAS
  kernels accumulate in a different order. Every large drift sits on a
  low-confidence torn item (0.35–0.58), where the distribution is flat and small
  logit changes move the most probability.
- **No Q8_0 item crosses the 0.80 or 0.90 confidence band.** A gate of "act at
  ≥0.90, else escalate" makes the same decision on all 231 items on either
  device. The calibration property survives the device change.
- Q4_K_M on CPU flips **4** answers, net zero (2 lost, 2 gained). All four sit at
  confidence ≤0.578 — the model was already saying "unsure" on every one. Q8_0 is
  the more device-stable choice if you mix CPU and GPU workers.
- **Cost is ~2.8–3.8× on the mean, worse at the tail** (p95 degrades 2.4–3.2×,
  p50 4.6–6.6×). Short items fare worst in *ratio* terms because fixed overhead
  dominates; big-state items (`long_policy`, `multi_hop`) cost the most in
  absolute ms, since prompt ingestion is compute-bound.
- Practical read: **CPU-viable for batch/queue work** — 231 decisions in ~2.5–3.3
  min on a 5-year-old desktop CPU, 4.2 GB (Q8_0) or 2.6 GB (Q4_K_M) of RAM, no
  GPU. Not viable behind an interactive request path where a ~2.5 s p95 would be
  felt.

Findings that transfer:

- **Calibration survives quantisation.** The ≥0.90 confidence band gives 99.3%
  actual accuracy on *all four* precisions (n=143–146). The model **understates**
  its reliability when confident — only 1 of 27 errors sat above 0.90. This is the
  property the design promises, and it held.
- **Precision is not the lever.** Q8 vs Q4 differ on 6 of 231 answers; McNemar
  exact p=0.625. 24 items failed on all four; oracle best-of-four = 207/231.
- **`output_tokens` = 0** on every local run, so "output is free" is structural
  here. Note the hosted API *does* report output tokens (e.g. 39, 70) while still
  billing them at zero — so don't use `output_tokens == 0` as a health check
  against the hosted endpoint.
- **Failures are concentrated, not diffuse.** `temporal_numeric` 5/15 is the worst
  family, then `judge_hard` 12/17, `long_policy` 14/19, `multi_hop` 14/18. Date
  and duration arithmetic is the weak spot, and errors there are off-by-a-little
  (`sep_24` vs `sep_26`) arriving at *low* confidence (0.27–0.53) — i.e. the
  calibration catches them. Meanwhile `extraction` 24/24, `intent` 24/24,
  `routing` 12/12, `tool_selection` 12/12, `policy` 12/12 are solved.
- **One-sided permissive bias on `noul`.** Across all 74 noul questions: 7 errors
  wrongly "yes" vs 1 wrongly "no". Within `judge_hard`, *every* error was
  wrongly-permissive, zero in the other direction (gold was 10 False / 7 True, so
  not a label artifact). **Where the risky answer is "yes" — approve, allow,
  release, deductible — set the threshold high on that branch**, or invert the
  question so "no" is the easy answer.
- Re-run after an OS/kernel update returned **bit-identical** results (0 answers
  changed, max confidence drift 0.000e+00).

### Reading JevBench's own leaderboard

**The composite score is a weighted blend in which cost and speed together are
half or more** (v1.1 `(Capability+Speed+Cost)/3`; v1.2 Intelligence/Calibration/
Speed/Cost 25% each, geometric; v1.6 harmonic). Consequence: a model with the
best Intelligence (94.3) and Calibration (96.7) on the board ranks **#28** on
composite, dragged down by Cost 16.8. JevBench v1 deliberately published *no*
combined score.

So "open model ties commercial on composite" largely restates "it is cheaper."
Compare the **Intelligence and Calibration columns** instead. If we ever store
these in the catalog, they need their own fields — putting a composite in
`elo_score` would repeat the Arena `net-improvement` error from commit `25fc1d4`.

---

## 7. Limits, pricing, versions

As of **2026-10-22**:

| | Jev 1.14 | Jev 1.13 |
|---|---|---|
| Model ID | `jev-1.14.0` | `jev-1.13.0` |
| Price | **$0.042 / Mtok input** ($42/Btok) | same |
| Output | **free** | free |
| Per-request context | 64k | 64k |
| Per-question context | 32k | 32k |
| Input | text + images | text |

**Two context limits, not one** — this resolves what looked like a data conflict
in our catalog:
- `per_request_context` — state + *all* questions, 64k
- `per_question_context` — state + *any one* question, 32k
- **Long context**: send header `TypeSafe-Danger-Zone-Long-Context: true` and both
  become **262k**. Same price per token, but slower and **less accurate** — use
  only where needed.

No cap on question count, as long as both limits hold.

**Aliases move.** `jev-latest` → most recent *stable* (currently 1.13.0);
`jev-preview` → most recent *including* previews (1.14.0). The response's `model`
field always names the dated snapshot that served the call — **log it**. If you
have tuned thresholds against a version, pin the versioned ID.

**Language.** English is the primary training language and where accuracy is
best. Other languages are "handled but not equally well." → **For Finnish
workloads, measure before relying on it, and watch confidence when routing.**

**No fine-tuning.** Same weights for every account; no LoRA, no customer-data
training. You adapt it through `state`, `instructions`/`criteria`, and
decomposition — not weights.

**Not trained on your requests.** ZDR available for enterprise.

Rate limits are per-organisation, measured in tokens/sec and requests/sec; over
either → `429`. Official SDKs retry with backoff and honour `retry-after`.

---

## 8. Access and the wider category

Same request shape, different host — `model` selects, so switching is a base-URL
plus model-id change. **Re-validate thresholds after any switch.**

| Host | Endpoint |
|---|---|
| TypeSafe | `POST https://api.typesafe.ai/v1/systemone` |
| OpenRouter (System One) | `POST https://openrouter.ai/api/v1/systemone` |
| OpenRouter (Decisions) | `POST https://openrouter.ai/api/alpha/decisions` |

OpenRouter needs no TypeSafe account or waitlist — an OpenRouter key is enough,
and each response carries `usage.cost` in USD. Models appear there as
`typesafe/jev-1.14`, alias `~typesafe/jev-latest`.

Official SDKs (`typesafe_sdk` Python, `@typesafe-ai/sdk` TS) point at another
host with a one-line base-URL change:
```python
client = TypeSafeClient(api_key=os.environ["OPENROUTER_API_KEY"],
                        base_url="https://openrouter.ai/api")
```

**The category is now crowded.** OpenRouter listed 22 decision models from 13
makers as of 2026-10-16 (OpenAI `gpt-6-luna-decisions`, Microsoft, Perplexity
`pplx-decider`, Cloudflare, Upstage, …); JevBench v1.2 lists 36 systems. Treat
"Jev" as the reference implementation of a format, not the only option — and note
that open-weights models in this class run on one mid-range GPU (§6).

---

## 9. Notes for catalog and tooling authors

Decision models break assumptions most model catalogs and client libraries are
built on. If you maintain either, these are the ones that bite:

- **`output_modalities: decisions`** is a real modality on some routers. A schema
  that assumes text-in/text-out has nowhere to put it.
- **Pricing is input-only.** A completion price of `0` is correct, not missing data.
  Cost estimators that multiply output tokens by a rate will report nonsense.
- **Quality has at least two axes** (intelligence and calibration) and neither is
  an Elo. Storing a blended leaderboard score in an Elo field loses the
  distinction that matters — see the composite-score warning in section 6.
- **Context is two numbers plus an opt-in third** (per-request, per-question, and
  the long-context mode). A single `context_window` field cannot represent them,
  and sources that expose only one will disagree with each other.
- **Capabilities like `tools` and `reasoning` are meaningless here.** There is no
  reasoning mode to enable; sending a reasoning parameter is an error, not a
  no-op. Treat `supports_reasoning` as false.

## Sources

Upstream vendor docs (read 2026-10-22) — authoritative, re-check before relying:
- Primitives: [noul](https://docs.typesafe.ai/primitives/noul) ·
  [choice](https://docs.typesafe.ai/primitives/choice) ·
  [score](https://docs.typesafe.ai/primitives/score) ·
  [box](https://docs.typesafe.ai/go/box)
- [Confidence](https://docs.typesafe.ai/confidence) — the exact formulas
- [State](https://docs.typesafe.ai/concepts/state) ·
  [How to build with System One](https://docs.typesafe.ai/concepts/how-to-build-with-system-one) ·
  [System One](https://docs.typesafe.ai/concepts/system-one)
- [Models](https://docs.typesafe.ai/models) — pricing, context, aliases, language
- SDKs: [Python](https://docs.typesafe.ai/sdk/python) · [JavaScript](https://docs.typesafe.ai/sdk/javascript)

Host docs:
- [Jev on OpenRouter](https://openrouter.ai/docs/guides/community/jev) ·
  [Jev tutorial](https://openrouter.ai/docs/guides/community/jev-tutorial) (full
  worked request + a live captured response)
- [Decision models on OpenRouter](https://openrouter.ai/models?output_modalities=decisions)

Papers / benchmark:
- arXiv 2610.25553 "Narrow Decision Models" — **unverified**, not retrieved
- [Sys1Cal-v1 calibration dataset](https://arxiv.org/html/2609.35342v1) — reports
  noul and score better calibrated than choice, and choice less invariant under
  equivalent reformulations
- [Just Ask Jev](https://arxiv.org/html/2609.29429v1) — notes probabilities
  calibrated when pooled but not within a benchmark
- JevBench: `fstandhartinger/jevbench` (MIT) — read §6 before quoting a composite

Local measurement: see [`README.md`](README.md) and the scripts in this repository.

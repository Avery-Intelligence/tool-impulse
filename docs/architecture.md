# Architecture

`tool-impulse` selects a relevant subset of tools from a larger catalog before sending them to an LLM prompt. This document describes the retrieval pipeline, scoring math, and multi-turn context tracking.

## Pipeline Overview

```
User Query
  │
  ├─► Tokenizer & Stemmer ──► BM25 Lexical Score (0.0 to 1.0)
  │
  └─► Query Vector ─────────► Cosine Similarity (0.0 to 1.0)
                                 │
                                 ▼
                    Blended Score = α · Vector + (1 - α) · BM25
                                 │
                                 ▼
                    Apply Multi-Turn Trajectory & Inertia
                                 │
                                 ▼
                    Apply Companion Workflow Boosts
                                 │
                                 ▼
                    Filter by Domain Diversity Cap (maxPerDomain)
                                 │
                                 ▼
                    Top-K Tools Returned
```

---

## 1. Hybrid Ranking

When an embedder is configured, scores are combined using the `alpha` weight:

$$\text{score} = \alpha \cdot \text{cosine} + (1 - \alpha) \cdot \text{bm25}$$

- `alpha = 0.0`: Pure lexical BM25 (the default when no embedder is configured). Runs 100% in-process with zero network overhead.
- `alpha = 0.70` (default with embedder): Blends semantic similarity with keyword matching.
- `alpha = 1.0`: Pure dense vector similarity.

### BM25
- Standard Okapi BM25 with parameters $k_1 = 1.2$ and $b = 0.75$.
- Tokenizer splits `snake_case`, `camelCase`, and punctuation, normalizes Unicode diacritics via NFD, and applies Porter stemming.
- Raw scores are normalized against query-length maximums into the $[0.0, 1.0]$ interval.

### Dense Vectors
- Cosine similarity is computed across normalized `Float32Array` vectors.
- Embeddings can be pre-computed at build time, passed via built-in embedders (OpenAI, Gemini, Cloudflare), or computed locally using an in-process model (e.g. ONNX).

---

## 2. Multi-Turn Trajectory Blending

In conversational turns with pronouns (*"now refund that charge"*), the query often lacks domain keywords. `tool-impulse` preserves context across turns by blending query vectors:

$$\vec{v}_{\text{active}} = \text{normalize}(\beta \cdot \vec{v}_{\text{current}} + (1 - \beta) \cdot \vec{v}_{\text{prior}})$$

- Default `beta = 0.75` gives 75% weight to the immediate message and 25% to prior turn context.
- **Intent-Shift Cutoff:** If cosine similarity between the current and prior turn is below `driftThreshold` (default: 0.35), the user switched topics (e.g. moving from billing to weather). Prior context is dropped immediately to prevent cross-topic contamination.
- **Inertia Bonus:** Tools called in the immediate previous turn receive a score bonus (`inertiaBonus`, default: 0.20) to keep active workflows mounted.

---

## 3. Companion Workflow Graph

Tools frequently operate in sequential chains (e.g. `jira_get_issue` -> `jira_update_issue`).

You can define directed edges between tools. When an anchor tool matches the user's prompt, its companions receive an activation boost:

$$\text{boost} = \text{anchorScore} \cdot \text{edgeWeight} \cdot \text{companionBoost}$$

This makes related tools available without requiring explicit keywords for every step in the initial prompt.

---

## 4. Domain Diversity Capping

When an agent has many tools from one service (e.g. 20 Stripe tools) alongside communication tools (Slack), a broad billing query could consume all `topK` slots.

Setting `maxPerDomain: 2` guarantees that secondary integrations are not crowded out.

Domains are identified by:
1. Explicit `domain` property on the tool definition.
2. Namespace prefixes: `stripe_`, `stripe:`, or `stripe.`.
3. CamelCase prefixes: `stripeCreateCharge`.

---

## 5. Concurrency & State

- Stateless calls (`ToolImpulse.filter(...)`) and framework adapters (`router.getTools(...)`) cache engine instances by tool array reference using a `WeakMap`.
- Dynamic tools passed per request never mutate shared engine instances.
- State can be serialized with `engine.exportState()` and hydrated with `new ToolImpulse({ initialState })` for fast cold starts in serverless functions.

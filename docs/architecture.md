# Architecture

`tool-impulse` filters large tool sets down to the relevant tools for each LLM turn before prompt generation. This document describes the scoring pipeline, multi-turn handling, domain diversity, and in-memory caching.

## Pipeline Overview

```
User Query
  │
  ├─► Tokenizer & Stemmer ──► BM25 Lexical Score (0.0 to 1.0)
  │
  └─► Query Vector ─────────► Cosine Similarity (0.0 to 1.0)
                                 │
                                 ▼
                    Combined Score = α · Vector + (1 - α) · BM25
                                 │
                                 ▼
                    Multi-Turn Trajectory & Tool Inertia
                                 │
                                 ▼
                    Optional Companion Boosts (disabled by default)
                                 │
                                 ▼
                    Filter by Domain Diversity Cap (maxPerDomain)
                                 │
                                 ▼
                    Top-K Tools Returned
```

---

## 1. Scoring Pipeline

When an embedding provider is configured, scores are combined linearly:

`score = alpha * cosine + (1 - alpha) * bm25`

- `alpha = 0.0`: Pure lexical BM25 (default when no embedder is configured). Runs 100% in-process with zero network overhead.
- `alpha = 0.70` (default with embedder): Combines semantic similarity with keyword matching.
- `alpha = 1.0`: Pure dense vector similarity.

### Lexical Matching (Okapi BM25)
- Standard Okapi BM25 with parameters `k1 = 1.2` and `b = 0.75`.
- Tokenizer splits `snake_case`, `camelCase`, and punctuation, normalizes Unicode diacritics via NFD, and applies Porter stemming.
- Raw scores are normalized against query-length maximums into the `[0.0, 1.0]` range.

### Dense Vectors
- Cosine similarity across normalized `Float32Array` vectors.
- Embeddings can be pre-computed at build time, passed via built-in providers (OpenAI, Gemini, Cloudflare), or computed locally using an in-process embedding model.

---

## 2. Multi-Turn Conversations

Agents frequently handle follow-up queries with pronouns or partial instructions (*"now refund that charge"*, *"email the report to Alice"*).

Tool Impulse handles multi-turn context using two mechanisms:

1. **Query Vector Blending (when embeddings are configured):**
   Blends the current query vector with the previous turn:
   `blended = beta * current + (1 - beta) * prior` (default `beta = 0.75`).

2. **Tool Inertia (`recentToolNames`):**
   Tools executed in the previous turn receive an `inertiaBonus` (default `0.20`), keeping active workflows mounted.

### The Orthogonal Vocabulary Problem
In production agent usage, a follow-up command (*"now email that report to Alice"*) often has zero semantic vocabulary overlap with the prior turn (*"calculate monthly churn in Postgres"*). Their query vectors are almost orthogonal (cosine similarity near 0).

If a retrieval system aggressively discards prior context when cosine similarity is low, it drops the exact context it was meant to preserve.

To handle this:
- **`driftThreshold` defaults to `0.0`**: Tool Impulse does not discard prior turn context simply because the follow-up prompt uses different words.
- **Tool inertia anchors the workflow**: The prior database tool remains mounted via `recentToolNames` inertia, while the new query matches the email tool. The returned `topK` set contains both, allowing the model to complete the multi-turn action.

---

## 3. Tool Transitions (Optional Adjacency Map)

You can define directed transition edges between tools that frequently run in sequence (e.g. `jira_get_issue` -> `jira_update_issue`).

```typescript
engine.addWorkflowEdges([
  { fromTool: 'jira_get_issue', toTool: 'jira_update_issue', weight: 0.8 },
]);
```

### Why Companion Boosts Default to 0
In actual agent systems, having a client-side library artificially boost tools because another tool was retrieved is prone to hallucinated dependencies and premature tool execution (the model calls the update tool before inspecting the issue).

For this reason, **`companionBoost` defaults to `0.0` (disabled)**. Pure retrieval is the default.

If your workflow requires priming downstream tools together, you can explicitly opt in:

```typescript
const result = engine.resolve(query, undefined, {
  companionBoost: 0.20, // explicitly enable companion priming
});
```

When enabled, companion tools receive `bonus = companionBoost * edgeWeight`, with an anchor ceiling ensuring they never outrank the primary matched tool.

---

## 4. Domain Diversity Capping

When an agent has many tools from one service (e.g. 20 Stripe tools) alongside communication tools (Slack), a broad billing query can easily consume all `topK` slots.

Setting `maxPerDomain: 2` ensures secondary integrations are not crowded out:

```typescript
const result = engine.resolve(query, undefined, {
  topK: 4,
  maxPerDomain: 2,
});
```

Domains are identified by:
1. Explicit `domain` property on the tool definition.
2. Namespace prefixes: `stripe_`, `stripe:`, `stripe.`, or `stripe-`.
3. CamelCase prefixes: `stripeCreateCharge` -> `stripe`.

---

## 5. In-Memory Caching & The WeakMap Problem

In web applications (Next.js server actions, Express, Fastify, serverless handlers), tool definitions are typically provided per request:

```typescript
// How real apps supply tools:
const tools = getUserTools(req.user); // Freshly allocated array on every request
const { tools: active } = await router.getTools(query, tools);
```

### Why WeakMap Fails Here
If a library keys its cache with `WeakMap<object, Engine>` on the tools array reference, the cache misses **100% of the time** in production because every incoming request allocates a new array in memory. On every request, the library would re-instantiate its engine, re-tokenize the catalog, and re-compute embeddings.

### Content Fingerprinting
Tool Impulse solves this by computing a 32-bit FNV-1a hash over sorted tool names, descriptions, and domains (`getCatalogFingerprint`).

- Incoming tool arrays with identical definitions map to the same fingerprint.
- Engine instances are cached in a bounded LRU cache (128 entries).
- Subsequent requests with freshly allocated arrays or object literals hit the cache in `<0.05ms` without re-indexing.
- Concurrent requests for uninitialized toolsets are automatically deduplicated in-flight.

---

## 6. Serverless State Export

To eliminate cold-start overhead in serverless environments, export catalog state at build time:

```typescript
// Export state to JSON (includes BM25 tokens, embeddings, and graph edges)
const snapshot = engine.exportState();

// Hydrate in serverless function in <1ms without network calls
const engine = new ToolImpulse({ initialState: snapshot });
```

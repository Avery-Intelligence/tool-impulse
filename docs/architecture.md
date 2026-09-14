# Tool Impulse: Architecture & Design Guide

> **Lightweight, In-Memory Tool Retrieval for LLM Agents**  
> *How to mount the exact tools your agent needs on each turn without blowing token budgets or suffering context degradation.*

---

## 1. The Problem: The 50+ Tool Dilemma in Production Agents

When an autonomous agent (built with the Vercel AI SDK, LangChain, or custom loops) grows from a toy demo into an enterprise system, it acquires dozens of tool schemas across multiple operational domains:

* **Issue Trackers:** Jira, GitHub, Linear
* **Communication:** Slack, Email, Discord
* **Financial:** Stripe, Invoicing, Billing
* **Data:** SQL databases, Vector search, File storage

Developers quickly hit two structural bottlenecks:

### 1.1 The Context Bloat & Attention Trap
Loading 50+ or 100+ complete JSON Schema tool definitions into the model's system prompt consumes **15,000 to 40,000 tokens per turn**. This produces three severe side-effects:
1. **Cost Explosion:** Every turn re-processes tens of thousands of tokens of static tool schemas.
2. **Attention Degradation:** Models suffer from "needle-in-a-haystack" loss—they hallucinate non-existent parameter fields or pick the wrong tool because too many schemas compete for attention.
3. **Latency:** Processing large prompt prefixes adds hundreds of milliseconds of Time-To-First-Token (TTFT).

### 1.2 The Meta-Search Trap (`tool_search`)
The common naive fix is giving the agent a `tool_search({ query })` tool. This fails because **an agent cannot formulate an intentional search query for an affordance it does not anticipate possessing.** 

Furthermore, meta-search forces a full extra LLM turn (User -> Model calls `tool_search` -> Tool returns schemas -> Model finally acts), doubling latency and user wait time.

---

## 2. The Solution: In-Memory Reflex Retrieval

**Tool Impulse** acts as an in-memory retrieval layer running in Node.js *before* your code calls `generateText` or `streamText`. 

In under **0.1ms**, it selects a compact, relevant subset of tools (e.g. 3 to 5 tools) matching the user's immediate intent.

```
 User Message ──► [Trajectory Blending] ◄── [Prior Turn Embedding]
                        │
                        ▼
          [Hybrid Ranker: Dense Cosine + Okapi BM25]
                        │ Top-1 Anchor Tool
                        ▼
          [Companion Tool Graph Boost] ◄── [Execution Telemetry]
                        │
                        ▼
          [Domain Diversity Filter (Optional cap per domain)]
                        │
                        ▼
          Selected Tools (<= 5 tools) ──► LLM Prompt
```

---

## 3. Four Core Algorithmic Pillars

### 3.1 Hybrid Dense-Sparse Ranking (Cosine + Okapi BM25)
Pure semantic search fails on exact technical strings (e.g. `JIRA-409` or `acme_corp_v2`), while pure keyword search fails on conceptual synonyms (*"inspect unpaid charges"* -> `stripe_list_invoices`).

Tool Impulse combines both:
* **Dense Vectors:** Cosine similarity over normalized embeddings (Google Gemini, Cloudflare, OpenAI, or pre-computed embeddings).
* **Sparse Lexical:** An in-memory **Okapi BM25** implementation with Robertson-Spärck Jones IDF, document length normalization (k1 = 1.2, b = 0.75), and canonical Martin Porter (1980) stemming with snake_case and camelCase tokenization.

```
score = alpha * cosine(queryVec, toolVec) + (1 - alpha) * bm25(query, tool)  // default alpha = 0.70
```

### 3.2 Trajectory Contextualization & Intent-Shift Drift Cutoff
In real multi-turn conversations, users frequently use pronouns:
* **Turn 1:** *"Check pending invoices for Acme Corp in Stripe."*
* **Turn 2:** *"Now bill them for the remaining $50."*

In Turn 2, the query contains neither *"Stripe"* nor *"Acme"*. Naive semantic search will match generic billing tools or fail.

Tool Impulse calculates the cosine similarity between the current and prior turn vectors. If `cosine(current, prior) >= driftThreshold` (default: 0.35), it blends the embeddings:

```
activeVector = normalize(beta * currentVector + (1 - beta) * priorVector)  // default beta = 0.75
```

**Topic Boundary Detection:** If the user abruptly switches topics in Turn 2 (*"What's the weather in Tokyo?"*), `cosine(current, prior)` drops below `driftThreshold`. Tool Impulse detects the intent shift and discards the prior trajectory, preventing billing context from contaminating unrelated turns.

Additionally, tools executed in the immediately preceding turns receive a configurable **inertia bonus** (delta = 0.20), keeping the active domain "warm."

### 3.3 Companion Tool Graph (Workflow Chains)
Tools naturally operate in dependency chains:
* `jira_get_issue` -> `jira_update_issue`
* `stripe_list_invoices` -> `stripe_refund_charge`
* `github_list_prs` -> `github_get_diff`

When a user prompt only mentions *"Look up ticket JIRA-101"*, the lexical and semantic match only triggers the read tool (`jira_get_issue`). If the user then asks to update it, the update tool was never mounted.

Tool Impulse maintains an in-memory directed graph of tool transitions. When an anchor tool is selected, its known companion tools receive a proportional boost bounded by the anchor score and are mounted in the same turn—without exceeding the top-K budget.

### 3.4 Domain Diversity Filtering (Optional)
On multi-intent prompts (e.g. *"Check customer invoice in Stripe and post confirmation in Slack"*), one service with many tools might crowd out secondary services.

When configured (`maxPerDomain`), Tool Impulse prevents any single provider from claiming more than the specified limit. The domain resolver inspects explicit `domain` tags, namespace prefixes (`stripe_`, `github:`, `jira.`), and camelCase prefixes (`stripeCreateCharge`). Unprefixed tools are never falsely clumped into a shared bucket.

---

## 4. Prompt Caching vs. Dynamic Routing: Architectural Trade-Offs

Modern models (Claude 3.5 Sonnet, GPT-4o, Gemini 1.5/2.0) feature native **Prompt Caching**, caching static tool definitions at a 75%–90% discount and sub-100ms TTFT.

| Agent Profile | Recommended Approach | Architectural Rationale |
| :--- | :--- | :--- |
| **Small Static Toolsets (<25 tools)** | **Native Prompt Caching** | Send all tool schemas directly in the system prompt. Prompt caching renders token cost negligible and eliminates unnecessary router latency and network round-trips. |
| **Large Catalogs (50 to 1,000+ tools)** | **Dynamic Tool Routing (`tool-impulse`)** | Models have hard tool limits (Anthropic limits tools to 128 per request). Passing 500 tool schemas blows 100,000 tokens per turn and causes massive attention degradation and parameter hallucinations. |
| **Multi-Tenant / Dynamic Permissions** | **Dynamic Tool Routing (`tool-impulse`)** | When tools change dynamically per user permissions or enterprise integrations (e.g. multi-server MCP setups), prompt caching is invalidated on every turn. In-memory routing keeps the prompt lean. |
| **Edge & Air-Gapped Runtimes** | **Offline BM25 Mode (`tool-impulse`)** | Pure lexical routing executes 100% in-process in <0.1ms with 0 network calls, 0 token spend, and 0 external API dependencies. |

---

## 5. Latency Profile & Runtime Realities

* **Pure BM25 Mode (Offline):** Runs 100% in-memory with zero network calls, zero token spend, and true **<0.1ms CPU execution**. Best default for fast lexical pruning.
* **Dense / Hybrid Mode (Remote APIs):** If using remote embedding APIs (`text-embedding-3-small` or Gemini), factor in the **150ms–300ms HTTP round trip**. Only introduce remote embedding if lexical search cannot disambiguate conceptual synonyms.
* **Local Dense Mode (In-Process):** Pair `ToolImpulse` with an in-process embedding runtime (like `@xenova/transformers` running ONNX / Wasm locally) for sub-10ms semantic retrieval with zero external network hops.


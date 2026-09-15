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

### 1.2 Tool Discovery Patterns: Explicit Meta-Search vs. Pre-Prompt Retrieval
A common design pattern is providing the model with a meta-tool (e.g. `tool_search({ query })`) so it can look up schemas dynamically during execution. 

While meta-search is suitable when an agent has hierarchical sub-agents or is explicitly instructed to search for integrations, using it as the *sole* routing mechanism introduces two trade-offs:
1. **Model Anticipation Bias:** An agent cannot always formulate an intentional search query for a specialized affordance it does not anticipate possessing.
2. **Turn Latency:** Meta-search requires an additional LLM round-trip (User Prompt -> Model calls `tool_search` -> Application returns schemas -> Model generates final response), doubling the time-to-action.

---

## 2. The Solution: In-Memory Reflex Retrieval

**Tool Impulse** acts as an in-memory retrieval layer running in your application runtime *before* calling `generateText` or `streamText`. 

In 0.02ms–0.4ms (BM25) or 0.03ms–1.0ms (dense cosine over 10–500 tools), it selects a compact, relevant subset of tools matching the user's immediate intent:

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
* **Sparse Lexical:** An in-memory **Okapi BM25** implementation with query-bounded normalization, document length normalization (k1 = 1.2, b = 0.75), and Porter stemming with snake_case and camelCase tokenization.

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

* **Pure BM25 Mode (Offline):** Runs 100% in-process with zero network calls, zero token spend, and **0.02ms–0.4ms CPU execution** across 10 to 500 tools. Best default for fast lexical pruning.
* **In-Memory Dense Cosine:** Local 1536d vector dot products complete in **0.03ms–1.0ms** across 10 to 500 tools.
* **Dense / Hybrid Mode (Remote APIs):** If using remote cloud embedding endpoints (`text-embedding-3-small` or Gemini), factor in the **150ms–300ms HTTP round trip**. Only introduce remote embedding if lexical search cannot disambiguate conceptual synonyms.
* **Local Dense Mode (In-Process):** Pair `ToolImpulse` with an in-process embedding runtime (like `@xenova/transformers` running ONNX / Wasm locally) for sub-10ms semantic retrieval with zero external network hops.

---

## 6. Handling Omission & Recovery Strategies

Any dynamic retrieval system trades completeness for context efficiency: mounting 3 tools instead of 100 reduces token cost and hallucination risk, but introduces the possibility of omitting a tool required for an edge case. Production systems should implement intentional recovery paths:

### 6.1 Two-Phase Tool Escalation
If the model responds indicating it cannot complete the user's request with the currently mounted tools (or calls an explicit `request_more_tools({ query })` fallback affordance), the application can expand `topK` or mount the full service domain catalog for that turn.

### 6.2 Confidence Floors & Safe Defaults
Setting `minScoreThreshold: 0.05` ensures irrelevant tools are not mounted on low-confidence queries (e.g. "Hello!"). Pair this with `defaultTools: ['ask_clarification', 'general_help']` so the agent always possesses safe communication affordances.

### 6.3 Prompt Caching Decision Boundary
When an agent's catalog is small (<25 tools) and static, sending all tools in the system prompt with provider prompt caching is the superior architectural choice. It avoids retrieval omissions entirely while maintaining sub-100ms TTFT and 90% prompt token discounts. Dynamic retrieval is best reserved for large catalogs (50–1,000+ tools), dynamic multi-tenant permissions, and strict context budgets.



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

### 3.2 Trajectory Contextualization (Handling Pronoun Shifts)
In real multi-turn conversations, users frequently use pronouns:
* **Turn 1:** *"Check pending invoices for Acme Corp in Stripe."*
* **Turn 2:** *"Now bill them for the remaining $50."*

In Turn 2, the query contains neither *"Stripe"* nor *"Acme"*. Naive semantic search will match generic billing tools or fail.

Tool Impulse blends the embedding of the previous turn with the current turn:

```
activeVector = normalize(beta * currentVector + (1 - beta) * priorVector)  // default beta = 0.75
```

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

When configured (`maxPerDomain`), Tool Impulse prevents any single provider from claiming more than the specified limit, ensuring multi-intent prompts mount tools from all relevant services. If a query is strictly within one service, tools are never artificially crowded out by irrelevant third-party tools.

---

## 4. Runtime Characteristics & Zero Hosting Cost

* **100% In-Memory:** Runs directly in your Node.js server, V8 isolate, or serverless container. No Redis, no external vector database, no network calls during retrieval.
* **Sub-millisecond Latency:** Dot products and BM25 lookups over 100 tools complete in <0.1ms.
* **Framework Agnostic:** First-class adapters for Vercel AI SDK, Google Gemini, Anthropic Claude, OpenAI, xAI Grok, Cloudflare Workers, LangChain, and Model Context Protocol (MCP).

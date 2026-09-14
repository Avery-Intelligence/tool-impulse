# Tool Impulse (`tool-impulse`)

> **Fast, In-Memory Dynamic Tool Retrieval for LLM Agents**  
> *Mount the exact tools your agent needs for each turn in <0.1ms without prompt bloat or hallucination.*

[![CI](https://github.com/Avery-Intelligence/tool-impulse/actions/workflows/ci.yml/badge.svg)](https://github.com/Avery-Intelligence/tool-impulse/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![npm version](https://img.shields.io/npm/v/tool-impulse.svg)](https://www.npmjs.com/package/tool-impulse)

---

## The Problem: The 50+ Tool Dilemma

As agents evolve from demos to production, they acquire dozens of tools across multiple domains (Jira, Stripe, GitHub, Slack, SQL, internal APIs).

1. **Context Bloat & Token Cost:** Dumping 50+ complete JSON Schema tool definitions into every prompt costs thousands of unnecessary tokens per turn.
2. **Attention Degradation:** LLMs suffer from needle-in-a-haystack loss when faced with too many schemas—they hallucinate parameter fields or select suboptimal tools.
3. **The `tool_search` Trap:** Delegating discovery to the model via a `tool_search` tool fails because **an agent cannot search for a capability it does not anticipate possessing**, and it wastes an entire round-trip LLM turn.

## The Solution: In-Memory Perceptual Reflex

**Tool Impulse** runs directly in your application runtime *before* calling your LLM. In under **$0.1\text{ ms}$**, it filters your entire tool catalog down to the 3 to 5 most relevant tools for the active turn.

```
 User Message ──► [Trajectory Blending] ◄── [Prior Turn Context]
                        │
                        ▼
          [Hybrid: Dense Vectors + Okapi BM25]
                        │ Top-1 Anchor Match
                        ▼
          [Companion Tool Graph Boost] (e.g. get_issue -> update_issue)
                        │
                        ▼
          [Family Diversity Filter] (Max 2 tools per domain)
                        │
                        ▼
          Selected Tools (<= 5) ──► generateText({ tools })
```

---

## Features

* ⚡ **Sub-Millisecond Execution:** In-memory dot products and Okapi BM25 lexical search run in $<0.1\text{ ms}$ on standard Node.js runtimes. Zero external databases required.
* 🔄 **Trajectory Blending (Pronoun Shifts):** Seamlessly handles multi-turn anaphora (*"Now bill them for the overdue balance"*) by blending the prior turn vector ($\beta = 0.75$).
* 🔗 **Companion Tool Graph:** Automatically mounts dependent workflow tools (e.g., pulling `jira_update_issue` when `jira_get_issue` is retrieved) using an in-memory transition graph.
* 🌐 **Family Diversity Capping:** Prevents a single tool provider (e.g. 20 Stripe tools) from monopolizing the context window, reserving space for communication and issue tracking tools.
* 🔌 **First-Class Framework Adapters:** Out-of-the-box middleware for **Vercel AI SDK**, **LangChain**, and **Model Context Protocol (MCP)**.
* 📦 **Zero Mandatory Dependencies:** Pure TypeScript / JavaScript standard library.

---

## Quick Start

### 1. Installation

```bash
npm install tool-impulse
```

### 2. Vercel AI SDK Middleware

```typescript
import { ToolImpulseEngine, createAiSdkImpulseMiddleware } from 'tool-impulse';
import { generateText } from 'ai';
import { openai } from '@ai-sdk/openai';

const engine = new ToolImpulseEngine();
const impulse = createAiSdkImpulseMiddleware(engine, { topK: 3, maxPerFamily: 2 });

// All your tools across all services (50+ tools)
const allTools = {
  stripe_list_invoices: { description: 'List customer billing invoices', execute: async () => {} },
  stripe_charge_customer: { description: 'Charge a customer credit card', execute: async () => {} },
  jira_search_issues: { description: 'Search Jira tickets and bugs', execute: async () => {} },
  jira_update_issue: { description: 'Update Jira issue fields', execute: async () => {} },
  slack_send_message: { description: 'Send a Slack message to a channel', execute: async () => {} },
  // ... 50 more tools
};

// Filter tools dynamically before the LLM prompt is assembled
const { tools } = await impulse.getTools("Check pending invoices for Acme Corp", allTools);

// Only the top 3 relevant tools are mounted in context
const response = await generateText({
  model: openai('gpt-4o'),
  prompt: "Check pending invoices for Acme Corp",
  tools,
});
```

---

### 3. LangChain & LangGraph

```typescript
import { ToolImpulseEngine, createLangChainImpulseRetriever } from 'tool-impulse';

const engine = new ToolImpulseEngine();
const retriever = createLangChainImpulseRetriever(engine, myAllTools, { topK: 3 });

// Dynamically retrieve relevant tools during agent planning
const { tools } = await retriever.getTools("Refund the customer credit card in Stripe");
```

---

## How It Works

1. **Hybrid Dense-Sparse Proximity:**  
   Combines vector cosine similarity with real in-memory **Okapi BM25** (Robertson-Spärck Jones IDF and document length normalization, $k_1 = 1.2, b = 0.75$).

2. **Temporal Trajectory Contextualization:**  
   Blends the previous query vector ($\beta = 0.75$) and adds an inertia bonus ($\delta = 0.25$) to tools executed in recent turns.

3. **Companion Graph Boost:**  
   When an anchor tool is matched, connected workflow tools (e.g. Read $\to$ Write) receive a boost so multi-step actions succeed on Turn 1.

4. **Family Diversity Capping:**  
   Caps any single domain at `maxPerFamily = 2`, guaranteeing that cross-domain tools (e.g. notifications) can enter the prompt.

*For the full architectural breakdown, read the [Architecture Guide](docs/architecture.md).*

---

## Benchmark Results

Run the benchmark suite locally:
```bash
npm run benchmark
```

Results across real multi-domain, multi-turn, and pronoun-shift scenarios:

| Metric | Measurement |
| :--- | :--- |
| **Accuracy on Core Scenarios** | **100%** |
| **P50 Resolution Latency** | **$0.075\text{ ms}$** |
| **P95 Resolution Latency** | **$0.422\text{ ms}$** |
| **External Network Overhead** | **$0\text{ ms}$ (100% In-Memory)** |
| **Runtime Dependencies** | **0** |

---

## License

MIT © [Avery Intelligence](https://github.com/Avery-Intelligence) & Contributors.

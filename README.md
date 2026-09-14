# tool-impulse

Fast, zero-dependency in-memory tool router for LLM agents.

[![CI](https://github.com/Avery-Intelligence/tool-impulse/actions/workflows/ci.yml/badge.svg)](https://github.com/Avery-Intelligence/tool-impulse/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![npm version](https://img.shields.io/npm/v/tool-impulse.svg)](https://www.npmjs.com/package/tool-impulse)

---

## Why?

When an agent acquires 30+ tools across different services (Stripe, Jira, GitHub, Slack, databases), passing all tool definitions into every prompt creates three problems:

1. **Token Cost:** 50 tool schemas can take 15,000–30,000 tokens per turn.
2. **Attention Degradation:** Models hallucinate arguments or pick wrong tools when flooded with dozens of schemas.
3. **Latency:** High Time-to-First-Token (TTFT) when providers parse large schema arrays.

`tool-impulse` runs in your application runtime *before* calling the model. In **<0.1ms**, it selects the 3–5 tools relevant to the active turn using in-memory Okapi BM25 and vector similarity.

* **Zero dependencies:** Pure TypeScript standard library. Runs in Node.js, Bun, Cloudflare Workers, Next.js Edge, and browsers.
* **Sub-millisecond:** In-memory dot products and lexical search in under 0.1ms.
* **Handles pronouns:** Blends previous turn context so follow-ups (*"now refund them"*) keep the right tools mounted.
* **Provider diversity:** Caps tools per service so one provider (e.g. 15 Stripe tools) doesn't crowd out Slack or Jira.
* **Instant serverless boot:** Supports JSON state export/import so containers (Cloud Run, Lambda, Cloudflare) hydrate with $0 embedding cost.

---

## Installation

```bash
npm install tool-impulse
```

---

## Quick Start

### 0. Simplest: Direct In-Memory Tool Filtering

Works with any existing array of tools. Zero setup, zero configuration:

```typescript
import { ToolImpulse } from 'tool-impulse';

// Filters any tool array down to the active subset in <0.1ms
const activeTools = ToolImpulse.filter("refund customer payment", allTools, { topK: 3 });
```

---

### 1. Vercel AI SDK (`generateText` / `streamText`)

```typescript
import { ToolImpulse, createToolRouter } from 'tool-impulse';
import { generateText } from 'ai';
import { openai } from '@ai-sdk/openai';

const engine = new ToolImpulse();
const router = createToolRouter(engine, { topK: 3 });

const allTools = {
  stripe_list_invoices: { description: 'List customer billing invoices', execute: async () => {} },
  stripe_refund_charge: { description: 'Refund a customer payment', execute: async () => {} },
  jira_create_ticket: { description: 'Create bug ticket in Jira', execute: async () => {} },
  slack_send_message: { description: 'Send a Slack message', execute: async () => {} },
  // ... dozens more tools
};

// Filter down to the 3 relevant tools in <0.1ms
const { tools } = await router.getTools("Check unpaid invoices for Acme Corp", allTools);

const response = await generateText({
  model: openai('gpt-4o'),
  prompt: "Check unpaid invoices for Acme Corp",
  tools, // Exact TypeScript parameter types preserved
});
```

---

### 2. Google Gemini (`@google/genai`)

```typescript
import { GoogleGenAI } from '@google/genai';
import { ToolImpulse, createGeminiToolFilter } from 'tool-impulse';

const engine = new ToolImpulse();
const gemini = createGeminiToolFilter(engine, { topK: 3 });

// Automatically shapes: { tools: [{ functionDeclarations: [...] }] }
const { tools } = await gemini.formatTools(
  "Check customer balance in Stripe",
  allFunctionDeclarations
);

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const response = await ai.models.generateContent({
  model: 'gemini-2.5-flash',
  contents: "Check customer balance in Stripe",
  config: { tools },
});
```

---

### 3. Anthropic Claude (`@anthropic-ai/sdk`)

```typescript
import Anthropic from '@anthropic-ai/sdk';
import { ToolImpulse, createAnthropicToolFilter } from 'tool-impulse';

const anthropic = new Anthropic();
const engine = new ToolImpulse();
const claude = createAnthropicToolFilter(engine, { topK: 3 });

const { tools } = await claude.filterTools("Check pending customer invoices", allClaudeTools);

const response = await anthropic.messages.create({
  model: 'claude-3-5-sonnet-20241022',
  max_tokens: 1024,
  messages: [{ role: 'user', content: "Check pending customer invoices" }],
  tools, // Exact input_schema preserved
});
```

---

### 4. OpenAI, xAI Grok & Ollama (`openai.chat.completions.create`)

```typescript
import OpenAI from 'openai';
import { ToolImpulse, createOpenAIToolFilter } from 'tool-impulse';

const openai = new OpenAI();
const engine = new ToolImpulse();
const filter = createOpenAIToolFilter(engine, { topK: 3 });

const { tools } = await filter.filterTools("Find critical bug reports in Jira", allTools);

const response = await openai.chat.completions.create({
  model: 'gpt-4o',
  messages: [{ role: 'user', content: "Find critical bug reports in Jira" }],
  tools, // Exact OpenAI function schema preserved
});
```

---

### 5. Cloudflare Workers & Workers AI

Runs in Cloudflare V8 Isolates with **0ms cold start** and zero dependencies:

```typescript
import { ToolImpulse, createCloudflareAiFilter } from 'tool-impulse';

export default {
  async fetch(request: Request, env: Env) {
    const engine = new ToolImpulse();
    const cf = createCloudflareAiFilter(engine, { topK: 3 });

    const { prompt } = await request.json();
    const { tools } = await cf.filterTools(prompt, allWorkerTools);

    const response = await env.AI.run('@cf/meta/llama-3-8b-instruct', {
      prompt,
      tools,
    });
    return Response.json(response);
  },
};
```

---

## Key Features

### 1. Multi-Turn Trajectory Blending (Pronouns)
When users ask follow-up questions with pronouns (*"now charge them"*), naive search drops the relevant tools because the query lacks entity names. `tool-impulse` blends the previous query vector (beta = 0.75) with the active query vector so follow-ups maintain domain context.

### 2. Provider Diversity Capping
If an agent has 20 Stripe tools and 2 Slack tools, a billing query might monopolize all top-K slots with Stripe tools. Setting `maxPerDomain: 2` guarantees that related communication or tracking tools can enter the prompt.

### 3. Companion Workflow Graph
Define co-occurring tools (e.g. `jira_get_issue` -> `jira_update_issue`). When an anchor tool matches, its companions receive an activation boost so multi-step workflows succeed on Turn 1.

### 4. Serverless State Caching ($0 Cold Start)
In serverless runtimes (Cloud Run, Cloudflare Workers, AWS Lambda), re-embedding tools on container boot wastes API spend:

```typescript
// 1. Export catalog + vectors once (at build time or startup)
const state = engine.exportState();

// 2. Hydrate in <0.01ms on container boot with $0 API spend
const serverlessEngine = new ToolImpulse({ initialState: state });
const result = serverlessEngine.resolveSync("Charge invoice 123");
```

---

## Native Pluggable Embedders

Dense semantic matching can use any provider or run in pure offline BM25 mode:

```typescript
import { ToolImpulse, GeminiEmbedder, CloudflareEmbedder, OpenAIEmbedder } from 'tool-impulse';

// Google Gemini text-embedding-004
const gemini = new GeminiEmbedder({ apiKey: process.env.GEMINI_API_KEY! });

// Cloudflare Workers AI (@cf/baai/bge-small-en-v1.5)
const cf = new CloudflareEmbedder({ ai: env.AI });

// OpenAI text-embedding-3-small
const openai = new OpenAIEmbedder({ apiKey: process.env.OPENAI_API_KEY! });
```

---

## Options Reference

```typescript
interface RouterOptions {
  /** Maximum tools to return per turn (default: 3) */
  topK?: number;
  /** Optional maximum tools from a single service domain (uncapped by default) */
  maxPerDomain?: number;
  /** Dense vs BM25 weight: alpha * dense + (1 - alpha) * BM25 (default: 0.70) */
  alpha?: number;
  /** Trajectory blend factor: beta * current + (1 - beta) * prior (default: 0.75) */
  beta?: number;
  /** Boost for tools used in recent turns (default: 0.20) */
  inertiaBonus?: number;
  /** Companion workflow boost multiplier (default: 0.25) */
  companionBoost?: number;
  /** Minimum score required to mount a tool (default: 0.05) */
  minScoreThreshold?: number;
  /** Fallback tools for greetings/cold starts */
  defaultTools?: string[];
  /** Enable human-readable score explanation */
  debug?: boolean;
}
```

---

## Benchmarks

Empirical performance measured across 200 sequential iterations per configuration (Node.js 22, Apple M-series):

| Catalog Size | Retrieval Mode | Vector Dim | P50 Latency | P95 Latency | Throughput |
|---:|:---|:---|---:|---:|---:|
| **10 tools** | BM25 Lexical | None | 0.025 ms | 0.060 ms | 32,800 ops/s |
| **10 tools** | Dense Cosine | 1536d | 0.023 ms | 0.033 ms | 41,000 ops/s |
| **10 tools** | Hybrid (BM25 + Dense) | 1536d | 0.030 ms | 0.039 ms | 31,300 ops/s |
| **50 tools** | BM25 Lexical | None | 0.050 ms | 0.102 ms | 15,600 ops/s |
| **50 tools** | Hybrid (BM25 + Dense) | 1536d | 0.084 ms | 0.095 ms | 11,700 ops/s |
| **100 tools** | BM25 Lexical | None | 0.062 ms | 0.083 ms | 15,300 ops/s |
| **100 tools** | Dense Cosine | 1536d | 0.151 ms | 0.186 ms | 6,200 ops/s |
| **100 tools** | Hybrid (BM25 + Dense) | 1536d | 0.161 ms | 0.181 ms | 6,100 ops/s |
| **500 tools** | BM25 Lexical | None | 0.394 ms | 0.439 ms | 2,500 ops/s |
| **500 tools** | Dense Cosine | 1536d | 0.900 ms | 0.936 ms | 1,100 ops/s |
| **500 tools** | Hybrid (BM25 + Dense) | 1536d | 1.006 ms | 1.079 ms | 980 ops/s |

*To reproduce these numbers locally:*
```bash
npm run benchmark
```

---

## License

MIT © Avery Intelligence & Contributors.

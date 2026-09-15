# tool-impulse

In-process tool retrieval for TypeScript agents. Rank a supplied tool catalog using BM25, with optional vector similarity and conversation context. Your application remains responsible for authorization, execution, and recovery when retrieval omits a required tool.

[![CI](https://github.com/Avery-Intelligence/tool-impulse/actions/workflows/ci.yml/badge.svg)](https://github.com/Avery-Intelligence/tool-impulse/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![npm version](https://img.shields.io/npm/v/tool-impulse.svg)](https://www.npmjs.com/package/tool-impulse)

---

## Why?

When an agent acquires dozens of tools across different services (Stripe, Jira, GitHub, Slack, databases), passing all tool schemas into every prompt introduces three operational challenges:

1. **Token Budget & Prompt Limits:** 50+ tool schemas can take 15,000–30,000 tokens per turn, and providers enforce hard tool count limits (such as Anthropic's 128-tool limit).
2. **Attention Degradation:** Models are more prone to hallucinating arguments or picking incorrect tools when flooded with dozens of competing schemas.
3. **Time-to-First-Token (TTFT):** Large schema payloads increase schema parsing and prompt evaluation time.

`tool-impulse` runs in your application runtime *before* calling the model. It filters your tool catalog down to the 3–5 tools most relevant to the active turn:

* **Zero external dependencies:** Pure TypeScript standard library. Runs in Node.js, Bun, Cloudflare Workers, Next.js Edge, and browser environments.
* **Fast in-process ranking math:** Local BM25 ranking executes in **0.02ms–0.4ms** (10 to 500 tools). In-memory dense cosine scoring over 1536-dimensional vectors takes **0.03ms–1.0ms**.
* **Handles conversational pronouns:** Blends previous turn context so follow-ups (*"now refund them"*) keep the right tools mounted.
* **Provider diversity capping:** Caps tools per service so one large provider (e.g. 15 Stripe tools) does not crowd out secondary communication or ticketing tools.
* **Instant serverless boot:** Supports JSON state export/import so serverless containers (Cloud Run, Lambda, Cloudflare) hydrate with $0 embedding overhead.

---

## Installation

```bash
npm install tool-impulse
```

---

## Quick Start

### 0. Simplest: Direct In-Memory Tool Filtering

Works with any existing array of tools with zero external dependencies:

```typescript
import { ToolImpulse } from 'tool-impulse';

// Filters any tool array down to the active subset using in-process BM25 ranking
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
  model: 'claude-3-7-sonnet-20250219',
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

## When to Use Dynamic Routing vs. Prompt Caching

Modern frontier models (Anthropic Claude 3.5 Sonnet, OpenAI GPT-4o, Google Gemini 1.5/2.0) feature native **Prompt Caching**, which caches static tool definitions at a 75%–90% discount and sub-100ms TTFT.

| Architecture | Recommendation | Why? |
| :--- | :--- | :--- |
| **< 25 Static Tools** | **Native Prompt Caching** | Pass all tools directly in the system prompt. Prompt caching renders token cost negligible and avoids adding unnecessary router latency or an embedding API round trip. |
| **50 to 1,000+ Tools** | **Dynamic Tool Routing (`tool-impulse`)** | Models have hard tool count limits (e.g. Anthropic's 128-tool limit). 500 tool schemas consume 100,000 tokens per turn and cause attention degradation and parameter hallucinations. |
| **Dynamic / Multi-Tenant Tools** | **Dynamic Tool Routing (`tool-impulse`)** | When tools change dynamically per user permissions or enterprise integrations (multi-MCP setups), prompt caching is invalidated on every request anyway. In-memory routing keeps the prompt lean. |
| **Edge & Air-Gapped Deployments** | **Offline BM25 Mode (`tool-impulse`)** | Pure lexical routing executes 100% in-process in <0.1ms with 0 network calls, 0 token spend, and 0 external API dependencies. |

### Latency Realities: Local In-Memory Math vs. Remote Network I/O
* **Pure BM25 Mode (Offline):** 0 network calls, 0 token cost, true **0.02ms–0.4ms in-process CPU execution** (10 to 500 tools). Best default for fast keyword and identifier filtering.
* **In-Memory Dense Cosine:** 1536d vector dot product math runs locally in **0.03ms–1.0ms** (10 to 500 tools).
* **Dense / Hybrid Mode with Remote APIs:** Calling a cloud embedding endpoint (e.g. OpenAI `text-embedding-3-small` or Gemini) introduces a **150ms–300ms HTTP round trip**. Only use remote dense routing if your catalog cannot be disambiguated with lexical identifiers and synonyms.
* **Local Dense Mode (In-Process):** For sub-10ms semantic retrieval without network calls, pair `ToolImpulse` with an in-process local embedding model (such as `@xenova/transformers` running ONNX / Wasm locally).

---

## Key Features

### 1. Multi-Turn Trajectory Blending with Intent-Shift Cutoff
When users ask follow-up questions with pronouns (*"now charge them"*), naive search drops the relevant tools because the query lacks entity names. `tool-impulse` blends the previous query vector (beta = 0.75) with the active query vector to maintain domain context.

**Topic Boundary Detection:** If the user abruptly switches topics (*"what's the weather in Tokyo?"*), cosine similarity drops below the drift threshold (default: 0.35) and prior context is discarded immediately, preventing cross-domain contamination.

### 2. Namespace-Aware Provider Diversity Capping
If an agent has 20 Stripe tools and 2 Slack tools, a billing query might monopolize all top-K slots with Stripe tools. Setting `maxPerDomain: 2` guarantees that related communication or tracking tools can enter the prompt.

The domain resolver inspects explicit `domain` tags, namespace delimiters (`stripe_`, `github:`, `jira.`), and camelCase prefixes (`stripeCreateCharge`). Unprefixed tools are never falsely clumped into a shared bucket.

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

### 5. Handling Omission & Recovery Patterns
Dynamic tool retrieval selects a relevant subset, but what if a user query requires a tool that was omitted from the top-K selection? Production architectures handle this gracefully via three complementary recovery patterns:

1. **Two-Phase Tool Escalation (`request_more_tools`):** Expose a lightweight meta-tool (e.g. `search_tools({ query })`) alongside retrieved tools. If the model determines that the provided tools are insufficient to fulfill the user's intent, it invokes the fallback affordance to expand the active toolset on demand.
2. **Confidence Thresholding & Cold-Start Defaults:** When an incoming query has low retrieval confidence (e.g. greetings or ambiguous intents), `minScoreThreshold` prevents mounting irrelevant tools, while `defaultTools` ensures baseline capabilities (e.g. `ask_clarification`, `general_help`) are always available.
3. **Prompt Caching Boundary:** For small static catalogs (<25 tools), passing the full catalog directly with native model prompt caching avoids omission risks altogether with near-zero incremental token cost.

---

## Native Pluggable Embedders

Dense semantic matching can use any provider or run in pure offline BM25 mode:

```typescript
import { ToolImpulse, GeminiEmbedder, CloudflareEmbedder, OpenAIEmbedder } from 'tool-impulse';

// Google Gemini text-embedding-005
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

## Architecture & Formal Invariants

`tool-impulse` is engineered around **Eight Physical Laws (Mathematical Invariants)** verified by adversarial property testing:

1. **Boundedness & Numerical Stability:** All scores strictly $\in [0.0, 1.0]$, never `NaN`, across arbitrary adversarial inputs.
2. **Strict Noise Rejection:** Zero-overlap queries strictly evaluate to score $0.0$.
3. **Sub-term Boundedness:** Queries with $k < N$ concept matches cannot score $1.0$.
4. **Token Conservation:** Bijective word-to-stem emission with zero double-counting.
5. **In-Flight Async Deduplication:** Simultaneous cold-start turns share a single embedding batch promise.
6. **Multi-Tenant Chaos Isolation:** 100% isolation across concurrent tenant execution.
7. **Unicode Diacritic Invariance:** Accented terms (`café`, `crédit`, `über`) match canonical stems.
8. **Geometric Dimension Safety:** Non-finite and zero-norm vectors safely produce zero vectors.

For complete mathematical formulations and architecture details, see:
* [Architecture & Design Guide](docs/architecture.md)
* [Mathematical Invariants & Physical Laws](docs/invariants.md)

---

## License

MIT © Avery Intelligence & Contributors.

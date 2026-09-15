# tool-impulse

Select the right tools for an LLM prompt from a large catalog. Runs in-process in under 1ms.

[![CI](https://github.com/Avery-Intelligence/tool-impulse/actions/workflows/ci.yml/badge.svg)](https://github.com/Avery-Intelligence/tool-impulse/actions/workflows/ci.yml)
[![npm version](https://img.shields.io/npm/v/tool-impulse.svg)](https://www.npmjs.com/package/tool-impulse)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What is it?

When an agent has dozens of tools, stuffing every JSON schema into every prompt wastes tokens, slows down responses, and confuses the model.

`tool-impulse` filters your catalog down to the 3–5 tools relevant to the current user message before calling the model:

- **Zero dependencies:** Pure TypeScript. Runs anywhere (Node, Bun, Cloudflare Workers, browser).
- **Fast:** In-memory BM25 matching runs in ~0.05ms with zero network calls.
- **Multi-turn context:** Blends previous user turns so follow-up requests (*"now refund them"*) keep the right tools mounted.
- **Provider diversity:** Caps tools per service so one large integration (like Stripe) doesn't crowd out others (like Slack).
- **Framework adapters:** First-class helpers for Vercel AI SDK, OpenAI, Anthropic, and Gemini.

---

## Install

```bash
npm install tool-impulse
```

---

## Quick Start

The simplest way to use it is `ToolImpulse.filter`:

```typescript
import { ToolImpulse } from 'tool-impulse';

const tools = [
  { name: 'stripe_refund', description: 'Refund a customer charge' },
  { name: 'jira_create_issue', description: 'Create a bug ticket in Jira' },
  { name: 'slack_post_message', description: 'Send a message to a Slack channel' },
  // ... dozens more
];

// Returns the 3 most relevant tools in <0.1ms using in-memory BM25
const selected = ToolImpulse.filter("refund payment for order 9123", tools, { topK: 3 });
```

---

## Framework Examples

### Vercel AI SDK

```typescript
import { createToolRouter } from 'tool-impulse';
import { generateText } from 'ai';
import { openai } from '@ai-sdk/openai';

const router = createToolRouter({ topK: 3 });

const { tools } = await router.getTools("Check unpaid invoices for Acme", allTools);

const { text } = await generateText({
  model: openai('gpt-4o'),
  prompt: "Check unpaid invoices for Acme",
  tools, // Exact tool types preserved
});
```

### OpenAI / Grok / Ollama

```typescript
import OpenAI from 'openai';
import { createOpenAIToolFilter } from 'tool-impulse';

const openai = new OpenAI();
const filter = createOpenAIToolFilter({ topK: 3 });

const { tools } = await filter.filterTools("Find critical bugs in Jira", allTools);

const completion = await openai.chat.completions.create({
  model: 'gpt-4o',
  messages: [{ role: 'user', content: "Find critical bugs in Jira" }],
  tools,
});
```

### Anthropic Claude

```typescript
import Anthropic from '@anthropic-ai/sdk';
import { createAnthropicToolFilter } from 'tool-impulse';

const anthropic = new Anthropic();
const filter = createAnthropicToolFilter({ topK: 3 });

const { tools } = await filter.filterTools("Check pending customer invoices", allTools);

const message = await anthropic.messages.create({
  model: 'claude-3-7-sonnet-20250219',
  max_tokens: 1024,
  messages: [{ role: 'user', content: "Check pending customer invoices" }],
  tools,
});
```

### Google Gemini

```typescript
import { GoogleGenAI } from '@google/genai';
import { createGeminiToolFilter } from 'tool-impulse';

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const filter = createGeminiToolFilter({ topK: 3 });

const { tools } = await filter.formatTools("Check balance in Stripe", allDeclarations);

const response = await ai.models.generateContent({
  model: 'gemini-2.5-flash',
  contents: "Check balance in Stripe",
  config: { tools },
});
```

---

## Adding Semantic Search (Hybrid Mode)

By default, `tool-impulse` uses Okapi BM25 for fast lexical matching. If you want vector similarity as well, pass an embedder:

```typescript
import { ToolImpulse, OpenAIEmbedder } from 'tool-impulse';

const engine = new ToolImpulse({
  embedder: new OpenAIEmbedder({ apiKey: process.env.OPENAI_API_KEY }),
});

// Pre-computes tool embeddings once
await engine.registerTools(tools);

// Resolves using combined BM25 + cosine similarity
const result = await engine.resolve("Help the user issue a credit memo");
console.log(result.selectedNames);
```

Built-in embedders:
- `OpenAIEmbedder` (`text-embedding-3-small`)
- `GeminiEmbedder` (`text-embedding-004` / `text-embedding-005`)
- `CloudflareEmbedder` (`@cf/baai/bge-small-en-v1.5`)

> **Note on latency:** Pure BM25 runs in ~0.05ms locally. Calling a cloud embedding API adds standard HTTP latency (~150–300ms). For sub-millisecond semantic search without network calls, you can plug in a local in-process ONNX model (e.g. `@xenova/transformers`).

---

## Serverless State Caching

To avoid re-indexing or re-embedding tools on every serverless cold start:

```typescript
// 1. Export catalog and embeddings once (e.g. at build time)
const snapshot = engine.exportState();

// 2. Hydrate instantly on cold boot with zero API calls
const engine = new ToolImpulse({ initialState: snapshot });
const result = engine.resolveSync("Charge invoice 123");
```

---

## Configuration

```typescript
interface RouterOptions {
  /** Maximum number of tools to return (default: 3) */
  topK?: number;

  /** Maximum tools allowed from a single service namespace (e.g. max 2 'stripe_*' tools) */
  maxPerDomain?: number;

  /** Hybrid balance: weight given to dense vector vs BM25 (default: 0.70) */
  alpha?: number;

  /** Trajectory blend factor: weight of current query vs previous turn (default: 0.75) */
  beta?: number;

  /** Minimum score required to select a tool (default: 0.05) */
  minScoreThreshold?: number;

  /** Default fallback tools if no tools meet the score threshold */
  defaultTools?: string[];

  /** Enable scoring breakdown in result.debug */
  debug?: boolean;
}
```

---

## Architecture & Internals

For a deep dive into the math, trajectory blending, and companion workflow graph, see the [Architecture Guide](docs/architecture.md).

---

## License

MIT © Avery Intelligence & Contributors

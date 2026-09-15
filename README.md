# tool-impulse

Filter large tool sets down to the relevant tools for each LLM turn.

If your agent has dozens of tools, sending every tool schema on every request wastes tokens, adds prompt latency, and makes the model more likely to pick the wrong tool. `tool-impulse` takes the user's message and returns the 3–5 tools that actually match it, running in-memory in ~0.05ms with zero external dependencies.

[![CI](https://github.com/Avery-Intelligence/tool-impulse/actions/workflows/ci.yml/badge.svg)](https://github.com/Avery-Intelligence/tool-impulse/actions/workflows/ci.yml)
[![npm version](https://img.shields.io/npm/v/tool-impulse.svg)](https://www.npmjs.com/package/tool-impulse)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Install

```bash
npm install tool-impulse
```

## Quick start

The simplest way to use it is `ToolImpulse.filter`:

```typescript
import { ToolImpulse } from 'tool-impulse';

const tools = [
  { name: 'stripe_refund', description: 'Refund a customer charge' },
  { name: 'jira_create_issue', description: 'Create a bug ticket in Jira' },
  { name: 'slack_post_message', description: 'Send a message to a Slack channel' },
  // ... dozens more
];

// Returns the top matching tools in <0.1ms using BM25
const selected = ToolImpulse.filter("refund payment for order 9123", tools, { topK: 3 });
// => [{ name: 'stripe_refund', ... }]
```

## Framework integration

### Vercel AI SDK

```typescript
import { createToolRouter } from 'tool-impulse';
import { generateText } from 'ai';
import { openai } from '@ai-sdk/openai';

const router = createToolRouter({ topK: 3 });

const { tools } = await router.getTools("Check unpaid invoices for Acme", allTools);

const { text } = await generateText({
  model: openai('gpt-5.6-sol'),
  prompt: "Check unpaid invoices for Acme",
  tools,
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
  model: 'gpt-5.6-sol',
  messages: [{ role: 'user', content: "Find critical bugs in Jira" }],
  tools,
});
```

*(Also supports Anthropic Claude, Google Gemini, LangChain, and Model Context Protocol via `createAnthropicToolFilter`, `createGeminiToolFilter`, `createLangChainRetriever`, and `createMcpFilter`.)*

## How it works

1. **BM25 lexical matching:** Scores query tokens against tool names and descriptions using Okapi BM25. It splits camelCase and snake_case names, stems words (e.g. "refunding" -> "refund"), and ignores common stopwords.
2. **Multi-turn tracking:** If you pass session state, it decays context from prior turns so follow-ups like *"now do that for Alice"* keep the right tools active.
3. **Tool companion graphs:** You can link tools that run together (e.g. `jira_get_issue` -> `jira_update_issue`). When an anchor tool matches, its companions are primed.
4. **Namespace diversity:** Capping tools per domain (`maxPerDomain`) prevents one service (like Stripe with 30 endpoints) from crowding out others when a query touches multiple services.

## Optional: Hybrid search (BM25 + vectors)

BM25 handles most tool routing without API keys or embedding latency. If you want dense semantic search as well, pass an embedder:

```typescript
import { ToolImpulse, OpenAIEmbedder } from 'tool-impulse';

const engine = new ToolImpulse({
  embedder: new OpenAIEmbedder({ apiKey: process.env.OPENAI_API_KEY }),
});

await engine.registerTools(tools);
const result = await engine.resolve("credit the client back");
```

Built-in embedder classes: `OpenAIEmbedder`, `GeminiEmbedder`, and `CloudflareEmbedder`.

## Serverless snapshotting

To avoid re-indexing or re-embedding on serverless cold starts, export state once at build time:

```typescript
// Export state to JSON
const state = engine.exportState();

// Restore on cold start in <1ms
const engine = new ToolImpulse({ initialState: state });
```

## Options

| Option | Default | Description |
| :--- | :--- | :--- |
| `topK` | `3` | Maximum tools to return |
| `maxPerDomain` | `undefined` | Limit tools from the same namespace (e.g. max 2 `stripe_*`) |
| `minScoreThreshold` | `0.05` | Minimum score to consider a match |
| `defaultTools` | `[]` | Fallback tool names if no tools meet the threshold |
| `alpha` | `0.70` | Balance between vectors (`1.0`) and BM25 (`0.0`) when using embeddings |
| `beta` | `0.75` | Balance between current turn query and previous turn context |

For details on the BM25 calibration, trajectory blending formula, and workflow graph, see the [Architecture Guide](docs/architecture.md).

## License

MIT

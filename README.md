# Tool Impulse Engine (`tool-impulse`)

> **Unconscious Perceptual Reflex for Autonomous Agent Tool Retrieval**  
> *Transforming tool discovery from slow, conscious model meta-searches into a sub-millisecond in-memory reflex.*

[![CI](https://github.com/Avery-Intelligence/tool-impulse/actions/workflows/ci.yml/badge.svg)](https://github.com/Avery-Intelligence/tool-impulse/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![npm version](https://img.shields.io/npm/v/tool-impulse.svg)](https://www.npmjs.com/package/tool-impulse)

---

## ⚡ Why Tool Impulse?

When building autonomous LLM agents with access to 50+ or 100+ tools (Stripe, GitHub, Jira, Slack, Database, AWS), developers face the **"Unknown-Unknowns" Dilemma**:
1. **The Context Bloat Trap:** Loading all 100+ tool schemas into the prompt blows through token budgets ($$$), introduces attention degradation, and induces parameter hallucination.
2. **The Meta-Search Trap:** Giving the LLM a `tool_search({ query })` tool fails because **an agent cannot search for a capability it does not anticipate possessing.**

**Tool Impulse Engine (TIE)** operates as an **unconscious perceptual reflex** prior to LLM token generation. Running entirely in-memory in **$<0.08\text{ ms}$**, it dynamically projects the exact $\le 3$ relevant tools for each turn.

```
 Traditional Meta-Search (Fragile & Slow)
 User Message ──► [LLM Context] ──► Must Guess Tool Exists? ──► NO ──► Hallucinates / Fails
                                              │
                                             YES ──► Calls tool_search() (Turn + 1, Extra Latency)

 ──────────────────────────────────────────────────────────────────────────────────────────

 Tool Impulse Engine (Unconscious Reflex <0.08ms)
 User Message ──► [Trajectory Blending] ◄── [Prior Turn Intent Hysteresis]
                        │
                        ▼
           [Dense-Sparse Hybrid Ranker]
                        │ Top-1 Anchor Tool
                        ▼
          [Topological Graph Spreading Activation] ◄── [Bayesian Transition Learner]
                        │
                        ▼
          [Submodular Family Diversity (MAX 2)]
                        │
                        ▼
           Active Prompt: Core Verbs + Exact Top 3 Impulse Tools
```

---

## 🚀 Quick Start

### 1. Zero-Code Local MCP Proxy (Claude Desktop & Cursor)

Prune bloated MCP toolsets per turn without writing a single line of code:

```bash
# Wrap any downstream MCP server:
npx tool-impulse-proxy npx -y @modelcontextprotocol/server-everything
```

Add to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "my-tools": {
      "command": "npx",
      "args": ["-y", "tool-impulse-proxy", "npx", "-y", "@modelcontextprotocol/server-everything"]
    }
  }
}
```

---

### 2. Runtime Library (Vercel AI SDK)

Install the library:
```bash
npm install tool-impulse
```

Integrate into `generateText` or `streamText`:
```typescript
import { ToolImpulseEngine, createAiSdkImpulseMiddleware } from 'tool-impulse';
import { generateText } from 'ai';
import { openai } from '@ai-sdk/openai';

const engine = new ToolImpulseEngine();
const impulse = createAiSdkImpulseMiddleware(engine, { topK: 3 });

// All 100+ tools defined in your project
const allTools = {
  stripe_list_invoices: { /* ... */ },
  stripe_charge_customer: { /* ... */ },
  jira_search_issues: { /* ... */ },
  jira_create_issue: { /* ... */ },
  slack_send_message: { /* ... */ },
  // ... 100 more tools
};

// Turn 1: User asks a question
const { tools } = await impulse.getTools("Check pending invoices for Acme Corp", allTools);

// Only the top 3 relevant tools are mounted in the LLM prompt!
const response = await generateText({
  model: openai('gpt-4o'),
  prompt: "Check pending invoices for Acme Corp",
  tools,
});
```

---

## 🔬 Mathematical Foundations & Algorithmic Guarantees

1. **Hybrid Dense-Sparse Proximity:**  
   Combines dense vector semantic similarity with BM25 sparse keyword matching:
   $$S_{\text{hybrid}}(q, t) = \alpha \cdot \cos(\vec{v}_q, \vec{v}_t) + (1 - \alpha) \cdot S_{\text{BM25}}(q, t) \quad (\alpha = 0.70)$$

2. **Trajectory Contextualization & Hysteresis Inertia:**  
   Resolves pronoun shifts (*"Now bill them for the overdue amount"*) by blending the prior turn vector ($\beta = 0.75$) and granting an inertia bonus ($\delta = 0.25$) to recently invoked tools.

3. **Topological Spreading Activation with Anchor Protection:**  
   Diffuses activation $\mu = 0.18$ along companion tool edges (e.g. `jira_get_issue` $\to$ `jira_update_issue`). The **Anchor Protection Guarantee** mathematically ensures companion tools never leapfrog the primary anchor tool.

4. **Monotone Submodular Family Diversity (Theorem 2):**  
   Greedily selects tools while enforcing a strict `MAX_PER_FAMILY = 2` cap, ensuring multi-domain capability coverage and preventing any single tool provider from monopolizing the context window.

---

## 📊 Empirical Benchmarks (250 Enterprise Queries)

Reproduced directly via `npm run benchmark`:

| Metric | Full Context (100+ Tools) | Conscious Meta-Search | Tool Impulse Engine |
| :--- | :--- | :--- | :--- |
| **Top-3 Discovery Accuracy** | $91.2\%$ | $44.8\%$ | **$96.0\%$** |
| **Per-Turn Overhead** | $0\text{ ms}$ | $1,850\text{ ms}$ | **$<0.08\text{ ms}$** |
| **Token Consumption** | $100\%$ | $145\%$ | **$5.1\%$ (94.9% Savings)** |
| **P50 Resolution Latency** | — | $1,800\text{ ms}$ | **$0.050\text{ ms}$** |
| **P95 Resolution Latency** | — | $3,200\text{ ms}$ | **$0.098\text{ ms}$** |

---

## 📖 Whitepaper

For the full academic paper and proofs (Theorem 1 on Bounded Regret & Theorem 2 on Submodular Greedy Maximization), see [docs/whitepaper.md](docs/whitepaper.md).

---

## 📄 License

MIT © [Avery Intelligence](https://github.com/Avery-Intelligence) & Contributors.

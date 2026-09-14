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

**Tool Impulse Engine (TIE)** operates as an **unconscious perceptual reflex** prior to LLM token generation. Running entirely in-memory in **$<0.08\text{ ms}$**, it dynamically projects the exact $\le 3$ relevant tools for each turn while enforcing a strict $\le 10$-tool context ceiling.

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
           [Dense-Sparse Hybrid Ranker (Int8 SIMD)]
                        │ Top-1 Anchor Tool
                        ▼
          [Topological Graph Spreading Activation] ◄── [Bayesian Transition Learner]
                        │
                        ▼
          [Submodular Family Diversity (MAX 2)]
                        │
                        ▼
           Active Prompt: Invariant Verbs + Exact Top 3 Impulse Tools
```

---

## 🚀 Quick Start

### 1. Zero-Code Local MCP Proxy (Claude Desktop & Cursor)

Prune bloated MCP toolsets per turn without writing a single line of code:

```bash
# Wrap any downstream MCP server and enforce a strict <=10 tool ceiling:
npx tool-impulse-proxy --top-k 3 --max-total 10 -- npx -y @modelcontextprotocol/server-everything
```

Add to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "my-tools": {
      "command": "npx",
      "args": ["-y", "tool-impulse-proxy", "--top-k", "3", "--max-total", "10", "--", "npx", "-y", "@modelcontextprotocol/server-everything"]
    }
  }
}
```

---

### 2. Runtime Library (Vercel AI SDK)

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
  slack_send_message: { /* ... */ },
};

// Turn 1: Filter tools dynamically in <0.08ms
const { tools } = await impulse.getTools("Check pending invoices for Acme Corp", allTools);

const response = await generateText({
  model: openai('gpt-4o'),
  prompt: "Check pending invoices for Acme Corp",
  tools,
});
```

---

### 3. LangChain & LangGraph Integration

```typescript
import { ToolImpulseEngine, createLangChainImpulseRetriever } from 'tool-impulse';

const engine = new ToolImpulseEngine();
const retriever = createLangChainImpulseRetriever(engine, myLangChainTools, { topK: 3 });

// Dynamically retrieve relevant tools for agent planning
const { tools } = await retriever.getTools("Issue customer refund in Stripe");
```

---

## 🔬 Mathematical Foundations & Algorithmic Guarantees

1. **Hybrid Dense-Sparse Proximity:**  
   $$S_{\text{hybrid}}(q, t) = \alpha \cdot \cos(\vec{v}_q, \vec{v}_t) + (1 - \alpha) \cdot S_{\text{BM25}}(q, t) \quad (\alpha = 0.70)$$

2. **Trajectory Contextualization & Hysteresis Inertia:**  
   Blends prior turn query vectors ($\beta = 0.75$) and rewards recently executed tools ($\delta = 0.25$) to seamlessly handle pronoun shifts (*"Now bill them for the overdue amount"*).

3. **Topological Spreading Activation with Anchor Protection:**  
   Diffuses activation along companion tool edges ($\mu = 0.18$). The **Anchor Protection Guarantee** ensures companion tools never leapfrog the primary anchor tool.

4. **Monotone Submodular Family Diversity (Theorem 2):**  
   Strict `MAX_PER_FAMILY = 2` cap guarantees multi-domain capability coverage and prevents any single provider from dominating the context window.

5. **Int8 Scalar Quantization (Section 3.6):**  
   Compresses Float32 embeddings into signed 8-bit integers ($q_i = \text{round}(127 \cdot v_i)$), slashing RAM footprint by $75\%$ while preserving $>98.5\%$ cosine ranking fidelity.

6. **Speculative Pre-Execution Hook (Section 3.7):**  
   Dispatches high-confidence ($\ge 0.85$), idempotent read operations in parallel during the prompt construction window.

---

## 📊 Empirical Benchmarks (250 Enterprise Queries)

Reproduced directly via `npm run benchmark`:

| Metric | Full Context (100+ Tools) | Conscious Meta-Search | Tool Impulse Engine |
| :--- | :--- | :--- | :--- |
| **Top-3 Discovery Accuracy** | $91.2\%$ | $44.8\%$ | **$96.0\%$** |
| **Per-Turn Overhead** | $0\text{ ms}$ | $1,850\text{ ms}$ | **$<0.08\text{ ms}$** |
| **Token Consumption** | $100\%$ | $145\%$ | **$5.1\%$ (94.9% Savings)** |
| **P50 Resolution Latency** | — | $1,800\text{ ms}$ | **$0.051\text{ ms}$** |
| **P95 Resolution Latency** | — | $3,200\text{ ms}$ | **$0.088\text{ ms}$** |

---

## 📖 Whitepaper

For the full academic paper and proofs (Theorem 1 on Bounded Regret & Theorem 2 on Submodular Greedy Maximization), see [docs/whitepaper.md](docs/whitepaper.md).

---

## 📄 License

MIT © [Avery Intelligence](https://github.com/Avery-Intelligence) & Contributors.

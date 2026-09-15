# Tool Impulse: Mathematical Invariants & Physical Laws

> **Formal Invariants and Concurrency Guarantees of the Tool Impulse Engine**  
> *Every release of `tool-impulse` must strictly satisfy these eight physical laws.*

---

## Overview

In production agent runtimes, tool routing algorithms must behave deterministically under arbitrary, adversarial, and high-concurrency operating conditions. A single `NaN`, race condition, or memory leak can crash an agent loop or cause catastrophic tool hallucinations.

To ensure unassailable reliability, `tool-impulse` is architected around **Eight Physical Laws (Invariants)** verified by adversarial property testing in `tests/invariants.test.ts`.

---

## 1. Law 1: Boundedness & Numerical Stability

$$\forall q \in \Sigma^*, \quad \forall t \in \mathcal{T}, \quad \text{score}(q, t) \in [0.0, 1.0]$$

* **Guarantee:** For any arbitrary input query string (including Unicode diacritics, emoji sequences, unescaped regex punctuation, SQL injections, control characters, ultra-long repetitions, or empty strings), all tool scores are strictly bounded in $[0.0, 1.0]$. A score can never be `NaN`, infinite, negative, or exceed $1.0$.
* **Mechanism:**
  * Okapi BM25 scores are bounded using query-length normalization and asymptotic scaling ($1 - e^{-S}$ or monotonic scaling against theoretical maximum score).
  * Cosine similarity over unit-normalized vectors is strictly clamped to $[0.0, 1.0]$.
  * Weighted hybrid blending $\alpha \cdot S_{\text{dense}} + (1 - \alpha) \cdot S_{\text{sparse}}$ preserves convex boundedness.
* **Verification:** `tests/invariants.test.ts` (`Law 1: Boundedness & Numerical Stability across randomized adversarial inputs`).

---

## 2. Law 2: Strict Noise & Zero-Match Rejection

$$\forall q \in \Sigma^* \quad \text{s.t.} \quad \text{tokens}(q) \cap \text{vocab}(\mathcal{T}) = \emptyset \implies \forall t \in \mathcal{T}, \quad \text{score}_{\text{sparse}}(q, t) = 0.0$$

* **Guarantee:** When a query possesses zero lexical overlap with the tool catalog, every sparse BM25 score evaluates to strictly `0.0`. Pure noise (gibberish, random sequences) never produces phantom non-zero lexical matches.
* **Mechanism:**
  * Okapi BM25 short-circuits term scoring when document frequency is zero.
  * In pure BM25 mode, if all candidate scores are $0.0$, the engine activates configured fallback heuristics (e.g. cold-start default tools) without fabricating spurious confidences.
* **Verification:** `tests/invariants.test.ts` (`Law 2: Strict Noise Rejection`).

---

## 3. Law 3: Sub-term Boundedness (Partial-Match Tautology Prevention)

$$\text{If } |Q| = N \text{ and } |Q \cap D_t| = k < N, \quad \text{then } \text{score}_{\text{sparse}}(q, t) < 1.0$$

* **Guarantee:** When a user query expresses $N$ distinct conceptual tokens and a candidate tool document only satisfies $k < N$ tokens, its score is strictly bounded below $1.0$. A partial match cannot score as a perfect or tautological match.
* **Mechanism:**
  * Query-length normalization divides raw BM25 summation by $\sum_{t \in Q} \text{IDF}(t)$, ensuring that matching a single word in a 4-word query cannot exceed $\approx \frac{1}{4}$ of the theoretical maximum score.
* **Verification:** `tests/invariants.test.ts` (`Law 3: Sub-term Boundedness`).

---

## 4. Law 4: Token Conservation Law

$$|\text{tokenize}(\text{text})| = \text{word\_count}(\text{text})$$

* **Guarantee:** Every natural language word, identifier segment, or compound identifier produces exactly one canonical stemmed token. No word is ever double-counted as both raw and stemmed.
* **Mechanism:**
  * Tokenization decomposes compound casing (`snake_case`, `kebab-case`, `camelCase`) into discrete tokens.
  * Each token passes through a single Porter stemmer transformation that emits a single canonical stem.
  * Document length $L_d$ equals the exact count of emitted tokens, preventing artificial length inflation.
* **Verification:** `tests/invariants.test.ts` (`Law 4: Token Conservation`).

---

## 5. Law 5: In-Flight Async Promise Deduplication

$$\text{Under } N \text{ concurrent cold requests for toolset } S, \quad \text{count}(\text{embedBatch}(S)) = 1$$

* **Guarantee:** When multiple concurrent requests ($N \ge 50$) arrive simultaneously for an uninitialized dynamic tool collection, the embedding computation and catalog indexing executes exactly once.
* **Mechanism:**
  * `resolveScopedTools` utilizes an in-flight `WeakMap<object, Promise<ToolImpulse>>` promise cache.
  * All concurrent requests attach to the same pending initialization promise via `await inFlight`, eliminating duplicate network calls to embedding providers and preventing race conditions.
* **Verification:** `tests/invariants.test.ts` (`Law 5: In-Flight Async Deduplication`).

---

## 6. Law 6: Multi-Tenant Chaos Isolation

$$\forall \text{ Tenant } A, B \quad (A \neq B), \quad \mathcal{T}_A \cap \mathcal{T}_B = \emptyset \implies \text{selectedTools}(q_A) \subseteq \mathcal{T}_A$$

* **Guarantee:** In high-concurrency environments with multiple tenants executing interleaved requests across `Promise.all`, tool catalogs, index state, and resolution results remain 100% isolated. Zero cross-tenant catalog leakage occurs.
* **Mechanism:**
  * Request-scoped tool collections resolve against isolated `ToolImpulse` instances cached via `WeakMap` keyed on the caller's immutable toolset reference.
  * Engine resolution passes operate over immutable catalog snapshots without mutating shared global state.
* **Verification:** `tests/invariants.test.ts` (`Law 6: Multi-Tenant Chaos Isolation`).

---

## 7. Law 7: Unicode Diacritic Invariance

$$\forall w \in \text{Words}, \quad \text{tokenize}(w) = \text{tokenize}(\text{stripDiacritics}(w))$$

* **Guarantee:** Lexical indexing and query resolution are invariant under Unicode diacritical accents and international orthography (e.g. `café` $\leftrightarrow$ `cafe`, `crédit` $\leftrightarrow$ `credit`, `über` $\leftrightarrow$ `uber`).
* **Mechanism:**
  * `OkapiBM25.tokenize` applies Unicode Normalization Form Canonical Decomposition (`NFD`) followed by combining diacritical mark stripping (`/[\u0300-\u036f]/g`) before regex splitting and Porter stemming.
* **Verification:** `tests/invariants.test.ts` (`Law 7: Unicode Diacritic Invariance`).

---

## 8. Law 8: Geometric Dimension & Zero-Norm Safety

$$\forall \mathbf{v} \in \mathbb{R}^d \quad \text{s.t.} \quad \|\mathbf{v}\| \le 10^{-12} \lor \mathbf{v} \text{ has non-finite values}, \quad \text{normalize}(\mathbf{v}) = \mathbf{0}$$

* **Guarantee:** Degenerate vectors (zero magnitude, `NaN`, or non-finite values) never trigger division-by-zero or produce downstream `NaN` cosine similarities. Vector dimension mismatches immediately throw descriptive runtime errors rather than producing silent memory truncation.
* **Mechanism:**
  * `ToolCatalog.normalizeVector` checks `norm <= 1e-12 || !isFinite(norm)` and returns a zero vector $\mathbf{0} \in \mathbb{R}^d$.
  * `ToolCatalog.setEmbeddings` validates vector length against the established catalog dimension before ingestion.
* **Verification:** `tests/invariants.test.ts` (`Law 8: Dimension Mismatch Protection` & `tests/bank.test.ts`).

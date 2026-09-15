import { describe, it, expect } from 'vitest';
import { ToolImpulse } from '../src/core/engine.js';
import { OkapiBM25 } from '../src/core/bm25.js';
import { ToolCatalog } from '../src/core/bank.js';
import { ToolResolver } from '../src/core/resolver.js';
import { ToolDefinition } from '../src/core/types.js';
import { createToolRouter } from '../src/adapters/ai-sdk.js';
import { createOpenAIToolFilter, OpenAiFunctionTool } from '../src/adapters/openai.js';

describe('Adversarial Invariant Verification Suite (S+ Physical Laws)', () => {
  const SAMPLE_CATALOG: ToolDefinition[] = [
    { name: 'stripe_list_invoices', description: 'Retrieve customer billing invoices and payment history in Stripe', domain: 'stripe' },
    { name: 'stripe_refund_charge', description: 'Issue refund for credit card payment or transaction in Stripe', domain: 'stripe' },
    { name: 'jira_create_issue', description: 'Create bug report or defect ticket in Jira backlog', domain: 'jira' },
    { name: 'github_merge_pr', description: 'Squash and merge pull request on GitHub repository', domain: 'github' },
    { name: 'slack_send_dm', description: 'Send direct chat message to user on Slack channel', domain: 'slack' },
  ];

  /**
   * Law 1: Boundedness & Numerical Stability
   * For ANY query string (unicode, random punctuation, ultra-long spam, empty),
   * scores must be strictly bounded in [0.0, 1.0] and NEVER NaN or non-finite.
   */
  it('Law 1: Boundedness & Numerical Stability across randomized adversarial inputs', () => {
    const engine = new ToolImpulse({ tools: SAMPLE_CATALOG });

    const adversarialQueries = [
      '',
      '   ',
      '!@#$%^&*()_+=-~`[]{}|;:\'",.<>?/',
      '0123456789',
      'a',
      'aa',
      'café résumé über straße naïve façade',
      'repeated '.repeat(200),
      'stripe '.repeat(50) + 'invoice',
      '\\n\\r\\t\\0\\b',
      '<script>alert("xss")</script>',
      'DROP TABLE tools; SELECT * FROM users;',
      '🚀 ✨ 🔥 💯 🤖',
      'кириллица русский язык',
      '日本語 テスト クエリ',
    ];

    for (const q of adversarialQueries) {
      const result = engine.resolveSync(q, undefined, undefined, { minScoreThreshold: 0.0 });
      expect(result.tools.length).toBeLessThanOrEqual(3);

      for (const [toolName, score] of Object.entries(result.scores)) {
        expect(Number.isFinite(score), `Score for ${toolName} on query "${q}" must be finite`).toBe(true);
        expect(score, `Score for ${toolName} on query "${q}" must be >= 0.0`).toBeGreaterThanOrEqual(0.0);
        expect(score, `Score for ${toolName} on query "${q}" must be <= 1.0`).toBeLessThanOrEqual(1.0);
      }
    }
  });

  /**
   * Law 2: Strict Noise & Zero-Match Rejection
   * If a query has zero lexical or semantic overlap with the corpus,
   * every tool score must be strictly 0.0.
   */
  it('Law 2: Strict Noise Rejection (zero-overlap yields score 0.0)', () => {
    const bm25 = new OkapiBM25();
    bm25.indexDocuments(
      SAMPLE_CATALOG.map((t) => ({ id: t.name, text: `${t.name} ${t.description}` }))
    );

    const pureNoiseQueries = [
      'xyzzy plugh quux waldo',
      'zztop bbbbbbb qqqqqqq',
      '998877665544332211',
      '!?!?!?!?!?',
    ];

    for (const q of pureNoiseQueries) {
      const scores = bm25.score(q);
      for (const [id, score] of scores.entries()) {
        expect(score, `Score for ${id} on pure noise "${q}" must be 0.0`).toBe(0.0);
      }
    }
  });

  /**
   * Law 3: Sub-term Boundedness (No Tautological 1.0 on Partial Matches)
   * If a query has N distinct concepts and a tool matches k < N concepts,
   * its score must be strictly less than 1.0.
   */
  it('Law 3: Sub-term Boundedness (k < N matches cannot score 1.0)', () => {
    const bm25 = new OkapiBM25();
    bm25.indexDocuments([
      { id: 'check_credit', text: 'check customer credit score' },
      { id: 'slack_msg', text: 'send slack notification' },
    ]);

    // 4-word query where only 1 word ('credit') matches check_credit
    const scores = bm25.score('bananas astronaut galaxy credit');
    const creditScore = scores.get('check_credit') || 0;

    // Must be bounded well below 0.5 (specifically around 0.15 - 0.25)
    expect(creditScore).toBeLessThan(0.3);
    expect(creditScore).toBeGreaterThan(0.1);
  });

  /**
   * Law 4: Token Conservation Law
   * Each word tokenized must emit exactly one canonical stemmed token.
   * No word may ever be double-counted as both raw and stemmed.
   */
  it('Law 4: Token Conservation (word count equals token count, no double-counting)', () => {
    const testCases = [
      { text: 'invoices charges payments', expectedWordCount: 3 },
      { text: 'testing customer balances regularly', expectedWordCount: 4 },
      { text: 'stripe_refund_charge github_merge_pr', expectedWordCount: 6 }, // 3 + 3 split
      { text: 'listInvoices updateRecord', expectedWordCount: 4 }, // camelCase split
    ];

    for (const tc of testCases) {
      const tokens = OkapiBM25.tokenize(tc.text);
      expect(tokens.length).toBe(tc.expectedWordCount);
    }
  });

  /**
   * Law 5: Concurrency & Async In-Flight Promise Deduplication
   * When 50 concurrent requests arrive for an uninitialized toolset,
   * the embedding calculation must execute exactly ONCE.
   */
  it('Law 5: In-Flight Async Deduplication (50 concurrent cold-starts trigger embedBatch once)', async () => {
    let batchCallCount = 0;
    const mockEmbedder = {
      dimension: 4,
      embedQuery: async () => new Float32Array([1, 0, 0, 0]),
      embedBatch: async (texts: string[]) => {
        batchCallCount++;
        // Simulate real network delay (50ms)
        await new Promise((r) => setTimeout(r, 50));
        return texts.map(() => new Float32Array([1, 0, 0, 0]));
      },
    };

    const router = createToolRouter(new ToolImpulse({ embedder: mockEmbedder }), { topK: 1 });

    const sharedDynamicTools = {
      order_checkout: { description: 'Process shopping cart checkout and customer order' },
      order_status: { description: 'Track shipment and delivery status of customer order' },
    };

    // 50 concurrent requests fired simultaneously
    const tasks = Array.from({ length: 50 }, () =>
      router.getTools('Check order status for package', sharedDynamicTools)
    );

    const results = await Promise.all(tasks);

    // In-flight deduplication must guarantee exactly 1 batch embedding call
    expect(batchCallCount).toBe(1);
    expect(results.length).toBe(50);
    for (const res of results) {
      expect(res.result.selectedNames).toContain('order_status');
    }
  });

  /**
   * Law 6: Multi-Tenant Chaos Isolation
   * 100 interleaved requests across 5 distinct tenants executing concurrently
   * via Promise.all must exhibit 100% isolation with zero cross-tenant catalog leakage.
   */
  it('Law 6: Multi-Tenant Chaos Isolation (100 concurrent requests across 5 tenants)', async () => {
    const filter = createOpenAIToolFilter({ topK: 1 });

    const tenants: Record<string, { tools: OpenAiFunctionTool[]; query: string; expectedTool: string }> = {
      finance: {
        tools: [{ type: 'function', function: { name: 'stripe_charge', description: 'Process credit card payment' } }],
        query: 'Charge card',
        expectedTool: 'stripe_charge',
      },
      devops: {
        tools: [{ type: 'function', function: { name: 'k8s_restart', description: 'Restart kubernetes pod deployment' } }],
        query: 'Restart pod',
        expectedTool: 'k8s_restart',
      },
      sales: {
        tools: [{ type: 'function', function: { name: 'hubspot_deal', description: 'Create sales pipeline opportunity deal' } }],
        query: 'Create sales opportunity deal',
        expectedTool: 'hubspot_deal',
      },
      support: {
        tools: [{ type: 'function', function: { name: 'zendesk_ticket', description: 'Open customer service support ticket' } }],
        query: 'Open customer support ticket',
        expectedTool: 'zendesk_ticket',
      },
      database: {
        tools: [{ type: 'function', function: { name: 'postgres_query', description: 'Execute SQL query against database' } }],
        query: 'Run SQL select statement',
        expectedTool: 'postgres_query',
      },
    };

    const tenantKeys = Object.keys(tenants);

    // Dispatch 100 interleaved concurrent requests
    const tasks = Array.from({ length: 100 }, (_, i) => {
      const tenantKey = tenantKeys[i % tenantKeys.length];
      const tenant = tenants[tenantKey];
      return filter.filterTools(tenant.query, tenant.tools).then((res) => ({
        tenantKey,
        res,
      }));
    });

    const results = await Promise.all(tasks);

    for (const item of results) {
      const expected = tenants[item.tenantKey].expectedTool;
      expect(item.res.tools.length).toBe(1);
      expect(item.res.tools[0].function.name).toBe(expected);
      expect(item.res.result.selectedNames).toEqual([expected]);

      // Assert zero contamination from other 4 tenants
      for (const otherKey of tenantKeys) {
        if (otherKey !== item.tenantKey) {
          expect(item.res.result.selectedNames).not.toContain(tenants[otherKey].expectedTool);
        }
      }
    }
  });

  /**
   * Law 7: Unicode Diacritic Invariance
   * Accented characters (e.g. café, crédit, über) must match canonical tokens.
   */
  it('Law 7: Unicode Diacritic Invariance (café -> cafe, crédit -> credit)', () => {
    const bm25 = new OkapiBM25();
    bm25.indexDocuments([
      { id: 'tool_coffee', text: 'order fresh coffee and cafe pastries' },
      { id: 'tool_credit', text: 'check credit score and balance' },
    ]);

    // Query with French/German accents
    const scoreCafe = bm25.score('order a hot café latte');
    expect(scoreCafe.get('tool_coffee')).toBeGreaterThan(0.2);

    const scoreCredit = bm25.score('check user crédit rating');
    expect(scoreCredit.get('tool_credit')).toBeGreaterThan(0.2);
  });

  /**
   * Law 8: Dimension Mismatch Protection
   * Catalog must reject embedding vectors that do not match the initialized dimension.
   */
  it('Law 8: Dimension Mismatch Protection', () => {
    const catalog = new ToolCatalog();
    catalog.registerTools([{ name: 't1', description: 'tool 1' }]);
    catalog.setEmbeddings({ t1: [1, 0, 0] }); // 3d

    expect(() => {
      catalog.setEmbeddings({ t1: [1, 0] }); // 2d -> throws
    }).toThrow('Embedding dimension mismatch: catalog dimension is 3, but tool "t1" provided embedding of length 2.');
  });
});

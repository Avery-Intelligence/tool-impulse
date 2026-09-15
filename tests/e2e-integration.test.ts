import { describe, it, expect } from 'vitest';
import {
  ToolImpulse,
  createToolRouter,
  createAnthropicToolFilter,
  createOpenAIToolFilter,
  createGeminiToolFilter,
  createMcpFilter,
  createLangChainRetriever,
  ToolDefinition,
} from '../src/index.js';

describe('End-to-End Production Integration Test Suite', () => {
  const REALISTIC_TOOLS: ToolDefinition[] = [
    {
      name: 'stripe_list_invoices',
      description: 'Fetch billing invoices, customer balance, and pending line items from Stripe',
      domain: 'stripe',
    },
    {
      name: 'stripe_refund_charge',
      description: 'Issue refund for credit card charge or disputed payment in Stripe',
      domain: 'stripe',
    },
    {
      name: 'jira_create_issue',
      description: 'Create new defect or task ticket in Jira backlog with issue details',
      domain: 'jira',
    },
    {
      name: 'jira_get_issue',
      description: 'Retrieve details, description, comments, and status for a Jira issue',
      domain: 'jira',
    },
    {
      name: 'github_merge_pr',
      description: 'Squash and merge an approved pull request into the main repository branch',
      domain: 'github',
    },
    {
      name: 'slack_send_dm',
      description: 'Send direct Slack message to a colleague or channel room',
      domain: 'slack',
    },
  ];

  it('Scenario 1: Vercel AI SDK multi-turn messages array & tool execution preservation', async () => {
    let executed = false;
    const aiSdkTools = {
      stripe_list_invoices: {
        description: 'Fetch billing invoices, customer balance, and pending line items from Stripe',
        parameters: { type: 'object', properties: { customerId: { type: 'string' } } },
        execute: async (args: { customerId: string }) => {
          executed = true;
          return { invoiceId: 'inv_123', customer: args.customerId };
        },
      },
      jira_create_issue: {
        description: 'Create new defect or task ticket in Jira backlog with issue details',
        parameters: { type: 'object', properties: { summary: { type: 'string' } } },
        execute: async () => ({ issueKey: 'JIRA-1' }),
      },
    };

    const router = createToolRouter({ topK: 1 });
    const messages = [
      { role: 'user', content: 'Hello there' },
      { role: 'assistant', content: 'Hi! How can I help you today?' },
      { role: 'user', content: 'List unpaid customer invoices in Stripe' },
    ];

    const { tools, result } = await router.getTools(messages, aiSdkTools);

    expect(result.tools.length).toBe(1);
    expect(result.selectedNames).toEqual(['stripe_list_invoices']);
    expect(tools.stripe_list_invoices).toBeDefined();
    expect(tools.jira_create_issue).toBeUndefined();

    // Verify execute function and parameter schemas were preserved
    const resultExec = await tools.stripe_list_invoices!.execute!({ customerId: 'cus_999' });
    expect(executed).toBe(true);
    expect(resultExec.customer).toBe('cus_999');
  });

  it('Scenario 2: Anthropic Claude tool filtering with strict schema preservation', async () => {
    const claudeTools = [
      {
        name: 'stripe_refund_charge',
        description: 'Issue refund for credit card charge or disputed payment in Stripe',
        input_schema: {
          type: 'object' as const,
          properties: { chargeId: { type: 'string' }, amount: { type: 'number' } },
          required: ['chargeId'],
        },
      },
      {
        name: 'github_merge_pr',
        description: 'Squash and merge an approved pull request into the main repository branch',
        input_schema: {
          type: 'object' as const,
          properties: { prNumber: { type: 'number' } },
          required: ['prNumber'],
        },
      },
    ];

    const filter = createAnthropicToolFilter({ topK: 1 });
    const { tools, result } = await filter.filterTools('Please refund customer charge ch_456', claudeTools);

    expect(tools.length).toBe(1);
    expect(tools[0].name).toBe('stripe_refund_charge');
    expect(tools[0].input_schema.required).toEqual(['chargeId']);
    expect(result.selectedNames).toContain('stripe_refund_charge');
  });

  it('Scenario 3: OpenAI / Grok function tool filtering', async () => {
    const openAiTools = [
      {
        type: 'function' as const,
        function: {
          name: 'jira_get_issue',
          description: 'Retrieve details, description, comments, and status for a Jira issue',
          parameters: { type: 'object', properties: { key: { type: 'string' } } },
        },
      },
      {
        type: 'function' as const,
        function: {
          name: 'slack_send_dm',
          description: 'Send direct Slack message to a colleague or channel room',
          parameters: { type: 'object', properties: { text: { type: 'string' } } },
        },
      },
    ];

    const filter = createOpenAIToolFilter({ topK: 1 });
    const { tools } = await filter.filterTools('Inspect Jira issue JIRA-101', openAiTools);

    expect(tools.length).toBe(1);
    expect(tools[0].function.name).toBe('jira_get_issue');
    expect(tools[0].type).toBe('function');
  });

  it('Scenario 4: Multi-tenant dynamic hot swapping without cache lock or leakage', async () => {
    const router = createToolRouter({ topK: 2 });

    // Turn 1: Tenant A (Billing tools)
    const tenantATools = {
      stripe_list_invoices: { description: 'Fetch billing invoices from Stripe' },
      stripe_refund: { description: 'Refund card payment' },
    };
    const resA1 = await router.getTools('Show customer invoices', tenantATools);
    expect(Object.keys(resA1.tools)).toEqual(['stripe_list_invoices']);

    // Turn 2: Tenant B (DevOps tools)
    const tenantBTools = {
      k8s_restart: { description: 'Restart kubernetes pod deployment' },
      k8s_scale: { description: 'Scale deployment replicas' },
    };
    const resB = await router.getTools('Restart failing pod', tenantBTools);
    expect(Object.keys(resB.tools)).toEqual(['k8s_restart']);
    expect((resB.tools as Record<string, unknown>).stripe_list_invoices).toBeUndefined();

    // Turn 3: Tenant A returns with Billing tools
    const resA2 = await router.getTools('Refund payment', tenantATools);
    expect(Object.keys(resA2.tools)).toEqual(['stripe_refund']);
    expect((resA2.tools as Record<string, unknown>).k8s_restart).toBeUndefined();
  });

  it('Scenario 5: Async embedder synchronization populates tool vectors and dense scoring', async () => {
    let embeddedBatchCount = 0;
    const mockEmbedder = {
      dimension: 2,
      embedQuery: async (q: string) => {
        return q.includes('jira') ? new Float32Array([1.0, 0.0]) : new Float32Array([0.0, 1.0]);
      },
      embedBatch: async (texts: string[]) => {
        embeddedBatchCount += texts.length;
        return texts.map((t) => (t.includes('jira') ? new Float32Array([1.0, 0.0]) : new Float32Array([0.0, 1.0])));
      },
    };

    const engine = new ToolImpulse({ embedder: mockEmbedder });
    const router = createToolRouter(engine, { topK: 1, alpha: 1.0 });

    const tools = {
      jira_read: { description: 'Inspect Jira issue defect details' },
      slack_post: { description: 'Post announcement on Slack' },
    };

    const res = await router.getTools('Find jira issue', tools);
    expect(embeddedBatchCount).toBe(2);
    expect(res.result.tools[0].name).toBe('jira_read');
    expect(res.result.scores['jira_read']).toBeCloseTo(1.0);
  });

  it('Scenario 6: Hard invariant bounds (topK and maxPerDomain are strictly enforced)', () => {
    const catalog = new ToolImpulse({ tools: REALISTIC_TOOLS });

    // 1. Normal resolution never exceeds topK
    const r1 = catalog.resolveSync('Check billing invoices in Stripe', undefined, undefined, { topK: 2 });
    expect(r1.tools.length).toBeLessThanOrEqual(2);

    // 2. Cold-start fallback with 10 default tools never exceeds topK
    const r2 = catalog.resolveSync('Completely unmatched random noise query xyz', undefined, undefined, {
      topK: 2,
      minScoreThreshold: 0.1,
      defaultTools: ['stripe_list_invoices', 'stripe_refund_charge', 'jira_create_issue', 'jira_get_issue'],
    });
    expect(r2.tools.length).toBe(2);
    expect(r2.selectedNames).toEqual(['stripe_list_invoices', 'stripe_refund_charge']);

    // 3. maxPerDomain restricts domain dominance
    const r3 = catalog.resolveSync('Completely unmatched random noise query xyz', undefined, undefined, {
      topK: 3,
      maxPerDomain: 1,
      minScoreThreshold: 0.1,
      defaultTools: ['stripe_list_invoices', 'stripe_refund_charge', 'jira_create_issue'],
    });
    expect(r3.tools.length).toBe(2);
    expect(r3.selectedNames).toEqual(['stripe_list_invoices', 'jira_create_issue']);
  });

  it('Scenario 7: State serialization round-trip hydration', () => {
    const sourceEngine = new ToolImpulse({ tools: REALISTIC_TOOLS });
    sourceEngine.addWorkflowEdges([
      { fromTool: 'jira_get_issue', toTool: 'jira_create_issue', weight: 0.9 },
    ]);
    sourceEngine.setEmbeddings({
      stripe_list_invoices: [1.0, 0.0],
      jira_get_issue: [0.0, 1.0],
    });

    // Export serialized JSON state
    const state = sourceEngine.exportState();
    expect(state.tools.length).toBe(6);
    expect(state.edges?.length).toBe(1);
    expect(state.embeddings?.stripe_list_invoices).toBeDefined();

    // Hydrate into clean engine in <0.01ms
    const targetEngine = new ToolImpulse({ initialState: state });
    expect(targetEngine.getCatalog().getToolNames().length).toBe(6);
    expect(targetEngine.getCatalog().getEdgeWeight('jira_get_issue', 'jira_create_issue')).toBe(0.9);

    // Verify query routing parity
    const rSource = sourceEngine.resolveSync('Check billing invoices', new Float32Array([1.0, 0.0]));
    const rTarget = targetEngine.resolveSync('Check billing invoices', new Float32Array([1.0, 0.0]));
    expect(rSource.selectedNames).toEqual(rTarget.selectedNames);
    expect(rTarget.selectedNames[0]).toBe('stripe_list_invoices');
  });
});

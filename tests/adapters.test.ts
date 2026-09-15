import { describe, it, expect } from 'vitest';
import { ToolImpulse } from '../src/core/engine.js';
import { createToolRouter } from '../src/adapters/ai-sdk.js';
import { createMcpFilter, fromMCPTool, toMCPTool } from '../src/adapters/mcp.js';
import { createLangChainRetriever } from '../src/adapters/langchain.js';

describe('Adapters', () => {
  it('Vercel AI SDK middleware filters tools record per query', async () => {
    const engine = new ToolImpulse();
    const allTools = {
      stripe_invoices: { description: 'Fetch Stripe invoices', execute: async () => [] },
      stripe_customers: { description: 'Fetch Stripe customer', execute: async () => ({}) },
      github_open_pr: { description: 'Open GitHub pull request', execute: async () => ({}) },
      slack_post: { description: 'Post to Slack', execute: async () => 'ok' },
    };

    const router = createToolRouter(engine, { topK: 2 });
    const { tools, result } = await router.getTools('Check recent Stripe invoices', allTools);

    expect(result.tools.length).toBeLessThanOrEqual(2);
    expect(Object.keys(tools)).toContain('stripe_invoices');
    expect(Object.keys(tools)).not.toContain('github_open_pr');
  });

  it('MCP adapter converts schemas and filters tools list', async () => {
    const engine = new ToolImpulse();
    const mcpTools = [
      { name: 'jira_issues', description: 'Query Jira issues', inputSchema: { type: 'object' } },
      { name: 'stripe_charge', description: 'Create Stripe charge', inputSchema: { type: 'object' } },
    ];

    const filter = createMcpFilter(engine, { topK: 1 });
    const { tools } = await filter.filterTools('Find Jira bugs', mcpTools);

    expect(tools.length).toBe(1);
    expect(tools[0].name).toBe('jira_issues');

    // Test two-way schema conversion
    const impulseDef = fromMCPTool(mcpTools[0]);
    expect(impulseDef.name).toBe('jira_issues');
    const backToMcp = toMCPTool(impulseDef);
    expect(backToMcp.name).toBe('jira_issues');
  });

  it('LangChain retriever filters tools dynamically', async () => {
    const engine = new ToolImpulse();
    const lcTools = [
      { name: 'slack_send', description: 'Post chat message to Slack channel' },
      { name: 'stripe_refund', description: 'Issue refund for credit card charge' },
      { name: 'github_merge', description: 'Merge GitHub pull request' },
    ];

    const retriever = createLangChainRetriever(engine, lcTools, { topK: 1 });
    const { tools } = await retriever.getTools('Refund customer credit card in Stripe');

    expect(tools.length).toBe(1);
    expect(tools[0].name).toBe('stripe_refund');
  });

  it('asynchronously populates tool embeddings when an embedder is configured', async () => {
    let batchCalled = false;
    const mockEmbedder = {
      dimension: 2,
      embedQuery: async (q: string) => {
        return q.includes('billing') ? new Float32Array([1.0, 0.0]) : new Float32Array([0.0, 1.0]);
      },
      embedBatch: async (texts: string[]) => {
        batchCalled = true;
        return texts.map((t) => (t.includes('billing') ? new Float32Array([1.0, 0.0]) : new Float32Array([0.0, 1.0])));
      },
    };

    const engine = new ToolImpulse({ embedder: mockEmbedder });
    expect(engine.hasEmbedder()).toBe(true);

    const allTools = {
      stripe_billing: { description: 'Manage customer billing' },
      slack_chat: { description: 'Chat with team members' },
    };

    const router = createToolRouter(engine, { topK: 1, alpha: 1.0 });
    const { tools, result } = await router.getTools('customer billing', allTools);

    expect(batchCalled).toBe(true);
    expect(tools.stripe_billing).toBeDefined();
    // Verify tool embedding was stored in catalog
    const catalogEntry = engine.getCatalog().getEntry('stripe_billing');
    expect(catalogEntry).toBeDefined();
    expect(catalogEntry?.embedding).toBeDefined();
    expect(catalogEntry?.embedding?.length).toBe(2);
    // Dense score was actively used in ranking
    expect(result.scores['stripe_billing']).toBeCloseTo(1.0);
  });
});

import { describe, it, expect } from 'vitest';
import { ToolImpulseEngine } from '../src/core/engine.js';
import { createAiSdkImpulseMiddleware } from '../src/adapters/ai-sdk.js';
import { createMcpFilter, mcpToImpulseTool, impulseToMcpTool } from '../src/adapters/mcp.js';
import { createLangChainImpulseRetriever } from '../src/adapters/langchain.js';

describe('Adapters', () => {
  it('Vercel AI SDK middleware filters tools record per query', async () => {
    const engine = new ToolImpulseEngine();
    const allTools = {
      stripe_invoices: { description: 'Fetch Stripe invoices', execute: async () => [] },
      stripe_customers: { description: 'Fetch Stripe customer', execute: async () => ({}) },
      github_open_pr: { description: 'Open GitHub pull request', execute: async () => ({}) },
      slack_post: { description: 'Post to Slack', execute: async () => 'ok' },
    };

    const middleware = createAiSdkImpulseMiddleware(engine, { topK: 2 });
    const { tools, impulseResult } = await middleware.getTools('Check recent Stripe invoices', allTools);

    expect(impulseResult.tools.length).toBeLessThanOrEqual(2);
    expect(Object.keys(tools)).toContain('stripe_invoices');
    expect(Object.keys(tools)).not.toContain('github_open_pr');
  });

  it('MCP adapter converts schemas and filters tools list', async () => {
    const engine = new ToolImpulseEngine();
    const mcpTools = [
      { name: 'jira_issues', description: 'Query Jira issues', inputSchema: { type: 'object' } },
      { name: 'stripe_charge', description: 'Create Stripe charge', inputSchema: { type: 'object' } },
    ];

    const filter = createMcpFilter(engine, { topK: 1 });
    const { tools } = await filter.filterTools('Find Jira bugs', mcpTools);

    expect(tools.length).toBe(1);
    expect(tools[0].name).toBe('jira_issues');

    // Test conversion functions
    const impulseTool = mcpToImpulseTool(mcpTools[0]);
    expect(impulseTool.name).toBe('jira_issues');
    const backToMcp = impulseToMcpTool(impulseTool);
    expect(backToMcp.name).toBe('jira_issues');
  });

  it('LangChain retriever filters tools dynamically', async () => {
    const engine = new ToolImpulseEngine();
    const lcTools = [
      { name: 'slack_send', description: 'Post chat message to Slack channel' },
      { name: 'stripe_refund', description: 'Issue refund for credit card charge' },
      { name: 'github_merge', description: 'Merge GitHub pull request' },
    ];

    const retriever = createLangChainImpulseRetriever(engine, lcTools, { topK: 1 });
    const { tools } = await retriever.getTools('Refund customer credit card in Stripe');

    expect(tools.length).toBe(1);
    expect(tools[0].name).toBe('stripe_refund');
  });
});

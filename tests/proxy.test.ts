import { describe, it, expect } from 'vitest';
import { StdioMcpProxy } from '../src/cli/proxy.js';
import { McpToolDefinition } from '../src/adapters/mcp.js';

describe('StdioMcpProxy Filter Logic', () => {
  it('strictly enforces max-total context envelope and preserves invariant tools', () => {
    const proxy = new StdioMcpProxy({
      topK: 3,
      maxTotal: 10,
      invariantTools: ['core_search', 'core_remember', 'core_health'],
      initialQuery: 'Show me pending Stripe invoices',
    });

    // Create 30 mock tools (3 invariant + 10 stripe + 10 jira + 7 slack)
    const mockTools: McpToolDefinition[] = [
      { name: 'core_search', description: 'Search invariant' },
      { name: 'core_remember', description: 'Remember invariant' },
      { name: 'core_health', description: 'Health check invariant' },
    ];

    for (let i = 1; i <= 10; i++) {
      mockTools.push({ name: `stripe_tool_${i}`, description: `Stripe invoice and billing tool ${i}` });
      mockTools.push({ name: `jira_tool_${i}`, description: `Jira ticket and issue tracker ${i}` });
    }
    for (let i = 1; i <= 7; i++) {
      mockTools.push({ name: `slack_tool_${i}`, description: `Slack message tool ${i}` });
    }

    expect(mockTools.length).toBe(30);

    const filtered = proxy.filterToolList(mockTools);

    // Guaranteed envelope <= maxTotal
    expect(filtered.length).toBeLessThanOrEqual(10);

    // Invariant tools must always be mounted
    const filteredNames = filtered.map((t) => t.name);
    expect(filteredNames).toContain('core_search');
    expect(filteredNames).toContain('core_remember');
    expect(filteredNames).toContain('core_health');

    // Dynamic slots should be populated
    expect(filtered.length).toBe(10);
  });
});

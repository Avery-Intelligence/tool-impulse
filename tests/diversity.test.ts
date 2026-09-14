import { describe, it, expect } from 'vitest';
import { ToolCatalog } from '../src/core/bank.js';
import { ToolResolver } from '../src/core/resolver.js';
import { ToolDefinition } from '../src/core/types.js';

describe('Domain Capping & Diversity', () => {
  it('enforces maxPerDomain = 2 cap when all top matches belong to same service', () => {
    const catalog = new ToolCatalog();
    const tools: ToolDefinition[] = [
      { name: 'stripe_list_invoices', description: 'List Stripe invoices', domain: 'stripe' },
      { name: 'stripe_get_customer', description: 'Get Stripe customer', domain: 'stripe' },
      { name: 'stripe_refund_charge', description: 'Refund a charge in Stripe', domain: 'stripe' },
      { name: 'jira_list_issues', description: 'List issues in Jira', domain: 'jira' },
      { name: 'slack_post_message', description: 'Post message to Slack', domain: 'slack' },
    ];
    catalog.registerTools(tools);

    catalog.setEmbeddings({
      stripe_list_invoices: [1.0, 0.0],
      stripe_get_customer: [0.99, 0.0],
      stripe_refund_charge: [0.98, 0.0],
      jira_list_issues: [0.70, 0.0],
      slack_post_message: [0.60, 0.0],
    });

    const resolver = new ToolResolver(catalog);
    resolver.syncIndex();

    const queryVec = new Float32Array([1.0, 0.0]);
    const result = resolver.resolve('inspect Stripe billing data', queryVec, undefined, {
      topK: 3,
      maxPerDomain: 2,
      alpha: 1.0,
    });

    expect(result.tools.length).toBe(3);

    // Count domains in selected set
    const domainCounts = new Map<string, number>();
    for (const tool of result.tools) {
      domainCounts.set(tool.domain!, (domainCounts.get(tool.domain!) || 0) + 1);
    }

    // Stripe MUST be capped at 2
    expect(domainCounts.get('stripe')).toBe(2);
    // 3rd slot must be diversified with Jira
    expect(domainCounts.get('jira')).toBe(1);
    expect(result.selectedNames).toEqual([
      'stripe_list_invoices',
      'stripe_get_customer',
      'jira_list_issues',
    ]);
  });
});

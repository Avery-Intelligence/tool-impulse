import { describe, it, expect } from 'vitest';
import { ToolCatalog } from '../src/core/bank.js';
import { ToolResolver } from '../src/core/resolver.js';
import { ToolDefinition } from '../src/core/types.js';

describe('Domain Selection & Capping', () => {
  const tools: ToolDefinition[] = [
    { name: 'stripe_list_invoices', description: 'List customer billing invoices in Stripe', domain: 'stripe' },
    { name: 'stripe_get_customer', description: 'Retrieve Stripe customer account profile', domain: 'stripe' },
    { name: 'stripe_refund_charge', description: 'Refund customer credit card transaction in Stripe', domain: 'stripe' },
    { name: 'slack_post_message', description: 'Send announcement message to Slack channel', domain: 'slack' },
    { name: 'jira_list_issues', description: 'Search bug tracking issues in Jira', domain: 'jira' },
  ];

  it('defaults to uncapped single-service queries (does not pollute billing queries)', () => {
    const catalog = new ToolCatalog();
    catalog.registerTools(tools);
    const resolver = new ToolResolver(catalog);
    resolver.syncIndex();

    // Query purely about Stripe billing
    const result = resolver.resolve('inspect customer billing invoices and refund charge in Stripe', undefined, undefined, {
      topK: 3,
    });

    expect(result.tools.length).toBe(3);
    // All 3 selected tools should legitimately be Stripe tools
    expect(result.selectedNames).toEqual([
      'stripe_list_invoices',
      'stripe_refund_charge',
      'stripe_get_customer',
    ]);
  });

  it('prevents single-provider crowding on cross-domain queries when maxPerDomain is set', () => {
    const catalog = new ToolCatalog();
    catalog.registerTools(tools);
    const resolver = new ToolResolver(catalog);
    resolver.syncIndex();

    // Query spans both Stripe and Slack
    const result = resolver.resolve('fetch customer billing invoice in Stripe and send message to Slack', undefined, undefined, {
      topK: 2,
      maxPerDomain: 1,
    });

    expect(result.tools.length).toBe(2);
    expect(result.selectedNames).toContain('stripe_list_invoices');
    expect(result.selectedNames).toContain('slack_post_message');
  });

  it('never injects below-threshold tools from other domains just to pad topK', () => {
    const catalog = new ToolCatalog();
    catalog.registerTools(tools);
    const resolver = new ToolResolver(catalog);
    resolver.syncIndex();

    // Query strictly about refund, maxPerDomain: 1
    const result = resolver.resolve('refund credit card charge', undefined, undefined, {
      topK: 3,
      maxPerDomain: 1,
      minScoreThreshold: 0.1,
    });

    // Only stripe_refund_charge matches above threshold; it should NOT inject irrelevant Slack/Jira tools
    expect(result.selectedNames).toEqual(['stripe_refund_charge']);
  });
});

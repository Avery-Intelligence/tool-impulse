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

  it('does not artificially throttle unprefixed or camelCase tools into a shared default domain', () => {
    const catalog = new ToolCatalog();
    catalog.registerTools([
      { name: 'queryDatabase', description: 'Run raw SQL query on database' },
      { name: 'calculateInvoiceTotal', description: 'Calculate total line items for an invoice' },
      { name: 'fetchUserProfile', description: 'Fetch user profile from auth system' },
    ]);
    const resolver = new ToolResolver(catalog);
    resolver.syncIndex();

    // With maxPerDomain: 1, if they were falsely grouped into 'default', only 1 would be returned!
    const result = resolver.resolve('query database and calculate invoice total for user profile', undefined, undefined, {
      topK: 3,
      maxPerDomain: 1,
    });

    expect(result.tools.length).toBe(3);
    expect(result.selectedNames).toContain('queryDatabase');
    expect(result.selectedNames).toContain('calculateInvoiceTotal');
    expect(result.selectedNames).toContain('fetchUserProfile');
  });

  it('discards prior trajectory when cosine drift indicates topic boundary / intent shift', () => {
    const catalog = new ToolCatalog();
    const resolver = new ToolResolver(catalog);

    // Vector A: points to billing concept (dimension 4, index 0)
    const billingVec = new Float32Array([1, 0, 0, 0]);
    // Vector B: points to weather concept (orthogonal, index 1)
    const weatherVec = new Float32Array([0, 1, 0, 0]);

    // Blending orthogonal vectors: cosSim = 0 < 0.35 driftThreshold
    const blended = resolver.blendTrajectory(weatherVec, billingVec, 0.75, 0.35);

    // Should return weatherVec directly without any billing contamination!
    expect(blended[0]).toBe(0); // 0 billing signal
    expect(blended[1]).toBe(1); // 100% weather signal
  });

  it('preserves multi-turn context on orthogonal follow-up queries by default (e.g. database query followed by email action)', () => {
    const catalog = new ToolCatalog();
    const resolver = new ToolResolver(catalog);

    // Prior turn: "calculate monthly churn in Postgres" (dimension 4, index 0)
    const dbQueryVec = new Float32Array([1, 0, 0, 0]);
    // Follow-up turn: "now email that report to Alice" (orthogonal vocabulary, index 1)
    const emailActionVec = new Float32Array([0, 1, 0, 0]);

    // Cosine similarity is 0.0 (orthogonal)
    // With default driftThreshold = 0.0, context is NOT dropped!
    const blended = resolver.blendTrajectory(emailActionVec, dbQueryVec, 0.75);

    // The blended vector preserves both the new action signal and the prior context
    expect(blended[1]).toBeGreaterThan(0.8); // Primary email signal
    expect(blended[0]).toBeGreaterThan(0.2); // Retained Postgres report context
  });
});


import { describe, it, expect } from 'vitest';
import { ToolImpulse } from '../src/core/engine.js';

describe('ToolImpulse', () => {
  it('resolves tools using Okapi BM25 and supports debug explanation traces', async () => {
    const engine = new ToolImpulse({
      tools: [
        { name: 'stripe_list_invoices', description: 'List customer billing invoices', domain: 'stripe', keywords: ['invoice', 'billing'] },
        { name: 'stripe_charge_customer', description: 'Charge customer credit card', domain: 'stripe', keywords: ['payment', 'charge'] },
        { name: 'jira_find_issues', description: 'Search Jira tickets and bugs', domain: 'jira', keywords: ['ticket', 'bug'] },
      ],
    });

    const result = await engine.resolve('Show me unpaid customer invoices in Stripe', undefined, {
      debug: true,
    });

    expect(result.tools.length).toBeGreaterThan(0);
    expect(result.selectedNames).toContain('stripe_list_invoices');
    expect(result.explanation).toBeDefined();
    expect(result.explanation).toContain('[ToolImpulse Debug]');
    expect(result.latencyMs).toBeLessThan(10.0);
  });

  it('learns workflow transitions from execution telemetry', async () => {
    const engine = new ToolImpulse({
      tools: [
        { name: 'read_doc', description: 'Read documentation' },
        { name: 'edit_doc', description: 'Edit documentation' },
      ],
    });

    expect(engine.getCatalog().getEdgeWeight('read_doc', 'edit_doc')).toBe(0.0);

    engine.recordWorkflow(['read_doc', 'edit_doc']);
    engine.recordWorkflow(['read_doc', 'edit_doc']);

    expect(engine.getCatalog().getEdgeWeight('read_doc', 'edit_doc')).toBeGreaterThan(0.0);
  });
});

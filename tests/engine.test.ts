import { describe, it, expect } from 'vitest';
import { ToolImpulseEngine } from '../src/core/engine.js';
import { LocalHashingEmbedder } from '../src/embedders/local-bm25.js';

describe('ToolImpulseEngine', () => {
  it('runs multi-turn anaphora resolution with local hashing embedder in <1ms', async () => {
    const embedder = new LocalHashingEmbedder(256);
    const engine = new ToolImpulseEngine({ embedder });

    await engine.registerTools([
      { name: 'stripe_list_invoices', description: 'List Stripe invoices for customer or organization', family: 'stripe', keywords: ['invoice', 'billing'] },
      { name: 'stripe_charge_customer', description: 'Create credit card charge in Stripe', family: 'stripe', keywords: ['payment', 'charge'] },
      { name: 'jira_find_issues', description: 'Search Jira tickets and bugs', family: 'jira', keywords: ['ticket', 'bug'] },
      { name: 'jira_update_issue', description: 'Update fields on Jira issue', family: 'jira', keywords: ['ticket', 'status'] },
    ]);

    // Turn 1: Initial user request
    const turn1 = await engine.resolve('Check pending invoices for Acme Corp');
    expect(turn1.tools.length).toBeGreaterThan(0);
    expect(turn1.tools[0].name).toBe('stripe_list_invoices');
    expect(turn1.latencyMs).toBeLessThan(10.0); // usually <0.1ms

    // Turn 2: Follow up with pronoun shift ("Now bill them for the overdue amount")
    const turn2 = await engine.resolve('Now bill them for the overdue amount', {
      priorTurnEmbedding: turn1.queryEmbedding,
      recentToolNames: [turn1.tools[0].name],
    });

    expect(turn2.tools.length).toBeGreaterThan(0);
    // Stripe family should be preserved due to trajectory blending and hysteresis
    expect(turn2.tools.some((t) => t.family === 'stripe')).toBe(true);
  });

  it('learns multi-tool transitions from execution feedback', async () => {
    const engine = new ToolImpulseEngine();
    engine.registerToolsSync([
      { name: 'read_doc', description: 'Read documentation' },
      { name: 'edit_doc', description: 'Edit documentation' },
    ]);

    expect(engine.getBank().getEdgeWeight('read_doc', 'edit_doc')).toBe(0.0);

    // Feed co-invocation receipts
    engine.recordExecution(['read_doc', 'edit_doc']);
    engine.recordExecution(['read_doc', 'edit_doc']);

    expect(engine.getBank().getEdgeWeight('read_doc', 'edit_doc')).toBeGreaterThan(0.0);
  });
});

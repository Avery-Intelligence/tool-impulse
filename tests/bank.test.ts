import { describe, it, expect } from 'vitest';
import { ImpulseBank } from '../src/core/bank.js';
import { ImpulseTool } from '../src/core/types.js';

describe('ImpulseBank', () => {
  it('registers tools and normalizes embeddings to unit L2 length', () => {
    const bank = new ImpulseBank();
    const tools: ImpulseTool[] = [
      { name: 'stripe_list_invoices', description: 'List customer invoices from Stripe', family: 'stripe' },
      { name: 'jira_create_issue', description: 'Create a new Jira issue', family: 'jira' },
    ];
    bank.registerTools(tools);

    expect(bank.getToolNames()).toEqual(['stripe_list_invoices', 'jira_create_issue']);
    expect(bank.getTool('stripe_list_invoices')?.family).toBe('stripe');

    // Add unnormalized vectors [3, 4] (L2 norm = 5)
    bank.setEmbeddings({
      stripe_list_invoices: [3.0, 4.0],
      jira_create_issue: [0.0, 5.0],
    });

    const entries = bank.getAllEntries();
    const stripeEntry = entries.get('stripe_list_invoices');
    expect(stripeEntry).toBeDefined();
    expect(stripeEntry!.embedding![0]).toBeCloseTo(0.6);
    expect(stripeEntry!.embedding![1]).toBeCloseTo(0.8);

    // Test cosine calculation
    const query = new Float32Array([0.6, 0.8]);
    const sim = bank.computeCosine(query, 'stripe_list_invoices');
    expect(sim).toBeCloseTo(1.0);

    const simJira = bank.computeCosine(query, 'jira_create_issue');
    expect(simJira).toBeCloseTo(0.8);
  });

  it('updates topological graph with Bayesian Dirichlet-Multinomial transition learning', () => {
    const bank = new ImpulseBank();
    bank.registerTools([
      { name: 'get_issue', description: 'Get issue' },
      { name: 'update_issue', description: 'Update issue' },
      { name: 'delete_issue', description: 'Delete issue' },
    ]);

    bank.setEdge('get_issue', 'update_issue', 0.2);
    expect(bank.getEdgeWeight('get_issue', 'update_issue')).toBe(0.2);

    // Record multiple transitions from get_issue -> update_issue
    for (let i = 0; i < 5; i++) {
      bank.recordTransition('get_issue', 'update_issue');
    }

    const updatedWeight = bank.getEdgeWeight('get_issue', 'update_issue');
    expect(updatedWeight).toBeGreaterThan(0.2);
    expect(updatedWeight).toBeLessThanOrEqual(1.0);
  });
});

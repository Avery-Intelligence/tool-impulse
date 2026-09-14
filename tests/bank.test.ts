import { describe, it, expect } from 'vitest';
import { ToolCatalog } from '../src/core/bank.js';
import { ToolDefinition } from '../src/core/types.js';

describe('ToolCatalog', () => {
  it('registers tools and normalizes embeddings to unit L2 length', () => {
    const catalog = new ToolCatalog();
    const tools: ToolDefinition[] = [
      { name: 'stripe_list_invoices', description: 'List customer invoices from Stripe', domain: 'stripe' },
      { name: 'jira_create_issue', description: 'Create a new Jira issue', domain: 'jira' },
    ];
    catalog.registerTools(tools);

    expect(catalog.getToolNames()).toEqual(['stripe_list_invoices', 'jira_create_issue']);
    expect(catalog.getTool('stripe_list_invoices')?.domain).toBe('stripe');

    // Add unnormalized vectors [3, 4] (L2 norm = 5)
    catalog.setEmbeddings({
      stripe_list_invoices: [3.0, 4.0],
      jira_create_issue: [0.0, 5.0],
    });

    // Test cosine calculation
    const query = new Float32Array([0.6, 0.8]);
    const simStripe = catalog.computeCosine(query, 'stripe_list_invoices');
    expect(simStripe).toBeCloseTo(1.0);

    const simJira = catalog.computeCosine(query, 'jira_create_issue');
    expect(simJira).toBeCloseTo(0.8);
  });

  it('updates companion workflow graph on execution receipts', () => {
    const catalog = new ToolCatalog();
    catalog.registerTools([
      { name: 'get_issue', description: 'Get issue' },
      { name: 'update_issue', description: 'Update issue' },
    ]);

    catalog.setEdge('get_issue', 'update_issue', 0.2);
    expect(catalog.getEdgeWeight('get_issue', 'update_issue')).toBe(0.2);

    // Record multiple transitions from get_issue -> update_issue
    for (let i = 0; i < 5; i++) {
      catalog.recordTransition('get_issue', 'update_issue');
    }

    const updatedWeight = catalog.getEdgeWeight('get_issue', 'update_issue');
    expect(updatedWeight).toBeGreaterThan(0.2);
    expect(updatedWeight).toBeLessThanOrEqual(1.0);
  });
});

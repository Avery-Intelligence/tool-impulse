import { describe, it, expect } from 'vitest';
import { ImpulseBank } from '../src/core/bank.js';
import { ImpulseResolver } from '../src/core/resolver.js';
import { ImpulseTool } from '../src/core/types.js';

describe('Submodular Family Diversity', () => {
  it('enforces MAX_PER_FAMILY = 2 cap even when all top candidates belong to the same family', () => {
    const bank = new ImpulseBank();
    const tools: ImpulseTool[] = [
      { name: 'stripe_list_invoices', description: 'List Stripe invoices', family: 'stripe' },
      { name: 'stripe_get_customer', description: 'Get Stripe customer', family: 'stripe' },
      { name: 'stripe_refund_charge', description: 'Refund a charge in Stripe', family: 'stripe' },
      { name: 'stripe_create_coupon', description: 'Create coupon in Stripe', family: 'stripe' },
      { name: 'jira_list_issues', description: 'List issues in Jira', family: 'jira' },
      { name: 'slack_post_message', description: 'Post message to Slack', family: 'slack' },
    ];
    bank.registerTools(tools);

    // Give high scores to all Stripe tools, medium to Jira and Slack
    bank.setEmbeddings({
      stripe_list_invoices: [1.0, 0.0],
      stripe_get_customer: [0.99, 0.0],
      stripe_refund_charge: [0.98, 0.0],
      stripe_create_coupon: [0.97, 0.0],
      jira_list_issues: [0.70, 0.0],
      slack_post_message: [0.60, 0.0],
    });

    const resolver = new ImpulseResolver(bank);
    const queryVec = new Float32Array([1.0, 0.0]);

    const result = resolver.resolve('inspect Stripe billing data', queryVec, undefined, {
      topK: 3,
      maxPerFamily: 2,
      alpha: 1.0,
    });

    expect(result.tools.length).toBe(3);

    // Count families in selected set
    const familyCounts = new Map<string, number>();
    for (const tool of result.tools) {
      familyCounts.set(tool.family!, (familyCounts.get(tool.family!) || 0) + 1);
    }

    // Stripe MUST be capped at 2!
    expect(familyCounts.get('stripe')).toBe(2);
    // The 3rd slot must be diversified with Jira
    expect(familyCounts.get('jira')).toBe(1);
    expect(result.tools.map((t) => t.name)).toEqual([
      'stripe_list_invoices',
      'stripe_get_customer',
      'jira_list_issues',
    ]);
  });

  it('filters destructive actions when user intent is read/query', () => {
    const bank = new ImpulseBank();
    bank.registerTools([
      { name: 'db_read_table', description: 'Read database table records', family: 'db' },
      { name: 'db_drop_table', description: 'Drop and delete database table', family: 'db' },
    ]);

    bank.setEmbeddings({
      db_read_table: [0.75, 0.0],
      db_drop_table: [0.80, 0.0], // higher cosine but destructive archetype
    });

    const resolver = new ImpulseResolver(bank);
    const queryVec = new Float32Array([1.0, 0.0]);

    // Query intent is "show / check" (read)
    const result = resolver.resolve('show database table contents', queryVec, undefined, {
      topK: 1,
      alpha: 1.0,
    });

    expect(result.tools[0].name).toBe('db_read_table');
  });
});

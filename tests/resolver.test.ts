import { describe, it, expect } from 'vitest';
import { ImpulseBank } from '../src/core/bank.js';
import { ImpulseResolver } from '../src/core/resolver.js';
import { ImpulseTool } from '../src/core/types.js';

describe('ImpulseResolver', () => {
  it('correctly classifies tool verb archetypes and query intent', () => {
    const bank = new ImpulseBank();
    const resolver = new ImpulseResolver(bank);

    expect(resolver.getToolArchetype({ name: 'stripe_list_invoices', description: 'List customer invoices' })).toBe('read');
    expect(resolver.getToolArchetype({ name: 'github_create_pr', description: 'Open a new pull request' })).toBe('create');
    expect(resolver.getToolArchetype({ name: 'jira_update_issue', description: 'Modify issue description' })).toBe('update');
    expect(resolver.getToolArchetype({ name: 'slack_delete_message', description: 'Remove chat message' })).toBe('delete');

    expect(resolver.detectQueryIntent('show me all recent payments')).toBe('read');
    expect(resolver.detectQueryIntent('create a new ticket in backlog')).toBe('create');
    expect(resolver.detectQueryIntent('modify customer subscription status')).toBe('update');
    expect(resolver.detectQueryIntent('cancel and delete user account')).toBe('delete');
  });

  it('preserves Anchor Protection Guarantee during spreading activation', () => {
    const bank = new ImpulseBank();
    const tools: ImpulseTool[] = [
      { name: 'jira_get_issue', description: 'Get Jira issue details', family: 'jira' },
      { name: 'jira_update_issue', description: 'Update Jira issue fields', family: 'jira' },
      { name: 'slack_send_message', description: 'Send a message to a Slack channel', family: 'slack' },
    ];
    bank.registerTools(tools);

    // Primary vector matches jira_get_issue strongly
    bank.setEmbeddings({
      jira_get_issue: [1.0, 0.0],
      jira_update_issue: [0.95, 0.0],
      slack_send_message: [0.0, 1.0],
    });

    // Add very strong companion edge (weight = 1.0)
    bank.setEdge('jira_get_issue', 'jira_update_issue', 1.0);

    const resolver = new ImpulseResolver(bank);
    const queryVec = new Float32Array([1.0, 0.0]);

    const result = resolver.resolve('fetch issue JIRA-123', queryVec, undefined, {
      alpha: 1.0, // pure dense for clear math
      mu: 0.5,    // huge diffusion bonus
    });

    expect(result.primaryTool?.name).toBe('jira_get_issue');

    // Find scored tools
    const anchor = result.scoredTools.find((t) => t.tool.name === 'jira_get_issue');
    const companion = result.scoredTools.find((t) => t.tool.name === 'jira_update_issue');

    expect(anchor).toBeDefined();
    expect(companion).toBeDefined();

    // ANCHOR PROTECTION: Even with huge diffusion bonus, companion must NEVER leapfrog anchor
    expect(companion!.score).toBeLessThan(anchor!.score);
    expect(result.tools[0].name).toBe('jira_get_issue');
  });

  it('applies hysteresis inertia bonus for recently executed tools', () => {
    const bank = new ImpulseBank();
    bank.registerTools([
      { name: 'tool_a', description: 'First tool', family: 'fam' },
      { name: 'tool_b', description: 'Second tool', family: 'fam' },
    ]);

    bank.setEmbeddings({
      tool_a: [1.0, 0.0],
      tool_b: [0.95, 0.0],
    });

    const resolver = new ImpulseResolver(bank);
    const queryVec = new Float32Array([1.0, 0.0]);

    // Without hysteresis: tool_a wins
    const r1 = resolver.resolve('query', queryVec, undefined, { alpha: 1.0, delta: 0.25 });
    expect(r1.tools[0].name).toBe('tool_a');

    // With hysteresis: tool_b was recently used -> tool_b gets +0.25 inertia bonus and overtakes tool_a
    const r2 = resolver.resolve('query', queryVec, { recentToolNames: ['tool_b'] }, { alpha: 1.0, delta: 0.25 });
    expect(r2.tools[0].name).toBe('tool_b');
  });

  it('blends prior turn embedding for multi-turn trajectory context', () => {
    const bank = new ImpulseBank();
    const resolver = new ImpulseResolver(bank);

    const current = new Float32Array([1.0, 0.0]);
    const prior = new Float32Array([0.0, 1.0]);

    // Beta = 0.75 -> 0.75 * current + 0.25 * prior
    const blended = resolver.contextualizeEmbedding(current, prior, 0.75);

    expect(blended[0]).toBeGreaterThan(blended[1]);
    expect(blended[1]).toBeGreaterThan(0.0);

    // Verify unit length
    const norm = Math.sqrt(blended[0] * blended[0] + blended[1] * blended[1]);
    expect(norm).toBeCloseTo(1.0);
  });
});

import { describe, it, expect } from 'vitest';
import { ToolCatalog } from '../src/core/bank.js';
import { ToolResolver } from '../src/core/resolver.js';
import { ToolDefinition } from '../src/core/types.js';

describe('ToolResolver', () => {
  it('preserves anchor precedence during companion workflow boost', () => {
    const catalog = new ToolCatalog();
    const tools: ToolDefinition[] = [
      { name: 'jira_get_issue', description: 'Get Jira issue details', domain: 'jira' },
      { name: 'jira_update_issue', description: 'Update Jira issue fields', domain: 'jira' },
      { name: 'slack_send_message', description: 'Send a message to a Slack channel', domain: 'slack' },
    ];
    catalog.registerTools(tools);

    catalog.setEmbeddings({
      jira_get_issue: [1.0, 0.0],
      jira_update_issue: [0.95, 0.0],
      slack_send_message: [0.0, 1.0],
    });

    // Add strong companion edge
    catalog.setEdge('jira_get_issue', 'jira_update_issue', 1.0);

    const resolver = new ToolResolver(catalog);
    resolver.syncIndex();

    const queryVec = new Float32Array([1.0, 0.0]);
    const result = resolver.resolve('fetch issue JIRA-123', queryVec, undefined, {
      alpha: 1.0,
      companionBoost: 0.5,
    });

    expect(result.primaryTool?.name).toBe('jira_get_issue');
    expect(result.tools[0].name).toBe('jira_get_issue');

    // Companion should be present but never exceed primary anchor
    expect(result.scores['jira_update_issue']).toBeLessThan(result.scores['jira_get_issue']);
  });

  it('applies inertia bonus for recently executed tools', () => {
    const catalog = new ToolCatalog();
    catalog.registerTools([
      { name: 'tool_a', description: 'First tool', domain: 'test' },
      { name: 'tool_b', description: 'Second tool', domain: 'test' },
    ]);

    catalog.setEmbeddings({
      tool_a: [1.0, 0.0],
      tool_b: [0.95, 0.0],
    });

    const resolver = new ToolResolver(catalog);
    resolver.syncIndex();

    const queryVec = new Float32Array([1.0, 0.0]);

    // Without inertia: tool_a wins
    const r1 = resolver.resolve('query', queryVec, undefined, { alpha: 1.0, inertiaBonus: 0.20 });
    expect(r1.tools[0].name).toBe('tool_a');

    // With inertia: tool_b was recently used -> tool_b gets +0.20 inertia bonus and overtakes tool_a
    const r2 = resolver.resolve('query', queryVec, { recentToolNames: ['tool_b'] }, { alpha: 1.0, inertiaBonus: 0.20 });
    expect(r2.tools[0].name).toBe('tool_b');
  });

  it('blends prior turn embedding for multi-turn trajectory context when topics correlate', () => {
    const catalog = new ToolCatalog();
    const resolver = new ToolResolver(catalog);

    // Follow-up query sharing vector subspace (dot = 0.8*0.6 + 0.6*0.8 = 0.96 > 0.35)
    const current = new Float32Array([0.8, 0.6]);
    const prior = new Float32Array([0.6, 0.8]);

    const blended = resolver.blendTrajectory(current, prior, 0.75);
    expect(blended[0]).toBeGreaterThan(blended[1]);
    expect(blended[1]).toBeGreaterThan(0.0);

    const norm = Math.sqrt(blended[0] * blended[0] + blended[1] * blended[1]);
    expect(norm).toBeCloseTo(1.0);
  });

  it('enforces topK and maxPerDomain limits on cold start fallback defaultTools', () => {
    const catalog = new ToolCatalog();
    catalog.registerTools([
      { name: 'stripe_charge', description: 'Process payment', domain: 'stripe' },
      { name: 'stripe_refund', description: 'Refund payment', domain: 'stripe' },
      { name: 'slack_msg', description: 'Send slack message', domain: 'slack' },
      { name: 'jira_ticket', description: 'Create ticket', domain: 'jira' },
    ]);

    const resolver = new ToolResolver(catalog);
    resolver.syncIndex();

    // Query matches nothing (score 0), defaultTools has 4 tools, but topK is 2
    const resTopK = resolver.resolve('unrelated gibberish query 12345', undefined, undefined, {
      topK: 2,
      minScoreThreshold: 0.1,
      defaultTools: ['stripe_charge', 'stripe_refund', 'slack_msg', 'jira_ticket'],
    });

    expect(resTopK.tools.length).toBe(2);
    expect(resTopK.selectedNames).toEqual(['stripe_charge', 'stripe_refund']);

    // Query matches nothing, defaultTools has multiple stripe tools, but maxPerDomain is 1
    const resDomain = resolver.resolve('unrelated gibberish query 12345', undefined, undefined, {
      topK: 3,
      maxPerDomain: 1,
      minScoreThreshold: 0.1,
      defaultTools: ['stripe_charge', 'stripe_refund', 'slack_msg'],
    });

    expect(resDomain.tools.length).toBe(2);
    expect(resDomain.selectedNames).toEqual(['stripe_charge', 'slack_msg']);
  });
});

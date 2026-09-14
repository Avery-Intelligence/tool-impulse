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

  it('blends prior turn embedding for multi-turn trajectory context', () => {
    const catalog = new ToolCatalog();
    const resolver = new ToolResolver(catalog);

    const current = new Float32Array([1.0, 0.0]);
    const prior = new Float32Array([0.0, 1.0]);

    const blended = resolver.blendTrajectory(current, prior, 0.75);
    expect(blended[0]).toBeGreaterThan(blended[1]);
    expect(blended[1]).toBeGreaterThan(0.0);

    const norm = Math.sqrt(blended[0] * blended[0] + blended[1] * blended[1]);
    expect(norm).toBeCloseTo(1.0);
  });
});

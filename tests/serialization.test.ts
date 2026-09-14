import { describe, it, expect } from 'vitest';
import { ToolImpulse } from '../src/core/engine.js';
import { ToolDefinition, ToolTransitionEdge } from '../src/core/types.js';

describe('State Serialization & Instant Hydration', () => {
  const tools: ToolDefinition[] = [
    { name: 'stripe_charge', description: 'Charge customer credit card', domain: 'stripe' },
    { name: 'stripe_refund', description: 'Refund customer charge', domain: 'stripe' },
    { name: 'slack_send', description: 'Send message to Slack', domain: 'slack' },
  ];

  const edges: ToolTransitionEdge[] = [
    { fromTool: 'stripe_charge', toTool: 'stripe_refund', weight: 0.8 },
  ];

  it('exports catalog state including embeddings, tools, and companion edges', () => {
    const engine = new ToolImpulse();
    engine.registerToolsSync(tools);
    engine.addWorkflowEdges(edges);
    engine.setEmbeddings({
      stripe_charge: [1.0, 0.0],
      stripe_refund: [0.9, 0.1],
      slack_send: [0.0, 1.0],
    });

    const state = engine.exportState();
    expect(state.version).toBe(1);
    expect(state.tools.length).toBe(3);
    expect(state.embeddings).toBeDefined();
    expect(state.embeddings!['stripe_charge']).toBeDefined();
    expect(state.edges?.length).toBe(1);
    expect(state.edges![0].fromTool).toBe('stripe_charge');
  });

  it('hydrates from exported state with zero embedding overhead', () => {
    const original = new ToolImpulse();
    original.registerToolsSync(tools);
    original.addWorkflowEdges(edges);
    original.setEmbeddings({
      stripe_charge: [1.0, 0.0],
      stripe_refund: [0.9, 0.1],
      slack_send: [0.0, 1.0],
    });

    const serializedState = original.exportState();

    // Re-instantiate a fresh instance with initialState (simulating serverless cold start)
    const hydrated = new ToolImpulse({ initialState: serializedState });

    expect(hydrated.getCatalog().getToolNames()).toEqual(['stripe_charge', 'stripe_refund', 'slack_send']);
    expect(hydrated.getCatalog().getNeighbors('stripe_charge').get('stripe_refund')).toBe(0.8);

    // Synchronous resolution works immediately without needing re-embedding
    const queryVec = new Float32Array([1.0, 0.0]);
    const rOriginal = original.resolveSync('charge customer', queryVec);
    const rHydrated = hydrated.resolveSync('charge customer', queryVec);

    expect(rHydrated.selectedNames).toEqual(rOriginal.selectedNames);
    expect(rHydrated.scores).toEqual(rOriginal.scores);
    expect(rHydrated.primaryTool?.name).toBe(rOriginal.primaryTool?.name);
  });
});

import { describe, it, expect } from 'vitest';
import { RegexCodeTopologyProvider } from '../src/core/topology.js';
import { ToolImpulseEngine } from '../src/core/engine.js';

describe('RegexCodeTopologyProvider', () => {
  it('extracts code symbols and boosts relevant code tools', async () => {
    const topology = new RegexCodeTopologyProvider();

    const symbols = topology.extractSymbols('Can you check the callers of processPayment in src/auth/service.ts?');
    expect(symbols).toContain('processPayment');
    expect(symbols).toContain('src/auth/service.ts');
    expect(symbols).toContain('callers');

    const relatedTools = topology.getRelatedTools(symbols);
    expect(relatedTools).toContain('code_inspect');
    expect(relatedTools).toContain('code_search');

    // Wire into engine and verify boosting
    const engine = new ToolImpulseEngine();
    engine.registerToolsSync([
      { name: 'code_inspect', description: 'Inspect AST callers and blast radius', family: 'code' },
      { name: 'slack_send', description: 'Send slack message', family: 'slack' },
    ]);

    const result = await engine.resolve('Check callers of handleCheckout in src/api.ts', undefined, {
      codeTopology: topology,
    });

    expect(result.tools[0].name).toBe('code_inspect');
  });
});

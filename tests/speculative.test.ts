import { describe, it, expect } from 'vitest';
import { SpeculativeRunner } from '../src/core/speculative.js';
import { ToolImpulseEngine } from '../src/core/engine.js';

describe('Speculative Pre-Execution', () => {
  it('dispatches parallel speculative execution for high-confidence idempotent reads', async () => {
    const engine = new ToolImpulseEngine();
    engine.registerToolsSync([
      { name: 'stripe_get_invoice', description: 'Fetch Stripe invoice by id', readOnly: true },
      { name: 'stripe_delete_invoice', description: 'Delete Stripe invoice by id', readOnly: false },
    ]);

    let executedTool = '';
    const executor = async (tool: any) => {
      executedTool = tool.name;
      return { invoiceId: 'inv_123', total: 5000 };
    };

    // Query matches stripe_get_invoice strongly
    const result = await engine.resolve('stripe get invoice details');
    const speculative = SpeculativeRunner.maybeDispatch(result, 'stripe get invoice details', executor, {
      confidenceThreshold: 0.50,
    });

    expect(speculative.status).toBe('dispatched');
    expect(speculative.promise).toBeDefined();

    const output = await speculative.promise;
    expect(output).toEqual({ invoiceId: 'inv_123', total: 5000 });
    expect(executedTool).toBe('stripe_get_invoice');
  });

  it('skips speculative execution for destructive write operations', async () => {
    const engine = new ToolImpulseEngine();
    engine.registerToolsSync([
      { name: 'delete_customer_db', description: 'Permanently drop customer record', readOnly: false },
    ]);

    const result = await engine.resolve('delete customer db');
    const speculative = SpeculativeRunner.maybeDispatch(result, 'delete customer db', async () => {});

    expect(speculative.status).toBe('skipped');
    expect(speculative.reason).toContain('not classified as an idempotent read');
  });
});

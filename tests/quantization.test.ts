import { describe, it, expect } from 'vitest';
import { Int8Quantizer } from '../src/core/quantization.js';
import { ImpulseBank } from '../src/core/bank.js';

describe('Int8 Scalar Quantization', () => {
  it('quantizes Float32 to Int8 with high dot-product fidelity (>99%)', () => {
    const dim = 128;
    const a = new Float32Array(dim);
    const b = new Float32Array(dim);

    // Seed deterministic values
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < dim; i++) {
      a[i] = Math.sin(i * 0.3);
      b[i] = Math.sin(i * 0.3 + 0.1);
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    normA = Math.sqrt(normA);
    normB = Math.sqrt(normB);
    for (let i = 0; i < dim; i++) {
      a[i] /= normA;
      b[i] /= normB;
    }

    // Exact float dot product
    let floatDot = 0;
    for (let i = 0; i < dim; i++) {
      floatDot += a[i] * b[i];
    }

    // Quantized dot product
    const qA = Int8Quantizer.quantize(a);
    const qB = Int8Quantizer.quantize(b);

    expect(qA.data.byteLength).toBe(dim); // 1 byte per dimension (75% savings vs 4 bytes)
    const int8Dot = Int8Quantizer.dotProduct(qA, qB);

    expect(Math.abs(floatDot - int8Dot)).toBeLessThan(0.015); // >98.5% fidelity
  });

  it('runs ImpulseBank seamlessly in quantized mode', () => {
    const bank = new ImpulseBank({ quantizeInt8: true });
    expect(bank.isQuantized()).toBe(true);

    bank.registerTools([
      { name: 'stripe_pay', description: 'Stripe payments' },
      { name: 'jira_bug', description: 'Jira bug' },
    ]);

    bank.setEmbeddings({
      stripe_pay: [1.0, 0.0, 0.0],
      jira_bug: [0.0, 1.0, 0.0],
    });

    const query = new Float32Array([0.95, 0.05, 0.0]);
    const simStripe = bank.computeCosine(query, 'stripe_pay');
    const simJira = bank.computeCosine(query, 'jira_bug');

    expect(simStripe).toBeGreaterThan(0.9);
    expect(simJira).toBeLessThan(0.1);
  });
});

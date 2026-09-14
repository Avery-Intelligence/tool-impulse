import { describe, it, expect } from 'vitest';
import { OkapiBM25 } from '../src/core/bm25.js';

describe('OkapiBM25', () => {
  it('indexes documents and computes Robertson-Spärck Jones scores with length normalization', () => {
    const bm25 = new OkapiBM25();

    bm25.indexDocuments([
      { id: 'doc1', text: 'stripe customer billing invoice payment' },
      { id: 'doc2', text: 'jira issue ticket bug tracker task' },
      { id: 'doc3', text: 'slack chat channel team notification message' },
    ]);

    const scoresStripe = bm25.score('unpaid stripe customer invoice');
    expect(scoresStripe.get('doc1')).toBeGreaterThan(0.8);
    expect(scoresStripe.get('doc2')).toBeLessThan(0.1);

    const scoresJira = bm25.score('file a high priority jira bug');
    expect(scoresJira.get('doc2')).toBeGreaterThan(0.8);
    expect(scoresJira.get('doc1')).toBeLessThan(0.1);
  });
});

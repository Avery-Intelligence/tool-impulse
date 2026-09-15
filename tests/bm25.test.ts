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
    expect(scoresStripe.get('doc1')).toBeGreaterThan(0.5);
    expect(scoresStripe.get('doc2')).toBeLessThan(0.1);

    const scoresJira = bm25.score('file a high priority jira bug');
    expect(scoresJira.get('doc2')).toBeGreaterThan(0.2);
    expect(scoresJira.get('doc1')).toBeLessThan(0.1);
  });

  it('matches plurals and suffixes via morphological stemming', () => {
    const bm25 = new OkapiBM25();

    bm25.indexDocuments([
      { id: 'tool_invoice', text: 'create customer invoice and process charge' },
      { id: 'tool_meeting', text: 'schedule calendar meeting and event' },
    ]);

    // Query with plurals "invoices" and "charges"
    const scoreInvoices = bm25.score('fetch unpaid customer invoices and charges');
    expect(scoreInvoices.get('tool_invoice')).toBeGreaterThan(0.3);

    // Query with plurals "meetings" and "events"
    const scoreMeetings = bm25.score('list upcoming team meetings and events');
    expect(scoreMeetings.get('tool_meeting')).toBeGreaterThan(0.2);
  });

  it('correctly splits snake_case and camelCase identifiers', () => {
    const bm25 = new OkapiBM25();
    bm25.indexDocuments([
      { id: 'stripe_refund_charge', text: 'stripe_refund_charge: settle balance' },
      { id: 'githubMergePullRequest', text: 'githubMergePullRequest: ship code' },
    ]);

    // Matches "refund charge" against snake_case tool name
    const scoreRefund = bm25.score('process refund charge');
    expect(scoreRefund.get('stripe_refund_charge')).toBeGreaterThan(0.4);

    // Matches "merge pull request" against camelCase tool name
    const scoreMerge = bm25.score('merge pull request');
    expect(scoreMerge.get('githubMergePullRequest')).toBeGreaterThan(0.4);
  });

  it('does not falsely score 1.0 on queries with accidental single-word matches or nonsense', () => {
    const bm25 = new OkapiBM25();
    bm25.indexDocuments([
      { id: 'check_credit', text: 'check credit score balance' },
      { id: 'send_slack', text: 'send slack notification message' },
    ]);

    // Nonsense query containing only 1 accidental word match out of 4 terms
    const scoresAccidental = bm25.score('bananas astronaut galaxy credit');
    const creditScore = scoresAccidental.get('check_credit') || 0;
    // Bounded normalization: 1 match out of 4 terms cannot score 1.0; properly calibrated around 0.15-0.25
    expect(creditScore).toBeLessThan(0.3);
    expect(creditScore).toBeGreaterThan(0.1);

    // Complete nonsense: 0 matches
    const scoresZero = bm25.score('bananas astronaut spaceship quantum');
    expect(scoresZero.get('check_credit')).toBe(0);
    expect(scoresZero.get('send_slack')).toBe(0);
  });

  it('preserves words that naive stemmers butcher (e.g. string, speed)', () => {
    expect(OkapiBM25.stem('string')).toBe('string');
    expect(OkapiBM25.stem('speed')).toBe('speed');
    expect(OkapiBM25.stem('agreed')).toBe('agre');
    expect(OkapiBM25.stem('invoices')).toBe('invoic');
    expect(OkapiBM25.stem('processing')).toBe('process');
  });
});


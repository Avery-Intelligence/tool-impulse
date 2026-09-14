/**
 * Real Okapi BM25 In-Memory Index
 *
 * Mathematically grounded lexical search using Robertson-Spärck Jones IDF
 * and term frequency saturation with document length normalization.
 *
 * Parameters:
 * - k1 = 1.2 (controls term frequency saturation)
 * - b = 0.75 (controls document length normalization)
 */

export interface BM25Document {
  id: string;
  tokens: string[];
}

export class OkapiBM25 {
  private k1: number;
  private b: number;
  private docs: Map<string, string[]> = new Map();
  private docLengths: Map<string, number> = new Map();
  private termDocFreq: Map<string, number> = new Map();
  private avgDocLength: number = 0;
  private totalDocs: number = 0;

  constructor(k1: number = 1.2, b: number = 0.75) {
    this.k1 = k1;
    this.b = b;
  }

  /**
   * Fast lexical tokenizer: handles camelCase, snake_case, and non-alphanumerics.
   */
  public static tokenize(text: string): string[] {
    return text
      .replace(/([a-z])([A-Z])/g, '$1 $2')
      .toLowerCase()
      .split(/[^a-z0-9_]+/)
      .filter((t) => t.length > 1);
  }

  /**
   * Index or update corpus documents.
   */
  public indexDocuments(documents: Array<{ id: string; text: string }>): void {
    this.docs.clear();
    this.docLengths.clear();
    this.termDocFreq.clear();

    let totalLength = 0;

    for (const doc of documents) {
      const tokens = OkapiBM25.tokenize(doc.text);
      this.docs.set(doc.id, tokens);
      this.docLengths.set(doc.id, tokens.length);
      totalLength += tokens.length;

      const uniqueTokens = new Set(tokens);
      for (const token of uniqueTokens) {
        this.termDocFreq.set(token, (this.termDocFreq.get(token) || 0) + 1);
      }
    }

    this.totalDocs = documents.length;
    this.avgDocLength = this.totalDocs > 0 ? totalLength / this.totalDocs : 1.0;
  }

  /**
   * Compute inverse document frequency (Robertson-Spärck Jones formula).
   */
  private computeIdf(term: string): number {
    const docFreq = this.termDocFreq.get(term) || 0;
    if (docFreq === 0) return 0;
    // Standard Okapi BM25 IDF with +1 smoothing to ensure non-negative weights
    return Math.log(((this.totalDocs - docFreq + 0.5) / (docFreq + 0.5)) + 1);
  }

  /**
   * Score all indexed documents against a query string.
   * Returns a map of docId -> score normalized in [0, 1].
   */
  public score(query: string): Map<string, number> {
    const queryTokens = OkapiBM25.tokenize(query);
    const scores = new Map<string, number>();
    if (queryTokens.length === 0 || this.totalDocs === 0) return scores;

    let maxScore = 0;

    for (const [id, docTokens] of this.docs.entries()) {
      const docLen = this.docLengths.get(id) || 1;
      let score = 0;

      // Count term frequencies in document
      const tfMap = new Map<string, number>();
      for (const t of docTokens) {
        tfMap.set(t, (tfMap.get(t) || 0) + 1);
      }

      for (const q of queryTokens) {
        const tf = tfMap.get(q) || 0;
        if (tf === 0) continue;

        const idf = this.computeIdf(q);
        const numerator = tf * (this.k1 + 1);
        const denominator = tf + this.k1 * (1 - this.b + this.b * (docLen / this.avgDocLength));

        score += idf * (numerator / denominator);
      }

      scores.set(id, score);
      if (score > maxScore) maxScore = score;
    }

    // Normalize scores to [0, 1]
    if (maxScore > 0) {
      for (const [id, val] of scores.entries()) {
        scores.set(id, val / maxScore);
      }
    }

    return scores;
  }
}

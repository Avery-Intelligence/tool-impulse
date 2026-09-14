/**
 * Okapi BM25 Lexical Index with Martin Porter (1980) Stemming
 *
 * Implements Robertson-Spärck Jones IDF and term frequency saturation
 * with document length normalization.
 *
 * Parameters:
 * - k1 = 1.2 (controls term frequency saturation)
 * - b = 0.75 (controls document length normalization)
 */

export interface BM25Document {
  id: string;
  tokens: string[];
}

/**
 * Standard Martin Porter Stemming Algorithm (1980).
 * Replaces ad-hoc string slicing with canonical morphological reduction.
 */
export class PorterStemmer {
  private static isConsonant(word: string, i: number): boolean {
    const ch = word[i];
    if (ch === 'a' || ch === 'e' || ch === 'i' || ch === 'o' || ch === 'u') return false;
    if (ch === 'y') {
      if (i === 0) return true;
      return !PorterStemmer.isConsonant(word, i - 1);
    }
    return true;
  }

  /** Measure m: number of VC sequences between 0 and k */
  private static measure(word: string, k: number): number {
    let n = 0;
    let i = 0;
    while (i <= k && PorterStemmer.isConsonant(word, i)) i++;
    if (i > k) return 0;
    while (i <= k) {
      while (i <= k && !PorterStemmer.isConsonant(word, i)) i++;
      if (i > k) break;
      while (i <= k && PorterStemmer.isConsonant(word, i)) i++;
      n++;
    }
    return n;
  }

  private static vowelInStem(word: string, k: number): boolean {
    for (let i = 0; i <= k; i++) {
      if (!PorterStemmer.isConsonant(word, i)) return true;
    }
    return false;
  }

  private static doubleConsonant(word: string, i: number): boolean {
    if (i < 1) return false;
    if (word[i] !== word[i - 1]) return false;
    return PorterStemmer.isConsonant(word, i);
  }

  private static cvc(word: string, i: number): boolean {
    if (i < 2 || !PorterStemmer.isConsonant(word, i) || PorterStemmer.isConsonant(word, i - 1) || !PorterStemmer.isConsonant(word, i - 2)) {
      return false;
    }
    const ch = word[i];
    return ch !== 'w' && ch !== 'x' && ch !== 'y';
  }

  public static stem(w: string): string {
    let word = w.toLowerCase();
    if (word.length <= 2) return word;

    let k = word.length - 1;

    // Step 1a
    if (word.endsWith('sses')) {
      word = word.slice(0, -2);
      k -= 2;
    } else if (word.endsWith('ies')) {
      word = word.slice(0, -2);
      k -= 2;
    } else if (!word.endsWith('ss') && word.endsWith('s')) {
      word = word.slice(0, -1);
      k -= 1;
    }

    // Step 1b
    let extraStep1b = false;
    if (word.endsWith('eed')) {
      const stemLen = k - 2;
      if (PorterStemmer.measure(word, stemLen - 1) > 0) {
        word = word.slice(0, -1);
        k -= 1;
      }
    } else if (word.endsWith('ed')) {
      const stemLen = k - 1;
      if (PorterStemmer.vowelInStem(word, stemLen - 1)) {
        word = word.slice(0, -2);
        k -= 2;
        extraStep1b = true;
      }
    } else if (word.endsWith('ing')) {
      const stemLen = k - 2;
      if (PorterStemmer.vowelInStem(word, stemLen - 1)) {
        word = word.slice(0, -3);
        k -= 3;
        extraStep1b = true;
      }
    }

    if (extraStep1b) {
      if (word.endsWith('at') || word.endsWith('bl') || word.endsWith('iz')) {
        word += 'e';
        k += 1;
      } else if (PorterStemmer.doubleConsonant(word, k) && !word.endsWith('l') && !word.endsWith('s') && !word.endsWith('z')) {
        word = word.slice(0, -1);
        k -= 1;
      } else if (PorterStemmer.measure(word, k) === 1 && PorterStemmer.cvc(word, k)) {
        word += 'e';
        k += 1;
      }
    }

    // Step 1c: y -> i
    if (word.endsWith('y') && PorterStemmer.vowelInStem(word, k - 1)) {
      word = word.slice(0, -1) + 'i';
    }

    // Step 2
    const step2Pairs: [string, string][] = [
      ['ational', 'ate'], ['tional', 'tion'], ['enci', 'ence'], ['anci', 'ance'],
      ['izer', 'ize'], ['bli', 'ble'], ['alli', 'al'], ['entli', 'ent'],
      ['eli', 'e'], ['ousli', 'ous'], ['ization', 'ize'], ['ation', 'ate'],
      ['ator', 'ate'], ['alism', 'al'], ['iveness', 'ive'], ['fulness', 'ful'],
      ['ousness', 'ous'], ['aliti', 'al'], ['iviti', 'ive'], ['biliti', 'ble'],
    ];

    for (const [suffix, replacement] of step2Pairs) {
      if (word.endsWith(suffix)) {
        const stemLen = word.length - suffix.length;
        if (PorterStemmer.measure(word, stemLen - 1) > 0) {
          word = word.slice(0, stemLen) + replacement;
          k = word.length - 1;
        }
        break;
      }
    }

    // Step 3
    const step3Pairs: [string, string][] = [
      ['icate', 'ic'], ['ative', ''], ['alize', 'al'], ['iciti', 'ic'],
      ['ical', 'ic'], ['ful', ''], ['ness', ''],
    ];

    for (const [suffix, replacement] of step3Pairs) {
      if (word.endsWith(suffix)) {
        const stemLen = word.length - suffix.length;
        if (PorterStemmer.measure(word, stemLen - 1) > 0) {
          word = word.slice(0, stemLen) + replacement;
          k = word.length - 1;
        }
        break;
      }
    }

    // Step 4
    const step4Suffixes = [
      'al', 'ance', 'ence', 'er', 'ic', 'able', 'ible', 'ant',
      'ement', 'ment', 'ent', 'ou', 'ism', 'ate', 'iti', 'ous', 'ive', 'ize',
    ];

    for (const suffix of step4Suffixes) {
      if (word.endsWith(suffix)) {
        const stemLen = word.length - suffix.length;
        if (PorterStemmer.measure(word, stemLen - 1) > 1) {
          word = word.slice(0, stemLen);
          k = word.length - 1;
        }
        break;
      }
    }

    if (word.endsWith('ion')) {
      const stemLen = word.length - 3;
      if (stemLen > 0 && (word[stemLen - 1] === 's' || word[stemLen - 1] === 't')) {
        if (PorterStemmer.measure(word, stemLen - 1) > 1) {
          word = word.slice(0, stemLen);
          k = word.length - 1;
        }
      }
    }

    // Step 5a
    if (word.endsWith('e')) {
      const stemLen = word.length - 1;
      const m = PorterStemmer.measure(word, stemLen - 1);
      if (m > 1 || (m === 1 && !PorterStemmer.cvc(word, stemLen - 1))) {
        word = word.slice(0, -1);
        k -= 1;
      }
    }

    // Step 5b
    if (word.endsWith('ll') && PorterStemmer.measure(word, k) > 1) {
      word = word.slice(0, -1);
    }

    return word;
  }
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
   * Delegates to PorterStemmer.
   */
  public static stem(word: string): string {
    return PorterStemmer.stem(word);
  }

  /**
   * Lexical tokenizer:
   * 1. Splits snake_case (e.g. "stripe_refund_charge" -> "stripe refund charge").
   * 2. Splits kebab-case (e.g. "github-merge-pr" -> "github merge pr").
   * 3. Splits camelCase (e.g. "listInvoices" -> "list Invoices").
   * 4. Tokenizes alphanumeric words of length >= 2.
   * 5. Emits both original token and its Porter-stemmed form.
   */
  public static tokenize(text: string): string[] {
    const normalized = text
      .replace(/[_-]+/g, ' ')
      .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
      .toLowerCase();

    const rawWords = normalized.split(/[^a-z0-9]+/).filter((w) => w.length > 1);
    const tokens: string[] = [];

    for (const w of rawWords) {
      tokens.push(w);
      const stemmed = PorterStemmer.stem(w);
      if (stemmed !== w && stemmed.length > 1) {
        tokens.push(stemmed);
      }
    }

    return tokens;
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
   * Compute inverse document frequency (Robertson-Spärck Jones formula with smoothing).
   */
  private computeIdf(term: string): number {
    const docFreq = this.termDocFreq.get(term) || 0;
    if (docFreq === 0) return 0;
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

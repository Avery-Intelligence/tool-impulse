import { CodeTopologyProvider } from './types.js';

export interface RegexCodeTopologyConfig {
  /** Custom mapping from detected keyword patterns to tool names */
  customMappings?: Record<string, string[]>;
}

export class RegexCodeTopologyProvider implements CodeTopologyProvider {
  private fileRegex = /(?:[\w.-]+\/)+[\w.-]+\.[a-zA-Z0-9]+/g;
  private symbolRegex = /\b[a-zA-Z_][a-zA-Z0-9_]*(?:[A-Z][a-zA-Z0-9_]+)+\b/g;
  private actionVerbs = new Set(['blast', 'caller', 'callers', 'callee', 'callees', 'deps', 'dependencies', 'ast', 'diff', 'refactor', 'symbol', 'tests', 'traces']);
  private customMappings: Record<string, string[]>;

  constructor(config: RegexCodeTopologyConfig = {}) {
    this.customMappings = config.customMappings || {
      blast: ['code_inspect', 'git_diff_impact'],
      callers: ['code_inspect', 'code_symbol'],
      diff: ['git_diff', 'code_inspect'],
      tests: ['code_inspect', 'run_tests'],
      refactor: ['code_inspect', 'code_search'],
      symbol: ['code_search', 'code_symbol'],
      map: ['code_map', 'code_tour'],
    };
  }

  /**
   * Extract source code identifiers, file paths, and AST action verbs from query string.
   */
  public extractSymbols(query: string): string[] {
    const symbols = new Set<string>();

    // 1. File paths
    const files = query.match(this.fileRegex) || [];
    for (const f of files) {
      symbols.add(f);
    }

    // 2. CamelCase / PascalCase symbols (e.g. processPayment, OrderService)
    const codeIdentifiers = query.match(this.symbolRegex) || [];
    for (const id of codeIdentifiers) {
      symbols.add(id);
    }

    // 3. Action verbs
    const words = query.toLowerCase().split(/[^a-z0-9_]+/);
    for (const w of words) {
      if (this.actionVerbs.has(w)) {
        symbols.add(w);
      }
    }

    return Array.from(symbols);
  }

  /**
   * Return tool names that should be boosted based on detected symbols.
   */
  public getRelatedTools(symbols: string[]): string[] {
    const related = new Set<string>();

    for (const s of symbols) {
      const lower = s.toLowerCase();
      // Check custom mappings
      if (this.customMappings[lower]) {
        for (const t of this.customMappings[lower]) {
          related.add(t);
        }
      }

      // If a file path was detected, boost file inspection tools
      if (s.includes('/') || s.includes('.')) {
        related.add('code_search');
        related.add('code_inspect');
      }
    }

    return Array.from(related);
  }
}

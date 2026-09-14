import { spawn } from 'node:child_process';
import * as readline from 'node:readline';
import { ToolImpulseEngine } from '../core/engine.js';
import { McpToolDefinition, mcpToImpulseTool } from '../adapters/mcp.js';
import { ImpulseSessionState } from '../core/types.js';

export class StdioMcpProxy {
  private engine: ToolImpulseEngine;
  private session: ImpulseSessionState = {
    recentToolNames: [],
    turnCount: 0,
  };
  private fullToolCatalog: McpToolDefinition[] = [];

  constructor() {
    this.engine = new ToolImpulseEngine();
  }

  public async start(command: string, args: string[]): Promise<void> {
    const child = spawn(command, args, {
      stdio: ['pipe', 'pipe', 'inherit'],
    });

    const clientReader = readline.createInterface({
      input: process.stdin,
      terminal: false,
    });

    const downstreamReader = readline.createInterface({
      input: child.stdout,
      terminal: false,
    });

    // Handle messages from MCP client (e.g. Claude Desktop)
    clientReader.on('line', async (line) => {
      if (!line.trim()) return;
      try {
        const msg = JSON.parse(line);

        // Intercept tool execution to update hysteresis and Bayesian transition graph
        if (msg.method === 'tools/call' && msg.params?.name) {
          const toolName = msg.params.name;
          this.session.recentToolNames = [toolName, ...(this.session.recentToolNames || []).slice(0, 4)];
          this.session.turnCount = (this.session.turnCount || 0) + 1;
          this.engine.recordExecution([toolName]);
        }

        // Pass message downstream to wrapped server
        child.stdin.write(line + '\n');
      } catch (err) {
        child.stdin.write(line + '\n');
      }
    });

    // Handle messages from downstream MCP server
    downstreamReader.on('line', async (line) => {
      if (!line.trim()) return;
      try {
        const msg = JSON.parse(line);

        // Intercept tools/list response to catalog tools
        if (msg.result?.tools && Array.isArray(msg.result.tools)) {
          this.fullToolCatalog = msg.result.tools;
          const impulseTools = this.fullToolCatalog.map(mcpToImpulseTool);
          this.engine.registerToolsSync(impulseTools);
        }

        // Forward to client
        process.stdout.write(line + '\n');
      } catch (err) {
        process.stdout.write(line + '\n');
      }
    });

    child.on('exit', (code) => {
      process.exit(code || 0);
    });
  }
}

// CLI entry point
export async function runCli(): Promise<void> {
  const args = process.argv.slice(2);

  if (args.length === 0 || args.includes('--help') || args.includes('-h')) {
    console.error(`
Tool Impulse Engine (TIE) - Local MCP Proxy
Unconscious Perceptual Reflex for Autonomous Agent Tool Retrieval

Usage:
  npx tool-impulse-proxy <command> [args...]

Examples:
  npx tool-impulse-proxy npx -y @modelcontextprotocol/server-everything
  npx tool-impulse-proxy node ./my-custom-mcp-server.js

Options:
  -h, --help       Show this help message
  -v, --version    Show version
    `);
    process.exit(0);
  }

  if (args.includes('--version') || args.includes('-v')) {
    console.log('tool-impulse 1.0.0');
    process.exit(0);
  }

  const [cmd, ...cmdArgs] = args;
  const proxy = new StdioMcpProxy();
  await proxy.start(cmd, cmdArgs);
}

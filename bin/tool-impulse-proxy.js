#!/usr/bin/env node
import { runCli } from '../dist/cli/proxy.js';

runCli().catch((err) => {
  console.error('[tool-impulse-proxy] Fatal error:', err);
  process.exit(1);
});

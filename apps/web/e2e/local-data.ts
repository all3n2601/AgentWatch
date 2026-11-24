import os from "node:os";
import path from "node:path";

/**
 * The web server's AGENTWATCH_DATA_DIR in browser tests. It holds no local artifacts, so
 * pages show only fixture API data; a test that writes a local artifact here removes it.
 */
export const E2E_DATA_DIR = path.join(
  os.tmpdir(),
  "agentwatch-e2e-no-local-data",
);

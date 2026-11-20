/** Child entry for `port.test.ts`, run with a sandbox env: start OpenCode in process with
 * `argv[2]` as its config directory and print, as JSON, the commands it lists for the directory
 * `argv[3]`. Commands load after the host starts (measured 2026-11-11: the first list for a location
 * is empty, and the built-ins can arrive before the configured ones), so it polls until every name
 * in the comma list `argv[5]` is listed, or 20 s have passed. */

import { OpenCode } from "@opencode/sdk";

const [config, directory, database, wanted = ""] = process.argv.slice(2) as [string, string, string, string?];
const names = wanted.split(",").filter(Boolean);
const oc = await OpenCode.create({
  database: { path: database },
  config: { content: "{}", project: false, claude: false, directory: config },
  models: { fetch: false },
});
let data: { name: string; description?: string }[] = [];
for (let i = 0; i < 40; i++) {
  data = (await oc.command.list({ location: { directory } })).data as typeof data;
  if (data.length > 0 && names.every((n) => data.some((c) => c.name === n))) break;
  await Bun.sleep(500);
}
console.log(JSON.stringify(data));
await oc.close();
process.exit(0);

import assert from "node:assert/strict";
import test from "node:test";
import nextConfig from "../next.config";
import { routes, SECTION_PATHS } from "../lib/routes";

const RUN = "127d9be5-079e-4122-96fb-d5995bf71752";

test("route builders omit empty query values and encode path segments", () => {
  assert.equal(routes.overview(), "/");
  assert.equal(routes.overview("qwen3:1.7b"), "/?model=qwen3%3A1.7b");
  assert.equal(routes.runs(), "/runs");
  assert.equal(routes.runs({ q: "fix window" }), "/runs?q=fix+window");
  assert.equal(routes.run(RUN), `/runs/${RUN}`);
  assert.equal(routes.run(RUN, "timeline"), `/runs/${RUN}?tab=timeline`);
  assert.equal(routes.run("a/b"), "/runs/a%2Fb");
  assert.equal(routes.compare(RUN), `/compare?run=${RUN}`);
  assert.equal(routes.compare(RUN, RUN), `/compare?run=${RUN}&baseline=${RUN}`);
  assert.equal(routes.live(), "/live");
  assert.equal(routes.liveExport(RUN), `/api/live/runs/${RUN}/export`);
});

test("every section path is a distinct top-level route", () => {
  const paths = Object.values(SECTION_PATHS);
  assert.equal(new Set(paths).size, paths.length);
  for (const path of paths) assert.match(path, /^\/[a-z]*$/);
});

test("legacy pages redirect, with a run id to that run's tab and without one to the replacement", async () => {
  const redirects = await nextConfig.redirects!();
  const find = (source: string, withRun: boolean) =>
    redirects.find((r) => r.source === source && Boolean(r.has) === withRun);

  assert.equal(find("/overview", false)?.destination, "/");
  assert.equal(find("/dataset", false)?.destination, "/quality");
  for (const [source, tab] of [
    ["/traces", "timeline"],
    ["/agents", "agent"],
    ["/resources", "resources"],
  ]) {
    assert.equal(find(source, true)?.destination, `/runs/:run?tab=${tab}`);
    assert.equal(find(source, false)?.destination, "/runs");
  }
  // Temporary, so the targets can still change without browsers caching them.
  assert.ok(redirects.every((r) => r.permanent === false));
  // The run id must be a UUID; anything else falls through to the plain redirect.
  const pattern = new RegExp(
    `^${(find("/traces", true)!.has![0] as { value: string }).value}$`,
  );
  assert.ok(pattern.test(RUN));
  assert.ok(!pattern.test("../../etc"));
});

import assert from "node:assert/strict";
import test from "node:test";
import { SETTINGS_TABS } from "../components/pages/settings/sections";
import {
  checkApiReadiness,
  DEFAULT_PREFERENCES,
  displayServiceUrl,
  insightsStatus,
  parsePreferences,
  parseRefreshOption,
  PREFERENCES_CHANGE_EVENT,
  PREFERENCES_STORAGE_KEY,
  serializePreferences,
} from "../lib/settings-page";

test("the preferences storage contract stays fixed for the shell", () => {
  // The workspace shell and the legacy dashboard read these exact names.
  assert.equal(PREFERENCES_STORAGE_KEY, "agentwatch.preferences");
  assert.equal(PREFERENCES_CHANGE_EVENT, "agentwatch:preferences");
});

test("settings tabs have distinct URL ids and heading ids in page order", () => {
  assert.deepEqual(
    SETTINGS_TABS.map((tab) => tab.label),
    ["Workspace", "Display", "Data connection", "Insights"],
  );
  const ids = SETTINGS_TABS.map((tab) => tab.id);
  const headingIds = SETTINGS_TABS.map((tab) => tab.headingId);
  assert.equal(new Set(ids).size, ids.length);
  assert.equal(new Set(headingIds).size, headingIds.length);
  for (const id of ids) assert.match(id, /^[a-z-]+$/);
});

test("parsePreferences reads a valid stored value", () => {
  assert.deepEqual(
    parsePreferences(
      '{"density":"compact","refresh":30,"sidebar":"collapsed"}',
    ),
    { density: "compact", refresh: 30, sidebar: "collapsed" },
  );
});

test("parsePreferences fills fields saved before they existed", () => {
  assert.deepEqual(parsePreferences('{"density":"compact"}'), {
    ...DEFAULT_PREFERENCES,
    density: "compact",
  });
});

test("parsePreferences resets malformed or invalid storage to defaults", () => {
  for (const raw of [
    null,
    "",
    "{",
    "null",
    "42",
    '"compact"',
    "[]",
    '{"density":"tiny"}',
    '{"refresh":"30"}',
    '{"refresh":45}',
    '{"sidebar":null}',
  ])
    assert.deepEqual(parsePreferences(raw), DEFAULT_PREFERENCES, String(raw));
});

test("parsePreferences drops unknown fields", () => {
  assert.deepEqual(parsePreferences('{"density":"compact","extra":true}'), {
    ...DEFAULT_PREFERENCES,
    density: "compact",
  });
});

test("serializePreferences writes the dashboard's JSON shape", () => {
  assert.equal(
    serializePreferences(DEFAULT_PREFERENCES),
    '{"density":"comfortable","refresh":0,"sidebar":"expanded"}',
  );
  const saved = {
    density: "compact",
    refresh: 60,
    sidebar: "collapsed",
  } as const;
  assert.deepEqual(parsePreferences(serializePreferences(saved)), saved);
});

test("parseRefreshOption accepts only the offered intervals", () => {
  assert.equal(parseRefreshOption("0"), 0);
  assert.equal(parseRefreshOption("15"), 15);
  assert.equal(parseRefreshOption("45"), null);
  assert.equal(parseRefreshOption("manual"), null);
});

test("displayServiceUrl strips credentials, query and fragment", () => {
  assert.equal(
    displayServiceUrl(
      "https://user:secret@api.example.test:8443/base/?token=x#f",
    ),
    "https://api.example.test:8443/base",
  );
  assert.equal(
    displayServiceUrl("http://127.0.0.1:8090"),
    "http://127.0.0.1:8090",
  );
  assert.equal(displayServiceUrl(undefined), null);
  assert.equal(displayServiceUrl("  "), null);
  assert.equal(displayServiceUrl("not a url"), null);
  assert.equal(displayServiceUrl("file:///etc/passwd"), null);
});

const respond =
  (body: unknown, status = 200): typeof fetch =>
  async () =>
    new Response(typeof body === "string" ? body : JSON.stringify(body), {
      status,
    });

test("checkApiReadiness reports connected only for a ready body", async () => {
  let requested = "";
  const ready: typeof fetch = async (input) => {
    requested = String(input);
    return new Response('{"status":"ready","storage":"postgresql"}');
  };
  assert.deepEqual(await checkApiReadiness("http://api.test/", ready), {
    state: "connected",
  });
  assert.equal(requested, "http://api.test/readyz");
});

test("checkApiReadiness separates storage, unexpected and unreachable failures", async () => {
  assert.deepEqual(
    await checkApiReadiness(
      "http://api.test",
      respond({ detail: "Database unavailable" }, 503),
    ),
    { state: "storage-unavailable" },
  );
  assert.deepEqual(
    await checkApiReadiness("http://api.test", respond("missing", 404)),
    { state: "unexpected", httpStatus: 404 },
  );
  assert.deepEqual(
    await checkApiReadiness("http://api.test", respond({ status: "ok" })),
    { state: "unexpected", httpStatus: 200 },
  );
  assert.deepEqual(
    await checkApiReadiness("http://api.test", respond("<html>")),
    { state: "unexpected", httpStatus: 200 },
  );
  const refused: typeof fetch = async () => {
    throw new TypeError("fetch failed");
  };
  assert.deepEqual(await checkApiReadiness("http://api.test", refused), {
    state: "unreachable",
  });
});

test("checkApiReadiness gives up after its timeout", async () => {
  const hangs: typeof fetch = (_input, init) =>
    new Promise((_resolve, reject) =>
      init?.signal?.addEventListener("abort", () =>
        reject(init.signal?.reason),
      ),
    );
  assert.deepEqual(await checkApiReadiness("http://api.test", hangs, 10), {
    state: "unreachable",
  });
});

test("insightsStatus exposes only the on/off state and model", () => {
  const config = {
    baseUrl: "https://user:pw@gateway.test/v1",
    apiKey: "sk-test-not-a-real-key",
    model: "gpt-test",
  };
  const status = insightsStatus(config);
  assert.deepEqual(status, { enabled: true, model: "gpt-test" });
  assert.doesNotMatch(JSON.stringify(status), /sk-|gateway|pw/);
  assert.deepEqual(insightsStatus(null), { enabled: false });
});

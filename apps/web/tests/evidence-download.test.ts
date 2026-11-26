import assert from "node:assert/strict";
import test from "node:test";
import {
  anySignal,
  downloadFailure,
  downloadFilename,
} from "../lib/evidence-download";

const RUN = "127d9be5-079e-4122-96fb-d5995bf71752";

test("the saved file keeps the name the route gives it", () => {
  assert.equal(
    downloadFilename(
      `attachment; filename="agentwatch-${RUN}-spans.jsonl"`,
      "spans.jsonl",
    ),
    `agentwatch-${RUN}-spans.jsonl`,
  );
  // No usable header: the file's own name, never an empty or path-like one.
  assert.equal(downloadFilename(null, "spans.jsonl"), "spans.jsonl");
  assert.equal(downloadFilename("attachment", "spans.jsonl"), "spans.jsonl");
  assert.equal(
    downloadFilename('attachment; filename="../../etc/passwd"', "x.json"),
    "x.json",
  );
});

test("a failed download gives the route's reason, with the status, and says the page is unchanged", () => {
  // History down: the saved artifact route forwards the 503 with its own message.
  const unavailable = downloadFailure(
    { status: 503, body: { error: "Saved artifact unavailable" } },
    "spans.jsonl",
  );
  assert.match(unavailable, /^spans\.jsonl could not be downloaded/);
  assert.match(unavailable, /Saved artifact unavailable \(HTTP 503\)/);
  assert.match(unavailable, /run shown here is unchanged/);

  // A rewritten local artifact (409): the route explains what to do.
  assert.match(
    downloadFailure(
      {
        status: 409,
        body: { error: "The local artifact files now hold a different run." },
      },
      "results.json",
    ),
    /now hold a different run \(HTTP 409\)/,
  );
});

test("a live export 503 names the live API, not run history", () => {
  // The live export route's body when the live API is down (failureStatus.unavailable).
  const message = downloadFailure(
    {
      status: 503,
      body: {
        error: "The live API did not respond within 10 s.",
        kind: "unavailable",
      },
    },
    "live records",
  );
  assert.match(
    message,
    /The live API did not respond within 10 s \(HTTP 503\)/,
  );
  assert.doesNotMatch(message, /history/i);
});

test("a 404 for a missing run says the run was not found, not the file", () => {
  const message = downloadFailure(
    {
      status: 404,
      body: { error: "The live API has no such run.", kind: "not_found" },
    },
    "live records",
  );
  assert.match(message, /The live API has no such run \(HTTP 404\)/);
  assert.doesNotMatch(message, /file was not found/);
});

test("without the route's message, the wording is generic by status", () => {
  assert.match(
    downloadFailure({ status: 404, body: null }, "results.json"),
    /results\.json could not be downloaded: it was not found \(HTTP 404\)/,
  );
  assert.match(
    downloadFailure({ status: 503, body: {} }, "results.json"),
    /the server is unavailable right now \(HTTP 503\)/,
  );
  // A body that is not the route's error shape is not shown to the viewer.
  const html = downloadFailure(
    { status: 500, body: "<html>stack</html>" },
    "results.json",
  );
  assert.doesNotMatch(html, /stack/);
  assert.match(html, /the server refused it \(HTTP 500\)/);
  // Nor is an over-long message.
  assert.doesNotMatch(
    downloadFailure(
      { status: 500, body: { error: "x".repeat(301) } },
      "results.json",
    ),
    /xxx/,
  );
  assert.match(
    downloadFailure({ status: null, body: null }, "results.json"),
    /could not be reached/,
  );
});

/** Runs `check` as in a browser without AbortSignal.any, then restores it. */
function withoutAbortSignalAny(check: () => void) {
  const native = Object.getOwnPropertyDescriptor(AbortSignal, "any");
  assert.ok(native, "this Node has AbortSignal.any to remove");
  Object.defineProperty(AbortSignal, "any", {
    value: undefined,
    configurable: true,
  });
  try {
    check();
  } finally {
    Object.defineProperty(AbortSignal, "any", native);
  }
}

test("without AbortSignal.any, the combined signal still aborts when either signal does", () => {
  withoutAbortSignalAny(() => {
    assert.equal(typeof AbortSignal.any, "undefined");

    const unmounted = new AbortController();
    const timeout = new AbortController();
    const combined = anySignal([unmounted.signal, timeout.signal]);
    assert.equal(combined.aborted, false);
    timeout.abort("timed out");
    assert.equal(combined.aborted, true);
    assert.equal(combined.reason, "timed out");

    const first = new AbortController();
    const combinedFirst = anySignal([
      first.signal,
      new AbortController().signal,
    ]);
    first.abort("unmounted");
    assert.equal(combinedFirst.reason, "unmounted");

    // A signal that has already aborted aborts the combined one at once.
    const done = new AbortController();
    done.abort("already");
    assert.equal(anySignal([done.signal]).reason, "already");
  });
});

test("with AbortSignal.any, the combined signal aborts when either signal does", () => {
  const controller = new AbortController();
  const combined = anySignal([controller.signal, new AbortController().signal]);
  controller.abort("unmounted");
  assert.equal(combined.aborted, true);
  assert.equal(combined.reason, "unmounted");
});

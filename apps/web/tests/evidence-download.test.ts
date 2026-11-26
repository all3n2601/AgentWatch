import assert from "node:assert/strict";
import test from "node:test";
import { downloadFailure, downloadFilename } from "../lib/evidence-download";

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

test("a failed download says why, with the status, and that the page is unchanged", () => {
  const unavailable = downloadFailure(
    { status: 503, body: { error: "Saved artifact unavailable" } },
    "spans.jsonl",
  );
  assert.match(unavailable, /^spans\.jsonl could not be downloaded/);
  assert.match(unavailable, /Run history is unavailable/);
  assert.match(unavailable, /HTTP 503/);
  assert.match(unavailable, /run shown here is unchanged/);

  // The route's own reason is kept when it explains more (a rewritten local artifact).
  assert.match(
    downloadFailure(
      {
        status: 409,
        body: { error: "The local artifact files now hold a different run." },
      },
      "results.json",
    ),
    /now hold a different run.*HTTP 409/,
  );
  assert.match(
    downloadFailure({ status: 404, body: null }, "results.json"),
    /not found.*HTTP 404/,
  );
  // A body that is not the route's error shape is not shown to the viewer.
  assert.doesNotMatch(
    downloadFailure(
      { status: 500, body: "<html>stack</html>" },
      "results.json",
    ),
    /stack/,
  );
  assert.match(
    downloadFailure({ status: null, body: null }, "results.json"),
    /could not be reached/,
  );
});

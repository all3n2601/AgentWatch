import type { APIRequestContext } from "@playwright/test";
import { z } from "zod";

// Shared by the fixture API (e2e/fixture-api.ts) and the browser tests, which register
// synthetic runs and step them through scripted states.

export const FIXTURE_API_PORT = 8199;
export const WEB_PORT = 3199;
const CONTROL_URL = `http://127.0.0.1:${FIXTURE_API_PORT}/__control`;

const savedRecord = z.looseObject({ sequence: z.number().int() });
export const fixtureRunSchema = z.object({
  summary: z.looseObject({ run_id: z.uuid(), sequence: z.number().int() }),
  /** Served in order; a test advances to the next state explicitly. */
  states: z
    .array(
      z.union([
        z.object({
          snapshot: z.looseObject({ records: z.array(savedRecord) }),
          delayMs: z.number().int().nonnegative().optional(),
        }),
        z.object({
          status: z.number().int().min(400).max(599),
          body: z.unknown().optional(),
          delayMs: z.number().int().nonnegative().optional(),
        }),
      ]),
    )
    .min(1),
});
export type FixtureRun = z.infer<typeof fixtureRunSchema>;
export type FixtureState = FixtureRun["states"][number];

export async function resetFixtureApi(request: APIRequestContext) {
  const response = await request.post(`${CONTROL_URL}/reset`);
  if (!response.ok()) throw new Error("Fixture API reset failed");
}

export async function registerRun(
  request: APIRequestContext,
  fixture: FixtureRun,
) {
  const response = await request.put(
    `${CONTROL_URL}/runs/${fixture.summary.run_id}`,
    { data: fixture },
  );
  if (!response.ok())
    throw new Error(`Registering ${fixture.summary.run_id} failed`);
}

/** Moves a run to its next scripted state; the last state then repeats. */
export async function advanceRun(request: APIRequestContext, runId: string) {
  const response = await request.post(`${CONTROL_URL}/runs/${runId}/advance`);
  if (!response.ok()) throw new Error(`Advancing ${runId} failed`);
}

/** The `after` cursor of every answered read of a run, in order. */
export async function runReads(
  request: APIRequestContext,
  runId: string,
): Promise<number[]> {
  const response = await request.get(`${CONTROL_URL}/runs/${runId}/reads`);
  return z.array(z.number()).parse(await response.json());
}

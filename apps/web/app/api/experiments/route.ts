import { loadExperiments } from "@/lib/experiments";
export const dynamic = "force-dynamic";
export async function GET(request: Request) {
  const raw = new URL(request.url).searchParams.get("before");
  const before = raw === null ? undefined : Number(raw);
  if (before !== undefined && (!Number.isSafeInteger(before) || before <= 0)) {
    return Response.json({ error: "Invalid history cursor" }, { status: 400 });
  }
  try {
    return Response.json(await loadExperiments(before), {
      headers: { "Cache-Control": "no-store" },
    });
  } catch {
    return Response.json({ error: "History unavailable" }, { status: 503 });
  }
}

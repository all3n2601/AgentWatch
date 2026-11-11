"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  createPolicyBrowserApi,
  type PolicyFailure,
  type PolicyResult,
} from "@/lib/policy-api";
import {
  MAX_OPERATOR_LENGTH,
  MAX_REASON_LENGTH,
  type PolicyChangeRequest,
  type PolicyControls,
  type PolicyVersionView,
} from "@/lib/policy-api-schema";
import {
  limitsFromForm,
  policyStatusText,
  versionLimitsText,
} from "@/lib/policy-view";

// Minimal O02 controls: plain layout, complete states. Styling is refined in O05.

const policyApi = createPolicyBrowserApi();

type ChangeIntent = Omit<PolicyChangeRequest, "request_id">;
type Notice = { tone: "error" | "success"; text: string } | null;
type History = { versions: PolicyVersionView[]; nextBefore: number | null };

function useRequestIds() {
  // A request ID is reused only to resend the same unconfirmed change, so a retry after a
  // timeout is idempotent; any other change gets a fresh ID.
  const last = useRef<{ key: string; requestId: string } | null>(null);
  return {
    idFor(intent: ChangeIntent) {
      const key = JSON.stringify(intent);
      if (last.current?.key !== key)
        last.current = { key, requestId: crypto.randomUUID() };
      return last.current.requestId;
    },
    settle() {
      last.current = null;
    },
  };
}

function changeFailureText(failure: PolicyFailure) {
  if (failure.kind === "conflict")
    return `Not saved: the policy changed since this page loaded (${failure.message}). The latest policy is shown; review it and submit again.`;
  if (failure.kind === "invalid_request")
    return `Not saved: ${failure.message}`;
  return `Not confirmed: ${failure.message} Retry sends the same request, so it is recorded at most once.`;
}

function VersionRow({
  version,
  active,
  disabled,
  onRestore,
}: {
  version: PolicyVersionView;
  active: boolean;
  disabled: boolean;
  onRestore: () => void;
}) {
  return (
    <li className="flex flex-wrap items-start gap-3 rounded-lg border p-3 text-xs">
      <div className="min-w-0 flex-1 space-y-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-semibold">v{version.version}</span>
          <Badge variant="outline">{version.action}</Badge>
          {version.restored_from !== null && (
            <span>from v{version.restored_from}</span>
          )}
          {active && <Badge>Active</Badge>}
        </div>
        <p>{versionLimitsText(version)}</p>
        <p className="break-words text-muted-foreground">
          {version.created_at ? (
            <time dateTime={version.created_at} suppressHydrationWarning>
              {new Date(version.created_at).toLocaleString()}
            </time>
          ) : (
            "Built-in baseline"
          )}
          {version.operator && ` · ${version.operator}`}
          {version.reason && ` · ${version.reason}`}
        </p>
      </div>
      {!active && (
        <Button
          variant="outline"
          className="min-h-11"
          disabled={disabled}
          onClick={onRestore}
          aria-label={`Restore version ${version.version}`}
        >
          Restore
        </Button>
      )}
    </li>
  );
}

/** Active coding-agent concurrency limits, a change form and the version history. */
export function ConcurrencyControls() {
  const [loaded, setLoaded] = useState<PolicyResult<PolicyControls> | null>(
    null,
  );
  const [history, setHistory] = useState<History>({
    versions: [],
    nextBefore: null,
  });
  const [form, setForm] = useState({
    inference: "",
    tests: "",
    operator: "",
    reason: "",
  });
  const [pending, setPending] = useState(false);
  const [notice, setNotice] = useState<Notice>(null);
  const [retry, setRetry] = useState<ChangeIntent | null>(null);
  const busy = useRef(false);
  const requestIds = useRequestIds();

  // Reading again replaces older pages: the first page is the current history.
  const show = useCallback((result: PolicyResult<PolicyControls>) => {
    setLoaded(result);
    if (result.ok)
      setHistory({
        versions: result.data.history.versions,
        nextBefore: result.data.history.next_before,
      });
  }, []);
  const load = async () => show(await policyApi.controls());

  useEffect(() => {
    const controller = new AbortController();
    void policyApi.controls(undefined, controller.signal).then((result) => {
      if (!controller.signal.aborted) show(result);
    });
    return () => controller.abort();
  }, [show]);

  const active = loaded?.ok ? loaded.data.active : null;
  const status = loaded?.ok
    ? { ok: true as const, data: loaded.data.active }
    : loaded;

  async function submit(intent: ChangeIntent) {
    if (busy.current) return;
    busy.current = true;
    setPending(true);
    setNotice(null);
    const result = await policyApi.change({
      ...intent,
      request_id: requestIds.idFor(intent),
    });
    // Without a valid answer the change may still have been stored, so keep its request ID.
    const unconfirmed =
      !result.ok &&
      (result.kind === "unavailable" || result.kind === "invalid_response");
    if (!unconfirmed) requestIds.settle();
    setRetry(unconfirmed ? intent : null);
    if (result.ok)
      setNotice({
        tone: "success",
        text: result.data.created
          ? `Saved as version ${result.data.policy.version}.`
          : `Already recorded as version ${result.data.policy.version}.`,
      });
    else setNotice({ tone: "error", text: changeFailureText(result) });
    if (result.ok || result.kind === "conflict") await load();
    busy.current = false;
    setPending(false);
  }

  function attribution(): Pick<ChangeIntent, "operator" | "reason"> | null {
    const operator = form.operator.trim();
    const reason = form.reason.trim();
    if (operator && reason) return { operator, reason };
    setNotice({
      tone: "error",
      text: "Enter an operator name and a reason before changing the policy.",
    });
    return null;
  }

  function setLimits(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!active) return;
    const limits = limitsFromForm(form);
    if (!limits.ok) return setNotice({ tone: "error", text: limits.message });
    const who = attribution();
    if (who)
      void submit({
        action: "set",
        expected_version: active.version,
        limits: limits.limits,
        ...who,
      });
  }

  function disable() {
    if (!active) return;
    const who = attribution();
    if (!who) return;
    if (!window.confirm("Remove all concurrency limits for coding-agent runs?"))
      return;
    void submit({
      action: "disable",
      expected_version: active.version,
      ...who,
    });
  }

  function restore(version: number) {
    if (!active) return;
    const who = attribution();
    if (who)
      void submit({
        action: "restore",
        expected_version: active.version,
        restored_from: version,
        ...who,
      });
  }

  async function loadOlder() {
    if (busy.current || history.nextBefore === null) return;
    busy.current = true;
    setPending(true);
    const result = await policyApi.controls(history.nextBefore);
    if (result.ok)
      setHistory((old) => ({
        versions: [
          ...old.versions,
          ...result.data.history.versions.filter(
            (v) => !old.versions.some((o) => o.version === v.version),
          ),
        ],
        nextBefore: result.data.history.next_before,
      }));
    else
      setNotice({
        tone: "error",
        text: `Older versions could not be loaded: ${result.message}`,
      });
    busy.current = false;
    setPending(false);
  }

  const field = (name: keyof typeof form) => ({
    value: form[name],
    disabled: pending || !active,
    onChange: (event: React.ChangeEvent<HTMLInputElement>) =>
      setForm((old) => ({ ...old, [name]: event.target.value })),
  });

  return (
    <div className="grid min-w-0 grid-cols-1 gap-5 xl:grid-cols-2">
      <Card className="gap-4 p-5 sm:p-6">
        <div>
          <h2 className="text-sm font-semibold">Active policy</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            Target coding-agent · Scope: runner process · enforced by each agent
            runner process for its own work, not globally
          </p>
        </div>
        <p role="status" className="text-lg font-semibold">
          {policyStatusText(status)}
        </p>
        {loaded && !loaded.ok && (
          <div className="space-y-2 text-xs">
            <p className="break-words text-amber-200">{loaded.message}</p>
            <Button
              variant="outline"
              className="min-h-11"
              onClick={() => {
                setLoaded(null);
                void load();
              }}
            >
              Read again
            </Button>
          </div>
        )}
        {active && (
          <dl className="grid grid-cols-1 gap-2 text-xs sm:grid-cols-2">
            {Object.entries(active.controlled_services).map(
              ([control, service]) => (
                <div key={control}>
                  <dt className="text-muted-foreground">{control}</dt>
                  <dd>
                    {service.service_id} ({service.resource_class})
                  </dd>
                </div>
              ),
            )}
          </dl>
        )}
        <form className="space-y-3" onSubmit={setLimits} noValidate>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <label className="space-y-1 text-xs">
              <span>Inference requests</span>
              <Input
                type="text"
                inputMode="numeric"
                placeholder="No limit"
                className="min-h-11"
                {...field("inference")}
              />
            </label>
            <label className="space-y-1 text-xs">
              <span>Test jobs</span>
              <Input
                type="text"
                inputMode="numeric"
                placeholder="No limit"
                className="min-h-11"
                {...field("tests")}
              />
            </label>
            <label className="space-y-1 text-xs">
              <span>Operator</span>
              <Input
                autoComplete="name"
                maxLength={MAX_OPERATOR_LENGTH}
                className="min-h-11"
                {...field("operator")}
              />
            </label>
            <label className="space-y-1 text-xs">
              <span>Reason</span>
              <Input
                maxLength={MAX_REASON_LENGTH}
                className="min-h-11"
                {...field("reason")}
              />
            </label>
          </div>
          <p className="text-[11px] text-muted-foreground">
            Leave a limit empty for no limit. A new version is recorded and
            linked to coding-agent runs open now or registered later; nothing
            enforces it yet. Changes are checked against version{" "}
            {active ? active.version : "–"}.
          </p>
          <div className="flex flex-wrap gap-2">
            <Button
              type="submit"
              className="min-h-11"
              disabled={pending || !active}
            >
              {pending ? "Saving…" : "Set limits"}
            </Button>
            <Button
              type="button"
              variant="outline"
              className="min-h-11"
              disabled={pending || !active}
              onClick={disable}
            >
              Disable limits
            </Button>
            {retry && (
              <Button
                type="button"
                variant="outline"
                className="min-h-11"
                disabled={pending}
                onClick={() => void submit(retry)}
              >
                Retry
              </Button>
            )}
          </div>
          <div role="alert" className="min-h-5 text-xs break-words">
            {notice?.tone === "error" && (
              <p className="text-amber-200">{notice.text}</p>
            )}
          </div>
          <div role="status" className="text-xs">
            {notice?.tone === "success" && (
              <p className="text-emerald-300">{notice.text}</p>
            )}
          </div>
        </form>
      </Card>
      <Card className="gap-4 p-5 sm:p-6">
        <h2 className="text-sm font-semibold">Version history</h2>
        {loaded === null ? (
          <p className="text-xs text-muted-foreground">Loading versions…</p>
        ) : !loaded.ok && history.versions.length === 0 ? (
          <p className="text-xs text-muted-foreground">
            Version history is unavailable.
          </p>
        ) : (
          <ul className="space-y-2">
            {history.versions.map((version) => (
              <VersionRow
                key={version.version}
                version={version}
                active={version.version === active?.version}
                disabled={pending || !active}
                onRestore={() => restore(version.version)}
              />
            ))}
          </ul>
        )}
        {history.nextBefore !== null && (
          <Button
            variant="outline"
            className="min-h-11"
            disabled={pending}
            onClick={() => void loadOlder()}
          >
            Load older versions
          </Button>
        )}
      </Card>
    </div>
  );
}

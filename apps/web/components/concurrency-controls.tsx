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
  conflictCause,
  controlsConsistent,
  EFFECTIVE_LIMITS_UNAVAILABLE,
  formFromLimits,
  limitChanges,
  limitsFromForm,
  policyStatusText,
  versionLimitsText,
} from "@/lib/policy-view";

// Manual controls only: every change is an explicit operator action against a version the
// operator has seen. Nothing here changes limits automatically.

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
  if (failure.kind === "conflict") {
    const cause = conflictCause(failure.message);
    if (cause === "stale_version")
      return `Not saved: another change was recorded after this page read the policy (${failure.message}). The latest policy and its limits are shown; review them and submit again.`;
    if (cause === "request_reused")
      return `Not saved: this request ID was already used for a different change (${failure.message}). Submit again to send it as a new request.`;
    return `Not saved: the policy service reported a conflict (${failure.message}). The latest policy is shown; review it and submit again.`;
  }
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
  const [readAt, setReadAt] = useState<string | null>(null);
  const busy = useRef(false);
  const requestIds = useRequestIds();

  // Reading again replaces older pages: the first page is the current history. The limit
  // inputs restart from the active limits, so a `set` never drops a control left alone.
  const show = useCallback((result: PolicyResult<PolicyControls>) => {
    setLoaded(result);
    if (!result.ok) return;
    setReadAt(new Date().toISOString());
    setHistory({
      versions: result.data.history.versions,
      nextBefore: result.data.history.next_before,
    });
    setForm((old) => ({
      ...old,
      ...formFromLimits(result.data.active.limits),
    }));
  }, []);
  const load = async () => show(await policyApi.controls());

  useEffect(() => {
    const controller = new AbortController();
    void policyApi.controls(undefined, controller.signal).then((result) => {
      if (!controller.signal.aborted) show(result);
    });
    return () => controller.abort();
  }, [show]);

  const inconsistent = loaded?.ok === true && !controlsConsistent(loaded.data);
  // Changes need a version the operator can see in full; an inconsistent read offers none.
  const active = loaded?.ok && !inconsistent ? loaded.data.active : null;
  const status = loaded?.ok
    ? { ok: true as const, data: loaded.data.active }
    : loaded;
  const proposed = active ? limitsFromForm(form) : null;
  const changes =
    active && proposed?.ok ? limitChanges(active.limits, proposed.limits) : [];

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
    if (limitChanges(active.limits, limits.limits).length === 0)
      return setNotice({
        tone: "error",
        text: `These limits match active version ${active.version}; nothing to change.`,
      });
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

  function restore(version: PolicyVersionView) {
    if (!active) return;
    const who = attribution();
    if (!who) return;
    if (
      !window.confirm(
        `Restore the limits of version ${version.version} (${versionLimitsText(version)})? This records a new version.`,
      )
    )
      return;
    void submit({
      action: "restore",
      expected_version: active.version,
      restored_from: version.version,
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
        {loaded && (
          <div className="space-y-2 text-xs">
            {!loaded.ok && (
              <p className="break-words text-amber-200">{loaded.message}</p>
            )}
            {inconsistent && (
              <p className="break-words text-amber-200">
                Stale read: the active version and the newest history entry
                disagree, so another change may have been saved while this page
                was reading. Changes are paused until the policy is read again.
              </p>
            )}
            {readAt && (
              <p className="text-muted-foreground">
                {loaded.ok ? "Read at" : "Last successful read at"}{" "}
                <time dateTime={readAt} suppressHydrationWarning>
                  {new Date(readAt).toLocaleTimeString()}
                </time>
                . Other operators can change the policy at any time; a stale
                version is rejected when you save.
              </p>
            )}
            <Button
              variant="outline"
              className="min-h-11"
              disabled={pending}
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
            <div className="sm:col-span-2">
              <dt className="text-muted-foreground">
                Limits applied by runners
              </dt>
              <dd className="break-words">{EFFECTIVE_LIMITS_UNAVAILABLE}</dd>
            </div>
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
            The inputs start from the active limits; leave a limit empty for no
            limit. A new version is recorded and linked to coding-agent runs
            open now or registered later. Each agent runner process applies it
            to model requests and test jobs it admits afterwards; work already
            running is not stopped. Changes are checked against version{" "}
            {active ? active.version : "–"}.
          </p>
          {active && proposed?.ok && (
            <div className="text-xs">
              <h3 className="font-semibold">
                Proposed change from version {active.version}
              </h3>
              {changes.length === 0 ? (
                <p className="text-muted-foreground">
                  No change: these are the active limits.
                </p>
              ) : (
                <ul className="mt-1 space-y-1">
                  {changes.map((change) => (
                    <li key={change.control}>
                      {change.control}: {change.active} → {change.proposed}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
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
                onRestore={() => restore(version)}
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

"use client";
import { useCallback, useEffect, useId, useRef, useState } from "react";
import { RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { FieldRow, GlassInput } from "@/components/ui/field-row";
import { InsetCard } from "@/components/ui/glass-panel";
import { StatusBadge } from "@/components/ui/status-badge";
import { AlertBanner } from "@/components/widgets/alert-banner";
import { KeyValueList } from "@/components/widgets/evidence-lists";
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
type Notice = {
  tone: "danger" | "success";
  title: string;
  detail?: string;
} | null;
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

function changeFailureNotice(failure: PolicyFailure): Notice {
  const notSaved = (detail: string) => ({
    tone: "danger" as const,
    title: "Not saved",
    detail,
  });
  if (failure.kind === "conflict") {
    const cause = conflictCause(failure.message);
    if (cause === "stale_version")
      return notSaved(
        `Another change was recorded after this page read the policy (${failure.message}). The latest policy and its limits are shown; review them and submit again.`,
      );
    if (cause === "request_reused")
      return notSaved(
        `This request ID was already used for a different change (${failure.message}). Submit again to send it as a new request.`,
      );
    return notSaved(
      `The policy service reported a conflict (${failure.message}). The latest policy is shown; review it and submit again.`,
    );
  }
  if (failure.kind === "invalid_request") return notSaved(failure.message);
  return {
    tone: "danger",
    title: "Not confirmed",
    detail: `${failure.message} Retry sends the same request, so it is recorded at most once.`,
  };
}

/** Whether changes can be made, as a label that never relies on colour alone. */
function readState(
  loaded: PolicyResult<PolicyControls> | null,
  inconsistent: boolean,
) {
  if (loaded === null) return { label: "Loading", tone: "info" as const };
  if (!loaded.ok) return { label: "Unavailable", tone: "danger" as const };
  if (inconsistent) return { label: "Paused", tone: "warning" as const };
  return { label: "Read", tone: "info" as const };
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
    <li className="flex flex-wrap items-start gap-3 py-3 text-sm first:pt-0 last:pb-0">
      <div className="min-w-0 flex-1 space-y-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-semibold text-ink">v{version.version}</span>
          <StatusBadge label={version.action} tone="info" appearance="plain" />
          {version.restored_from !== null && (
            <span className="text-ink-muted">
              from v{version.restored_from}
            </span>
          )}
          {active && <StatusBadge label="Active" tone="success" />}
        </div>
        <p className="text-ink">{versionLimitsText(version)}</p>
        <p className="break-words text-ink-muted">
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
          variant="glass"
          size="pill"
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
        title: result.data.created
          ? `Saved as version ${result.data.policy.version}.`
          : `Already recorded as version ${result.data.policy.version}.`,
      });
    else setNotice(changeFailureNotice(result));
    if (result.ok || result.kind === "conflict") await load();
    busy.current = false;
    setPending(false);
  }

  function attribution(): Pick<ChangeIntent, "operator" | "reason"> | null {
    const operator = form.operator.trim();
    const reason = form.reason.trim();
    if (operator && reason) return { operator, reason };
    setNotice({
      tone: "danger",
      title: "Not saved",
      detail: "Enter an operator name and a reason before changing the policy.",
    });
    return null;
  }

  function setLimits(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!active) return;
    const limits = limitsFromForm(form);
    if (!limits.ok)
      return setNotice({
        tone: "danger",
        title: "Not saved",
        detail: limits.message,
      });
    if (limitChanges(active.limits, limits.limits).length === 0)
      return setNotice({
        tone: "danger",
        title: "Not saved",
        detail: `These limits match active version ${active.version}; nothing to change.`,
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
        tone: "danger",
        title: "Older versions could not be loaded",
        detail: result.message,
      });
    busy.current = false;
    setPending(false);
  }

  const idPrefix = useId();
  const field = (name: keyof typeof form) => ({
    id: `${idPrefix}-${name}`,
    value: form[name],
    disabled: pending || !active,
    onChange: (event: React.ChangeEvent<HTMLInputElement>) =>
      setForm((old) => ({ ...old, [name]: event.target.value })),
  });

  const state = readState(loaded, inconsistent);
  return (
    <div className="grid min-w-0 grid-cols-1 gap-4 xl:grid-cols-2">
      <InsetCard className="min-w-0 space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="text-base font-semibold text-ink">
                Active policy
              </h3>
              <StatusBadge label={state.label} tone={state.tone} />
            </div>
            <p className="mt-1 text-sm text-ink-muted">
              Target coding-agent · Scope: runner process · enforced by each
              agent runner process for its own work, not globally
            </p>
          </div>
          <Button
            variant="glass"
            size="pill"
            disabled={pending || loaded === null}
            onClick={() => {
              setLoaded(null);
              void load();
            }}
          >
            <RefreshCw aria-hidden />
            Read again
          </Button>
        </div>
        <p role="status" className="text-xl font-semibold text-ink">
          {policyStatusText(status)}
        </p>
        {loaded && !loaded.ok && (
          <AlertBanner
            tone="danger"
            headingLevel="h3"
            title="Policy could not be read"
            detail={`${loaded.message} Changes are disabled until the policy is read.`}
          />
        )}
        {inconsistent && (
          <AlertBanner
            tone="warning"
            headingLevel="h3"
            title="Stale read"
            detail="The active version and the newest history entry disagree, so another change may have been saved while this page was reading. Changes are paused until the policy is read again."
          />
        )}
        <KeyValueList
          dense
          items={[
            {
              id: "read-at",
              label: loaded?.ok ? "Read at" : "Last successful read",
              value: readAt ? (
                <time dateTime={readAt} suppressHydrationWarning>
                  {new Date(readAt).toLocaleTimeString()}
                </time>
              ) : (
                "Not read yet"
              ),
            },
            ...(active
              ? Object.entries(active.controlled_services).map(
                  ([control, service]) => ({
                    id: control,
                    label: control,
                    value: `${service.service_id} (${service.resource_class})`,
                  }),
                )
              : []),
            ...(active
              ? [
                  {
                    id: "applied",
                    label: "Limits applied by runners",
                    value: "Not available",
                    status: { label: "Not recorded", tone: "warning" as const },
                  },
                ]
              : []),
          ]}
        />
        <p className="text-sm text-ink-muted">
          Other operators can change the policy at any time; a stale version is
          rejected when you save.
          {active && ` ${EFFECTIVE_LIMITS_UNAVAILABLE}`}
        </p>
        <form className="space-y-3" onSubmit={setLimits} noValidate>
          <div>
            <FieldRow
              label="Inference requests"
              htmlFor={`${idPrefix}-inference`}
            >
              <GlassInput
                type="text"
                inputMode="numeric"
                placeholder="No limit"
                className="disabled:opacity-60"
                {...field("inference")}
              />
            </FieldRow>
            <FieldRow label="Test jobs" htmlFor={`${idPrefix}-tests`}>
              <GlassInput
                type="text"
                inputMode="numeric"
                placeholder="No limit"
                className="disabled:opacity-60"
                {...field("tests")}
              />
            </FieldRow>
            <FieldRow label="Operator" htmlFor={`${idPrefix}-operator`}>
              <GlassInput
                autoComplete="name"
                maxLength={MAX_OPERATOR_LENGTH}
                className="disabled:opacity-60"
                {...field("operator")}
              />
            </FieldRow>
            <FieldRow label="Reason" htmlFor={`${idPrefix}-reason`}>
              <GlassInput
                maxLength={MAX_REASON_LENGTH}
                className="disabled:opacity-60"
                {...field("reason")}
              />
            </FieldRow>
          </div>
          <p className="text-sm text-ink-muted">
            The inputs start from the active limits; leave a limit empty for no
            limit. A new version is recorded and linked to coding-agent runs
            open now or registered later. Each agent runner process applies it
            to model requests and test jobs it admits afterwards; work already
            running is not stopped. Changes are checked against version{" "}
            {active ? active.version : "–"}.
          </p>
          {active && proposed?.ok && (
            <div className="rounded-xl border border-line p-3 text-sm text-ink">
              <h4 className="font-semibold">
                Proposed change from version {active.version}
              </h4>
              {changes.length === 0 ? (
                <p className="text-ink-muted">
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
              variant="brand"
              size="pill"
              disabled={pending || !active}
            >
              {pending ? "Saving…" : "Set limits"}
            </Button>
            <Button
              type="button"
              variant="glass"
              size="pill"
              disabled={pending || !active}
              onClick={disable}
            >
              Disable limits
            </Button>
            {retry && (
              <Button
                type="button"
                variant="glass"
                size="pill"
                disabled={pending}
                onClick={() => void submit(retry)}
              >
                Retry
              </Button>
            )}
          </div>
          {notice && (
            <AlertBanner
              tone={notice.tone}
              headingLevel="h3"
              title={notice.title}
              detail={notice.detail}
              className="py-3"
            />
          )}
        </form>
      </InsetCard>
      <InsetCard className="min-w-0 space-y-4">
        <h3 className="text-base font-semibold text-ink">Version history</h3>
        {loaded === null ? (
          <p className="text-sm text-ink-muted">Loading versions…</p>
        ) : !loaded.ok && history.versions.length === 0 ? (
          <p className="text-sm text-ink-muted">
            Version history is unavailable.
          </p>
        ) : (
          <ul className="divide-y divide-line">
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
            variant="glass"
            size="pill"
            disabled={pending}
            onClick={() => void loadOlder()}
          >
            Load older versions
          </Button>
        )}
      </InsetCard>
    </div>
  );
}

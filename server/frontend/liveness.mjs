export function livenessRuntimeLabel(enabled) {
  if (enabled === true) return "Liveness enabled";
  if (enabled === false) return "Liveness disabled";
  return "Liveness status unknown";
}

export function livenessResultText(result, { translate, formatScore, compact = false, side } = {}) {
  if (!result) return "";
  let message;
  if (result.status === "input_rejected") {
    const reason = typeof result.reason === "string" && result.reason.trim() ? result.reason : "Liveness input rejected";
    message = translate(compact ? "Liveness: input rejected" : reason);
  } else {
    message = `${translate(result.is_live === true ? "Liveness passed" : "Liveness failed")} · ${formatScore(result.live_score)}`;
  }
  const label = side === "source" ? "Source face" : side === "target" ? "Target face" : "";
  return label ? `${translate(label)} · ${message}` : message;
}

// Localize stable codes, never a backend sentence containing paths or details.
export const LIVENESS_MESSAGES = Object.freeze({
  config_file_missing: "Set INSIGHTFACE_CONFIG_FILE to an editable server.toml.",
  config_file_not_regular: "Use an existing regular server.toml file, without a symbolic link.",
  config_file_mount: "Mount the configuration directory writable, then recreate the container.",
  config_not_writable: "Give the Server user write access to the configuration file and directory, then recreate the container.",
  addon_directory_not_writable: "Mount /models writable and allow the Server to create or write its addons directory, then recreate the container.",
  addon_config_invalid: "Correct the unreadable or invalid server.toml before enabling liveness.",
  addon_model_invalid: "The local liveness model is invalid. Restore the official file or remove it and retry; it will not be overwritten automatically.",
  server_stopping: "Server is shutting down. Try again after it restarts.",
  addon_download_failed: "Download or verification failed. Check the Server network and proxy settings, then retry. The startup configuration was not changed.",
  addon_config_save_failed: "Could not save startup settings. Check configuration, directory permissions, and concurrent edits, then retry. The installed model can be reused.",
  addon_job_in_progress: "Another Server is preparing liveness. Wait and refresh.",
  addon_management_unavailable: "Liveness preparation is unavailable. Refresh its status for configuration and permission details.",
  invalid_addon_request: "Send an empty JSON object.",
  json_required: "Use Content-Type: application/json and an empty JSON object.",
  origin_not_allowed: "This website is not allowed to change Server settings. Open the Server Web UI directly or configure an allowed CORS origin.",
  liveness_fake: "Liveness failed",
  liveness_input_rejected: "Liveness input rejected",
  liveness_unavailable: "Liveness inference is unavailable. Check the Server logs before retrying.",
  collection_model_mismatch: "This Collection belongs to another model. Use its original model or create a new Collection and register people again.",
});

export function livenessMessage(code, fallback, translate) {
  return translate(LIVENESS_MESSAGES[code] ?? fallback ?? "");
}

export function livenessErrorText(error, { translate, formatScore }) {
  const detail = livenessResultText(error.details?.liveness, { translate, formatScore, side: error.details?.side });
  if (detail && ["liveness_input_rejected", "liveness_fake"].includes(error.code)) return detail;
  const message = `${error.code}: ${livenessMessage(error.code, error.message, translate)}`;
  return [message, detail].filter(Boolean).join(" · ");
}

export function livenessManagementView({ status, error, submitting, loading } = {}) {
  const known = Boolean(status) && !error;
  const downloading = status?.state === "downloading";
  const ready = Boolean(status?.restart_required && status?.configured_enabled && status?.installed);
  return {
    runtimeLabel: livenessRuntimeLabel(known ? status.enabled : null),
    installedLabel: known ? (status.installed ? "Installed" : "Not installed") : "Unknown",
    afterRestartLabel: known ? (status.configured_enabled ? "enabled" : "disabled") : "unknown",
    showAction: known && !status.enabled && !ready && status.can_enable,
    actionDisabled: Boolean(submitting || loading || downloading),
    actionLabel: submitting ? "Saving…" : downloading ? "Downloading and verifying…"
      : status?.installed ? "Enable after restart" : "Download and enable after restart",
    notice: downloading ? "Downloading and verifying the model. You can leave this page; the server will continue."
      : ready ? "The model is installed and the startup configuration is saved. Restart Server manually to enable liveness."
        : known && status.restart_required ? "Restart Server manually to apply the saved liveness setting." : "",
    unavailableReason: known && !status.enabled && !ready && !status.can_enable ? status.unavailable_reason : "",
    unavailableCode: status?.unavailable_code,
    error: error || status?.error || null,
  };
}

// The download belongs to the server. Navigating away only stops status polling.
export function createLivenessManager({
  client,
  onChange,
  setTimer = (callback, delay) => globalThis.setTimeout(callback, delay),
  clearTimer = (timer) => globalThis.clearTimeout(timer),
  pollInterval = 1500,
}) {
  let snapshot = { status: null, error: null, submitting: false, loading: false };
  let active = false;
  let timer = null;
  let revision = 0;

  function emit(change) {
    snapshot = { ...snapshot, ...change };
    if (active) onChange(snapshot);
  }

  function cancelPoll() {
    if (timer !== null) clearTimer(timer);
    timer = null;
  }

  function schedulePoll() {
    cancelPoll();
    if (active && snapshot.status?.state === "downloading") {
      timer = setTimer(() => refresh(), pollInterval);
    }
  }

  async function refresh() {
    if (!active || snapshot.submitting) return;
    cancelPoll();
    const requestRevision = ++revision;
    emit({ loading: true });
    try {
      const status = await client.liveness();
      if (active && requestRevision === revision) emit({ status, error: null });
    } catch (error) {
      if (active && requestRevision === revision) emit({ error });
    } finally {
      if (active && requestRevision === revision) {
        emit({ loading: false });
        schedulePoll();
      }
    }
  }

  async function enable() {
    const view = livenessManagementView(snapshot);
    if (!active || !view.showAction || view.actionDisabled) return false;
    cancelPoll();
    const requestRevision = ++revision;
    emit({ submitting: true, error: null });
    try {
      const status = await client.enableLiveness();
      if (active && requestRevision === revision) emit({ status, error: null });
      return true;
    } catch (error) {
      if (active && requestRevision === revision) emit({ error });
      return false;
    } finally {
      snapshot = { ...snapshot, submitting: false };
      if (active && requestRevision === revision) {
        emit({});
        schedulePoll();
      } else if (active) {
        void refresh();
      }
    }
  }

  return {
    start() { active = true; return refresh(); },
    stop() { active = false; revision += 1; cancelPoll(); },
    refresh,
    enable,
  };
}

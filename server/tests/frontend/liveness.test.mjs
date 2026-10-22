import assert from "node:assert/strict";
import test from "node:test";
import { ApiClient, ApiError } from "../../frontend/api.mjs";
import { formatScore } from "../../frontend/core.mjs";
import { hasTranslation, LANGUAGES, t } from "../../frontend/i18n.mjs";
import {
  LIVENESS_MESSAGES,
  createLivenessManager,
  livenessErrorText,
  livenessManagementView,
  livenessMessage,
  livenessResultText,
  livenessRuntimeLabel,
} from "../../frontend/liveness.mjs";

const inputReason = "Insufficient image area around the face for liveness detection. Move the face toward the center, step back from the camera, or use a less tightly cropped image.";
const rejectedInput = Object.freeze({ status: "input_rejected", is_live: null, live_score: null, reason: inputReason });
const resultOptions = (language = "en", options = {}) => ({ translate: (message) => t(message, {}, language), formatScore, ...options });

test("liveness reasons are translated for presentation in every language without changing API data", () => {
  for (const { code } of LANGUAGES) {
    const options = resultOptions(code);
    if (code !== "en") {
      assert.ok(hasTranslation(inputReason, code), `${code}: full input reason`);
      assert.ok(hasTranslation("Liveness: input rejected", code), `${code}: compact input status`);
      assert.notEqual(options.translate(inputReason), inputReason);
    }
    assert.equal(livenessResultText(rejectedInput, options), options.translate(inputReason));
    assert.equal(livenessResultText(rejectedInput, { ...options, compact: true }), options.translate("Liveness: input rejected"));
    assert.notEqual(livenessResultText(rejectedInput, { ...options, compact: true }), options.translate(inputReason));
  }
  assert.equal(rejectedInput.reason, inputReason);
  assert.match(livenessResultText(rejectedInput, resultOptions("zh")), /人脸周围.*画面中央.*摄像头/);
});

test("legacy liveness results use generic advice and unknown reasons remain unchanged", () => {
  const options = resultOptions("zh");
  for (const reason of [undefined, null, "", "   "]) {
    assert.equal(livenessResultText({ status: "input_rejected", reason }, options), t("Liveness input rejected", {}, "zh"));
  }
  for (const result of [undefined, null]) assert.equal(livenessResultText(result, options), "");
  const reason = '<img src=x onerror="alert(1)"> Future input requirement.';
  const future = Object.freeze({ ...rejectedInput, reason });
  assert.equal(livenessResultText(future, options), reason);
  assert.equal(future.reason, reason);
  for (const is_live of [true, false]) {
    const result = { status: "ok", is_live, live_score: 0.875, reason: "Ignored stale reason" };
    const expected = `${t(is_live ? "Liveness passed" : "Liveness failed", {}, "zh")} · 0.875`;
    assert.equal(livenessResultText(result, options), expected);
    assert.equal(livenessResultText(result, { ...options, compact: true }), expected);
  }
});

test("compare results and liveness errors preserve source and target while other errors retain their cause", () => {
  const options = resultOptions("zh");
  for (const side of ["source", "target"]) {
    const label = t(side === "source" ? "Source face" : "Target face", {}, "zh");
    const expected = `${label} · ${t(inputReason, {}, "zh")}`;
    assert.equal(livenessResultText(rejectedInput, { ...options, side }), expected);
    const error = new ApiError({ code: "liveness_input_rejected", message: "Generic rejection", details: { side, liveness: rejectedInput } });
    assert.equal(livenessErrorText(error, options), expected);
    const secondary = new ApiError({ code: "embedding_unavailable", message: "Face embedding is unavailable.", details: error.details });
    assert.equal(livenessErrorText(secondary, options), `embedding_unavailable: Face embedding is unavailable. · ${expected}`);
    assert.equal(secondary.details.liveness.reason, inputReason);
  }
  assert.equal(livenessResultText(rejectedInput, { ...options, side: "unknown" }), t(inputReason, {}, "zh"));
  const fake = new ApiError({ code: "liveness_fake", message: "Generic fake", details: { liveness: { status: "ok", is_live: false, live_score: 0.1 } } });
  assert.equal(livenessErrorText(fake, options), `${t("Liveness failed", {}, "zh")} · 0.100`);
  const noDetail = new ApiError({ code: "liveness_input_rejected", message: "Generic rejection" });
  assert.equal(livenessErrorText(noDetail, options), `liveness_input_rejected: ${t("Liveness input rejected", {}, "zh")}`);
  assert.equal(livenessErrorText(new ApiError({ code: "future_error", message: "Other failure" }), options), "future_error: Other failure");
});

test("API success and error payloads keep the English reason after localized display", async () => {
  for (const code of [null, "liveness_input_rejected", "embedding_unavailable"]) {
    const client = new ApiClient("http://localhost:18198", {
      fetchFn: async () => new Response(JSON.stringify(code ? {
        error: { code, message: "Operation failed.", details: { side: "target", liveness: rejectedInput } },
      } : { faces: [{ liveness: rejectedInput }] }), { status: code ? 422 : 200 }),
    });
    if (code) {
      await assert.rejects(client.request("/v1/compare"), (error) => {
        const original = JSON.stringify(error.details);
        assert.match(livenessErrorText(error, resultOptions("zh")), /人脸周围/);
        assert.equal(JSON.stringify(error.details), original);
        assert.equal(error.details.liveness.reason, inputReason);
        return error instanceof ApiError && error.code === code;
      });
    } else {
      const payload = await client.request("/v1/detect");
      const original = JSON.stringify(payload);
      assert.match(livenessResultText(payload.faces[0].liveness, resultOptions("zh")), /人脸周围/);
      assert.equal(JSON.stringify(payload), original);
      assert.equal(payload.faces[0].liveness.reason, inputReason);
    }
  }
});

function status(overrides = {}) {
  return {
    enabled: false,
    installed: false,
    configured_enabled: false,
    restart_required: false,
    can_enable: true,
    unavailable_reason: null,
    unavailable_code: null,
    state: "idle",
    error: null,
    ...overrides,
  };
}

function harness(client) {
  const timers = new Map();
  const changes = [];
  let timerId = 0;
  const manager = createLivenessManager({
    client,
    onChange(snapshot) { changes.push(snapshot); },
    setTimer(callback) { const id = ++timerId; timers.set(id, callback); return id; },
    clearTimer(id) { timers.delete(id); },
  });
  return { manager, timers, changes, last: () => changes.at(-1) };
}

test("installed or pending liveness never appears enabled before restart", () => {
  const disabled = livenessManagementView({ status: status() });
  assert.equal(disabled.runtimeLabel, "Liveness disabled");
  assert.equal(disabled.actionLabel, "Download and enable after restart");
  assert.equal(disabled.installedLabel, "Not installed");
  assert.equal(disabled.showAction, true);

  const cached = livenessManagementView({ status: status({ installed: true }) });
  assert.equal(cached.runtimeLabel, "Liveness disabled");
  assert.equal(cached.actionLabel, "Enable after restart");
  assert.equal(cached.afterRestartLabel, "disabled");

  const pending = livenessManagementView({ status: status({
    installed: true, configured_enabled: true, restart_required: true, state: "ready",
  }) });
  assert.equal(pending.runtimeLabel, "Liveness disabled");
  assert.equal(pending.afterRestartLabel, "enabled");
  assert.equal(pending.showAction, false);
  assert.match(pending.notice, /Restart Server manually/);

  const enabled = livenessManagementView({ status: status({
    enabled: true, installed: true, configured_enabled: true,
  }) });
  assert.equal(enabled.runtimeLabel, "Liveness enabled");
  assert.equal(enabled.showAction, false);
  assert.equal(enabled.notice, "");
});

test("management advice is translated by stable code despite changes to backend sentences", () => {
  const translate = (message) => t(message, {}, "zh");
  for (const [code, message] of Object.entries(LIVENESS_MESSAGES)) {
    assert.equal(livenessMessage(code, "A changed English sentence with /custom/model/path", translate), translate(message));
    for (const { code: language } of LANGUAGES.filter(({ code }) => code !== "en")) {
      assert.ok(hasTranslation(message, language), `${code} is missing ${language} translation`);
    }
  }
  assert.match(livenessMessage("config_not_writable", "New backend permission details", translate), /写入权限/);
  assert.equal(livenessMessage("future_error", "Actionable future error details", translate), "Actionable future error details");
  assert.equal(livenessMessage(null, null, translate), "");
  const view = livenessManagementView({ status: status({
    can_enable: false, unavailable_code: "addon_directory_not_writable", unavailable_reason: "The server-specific permission message changed.",
  }) });
  assert.equal(view.unavailableCode, "addon_directory_not_writable");
  const hint = livenessMessage(view.unavailableCode, view.unavailableReason, translate);
  assert.match(hint, /\/models/);
  assert.match(hint, /addons/);
});

test("missing runtime status remains unknown and unavailable deployments explain manual setup", () => {
  for (const value of [null, undefined]) assert.equal(livenessRuntimeLabel(value), "Liveness status unknown");
  assert.equal(livenessRuntimeLabel(false), "Liveness disabled");
  const failure = new Error("Offline");
  const unknown = livenessManagementView({ status: status(), error: failure });
  assert.equal(unknown.runtimeLabel, "Liveness status unknown");
  assert.equal(unknown.installedLabel, "Unknown");
  assert.equal(unknown.showAction, false);
  assert.equal(unknown.error, failure);
  const readOnly = livenessManagementView({ status: status({
    can_enable: false, unavailable_reason: "Startup configuration is read-only.",
  }) });
  assert.equal(readOnly.showAction, false);
  assert.equal(readOnly.unavailableReason, "Startup configuration is read-only.");
});

test("liveness API preserves authentication, asynchronous responses, and actionable errors", async () => {
  const calls = [];
  const client = new ApiClient("http://localhost:18198", {
    fetchFn: async (url, options) => {
      calls.push({ path: url.pathname, ...options });
      if (calls.length === 3) return new Response(JSON.stringify({ error: {
        code: "addon_config_read_only", message: "Startup configuration is read-only.",
      } }), { status: 409 });
      return new Response(JSON.stringify(status({ state: options.method === "POST" ? "downloading" : "idle" })), {
        status: options.method === "POST" ? 202 : 200,
        headers: { "x-request-id": "liveness-request" },
      });
    },
  });
  client.setApiKey("test-key");
  assert.equal((await client.liveness()).state, "idle");
  const accepted = await client.enableLiveness();
  assert.equal(accepted.state, "downloading");
  assert.equal(accepted.enabled, false);
  assert.equal(accepted.request_id, "liveness-request");
  assert.deepEqual(calls.map(({ path, method }) => [path, method]), [
    ["/v1/addons/liveness", "GET"], ["/v1/addons/liveness/enable", "POST"],
  ]);
  assert.equal(calls[1].headers.get("Authorization"), "Bearer test-key");
  assert.equal(calls[1].headers.get("Content-Type"), "application/json");
  assert.equal(calls[1].body, "{}");
  await assert.rejects(client.enableLiveness(), (error) => error instanceof ApiError
    && error.code === "addon_config_read_only" && error.status === 409);
});

test("download polling resumes after navigation and stops when a manual restart is required", async () => {
  let current = status();
  let postCount = 0;
  let resolvePost;
  const h = harness({
    liveness: async () => current,
    enableLiveness: () => {
      postCount += 1;
      return new Promise((resolve) => { resolvePost = resolve; });
    },
  });
  await h.manager.start();
  const submitting = h.manager.enable();
  assert.equal(h.last().submitting, true);
  assert.equal(await h.manager.enable(), false);
  assert.equal(postCount, 1);
  h.manager.stop();
  current = status({ state: "downloading" });
  resolvePost(current);
  await submitting;
  assert.equal(h.timers.size, 0);
  await h.manager.start();
  assert.equal(h.last().status.state, "downloading");
  assert.equal(h.timers.size, 1);
  assert.equal(livenessManagementView(h.last()).actionDisabled, true);
  current = status({ installed: true, configured_enabled: true, restart_required: true, state: "ready" });
  await [...h.timers.values()][0]();
  assert.equal(h.timers.size, 0);
  assert.equal(h.last().status.enabled, false);
  assert.equal(livenessManagementView(h.last()).showAction, false);
});

test("download failure exposes the real error and permits a retry", async () => {
  let current = status({ state: "downloading" });
  let postCount = 0;
  const h = harness({
    liveness: async () => current,
    enableLiveness: async () => { postCount += 1; return status({ state: "downloading" }); },
  });
  await h.manager.start();
  current = status({ state: "error", error: { code: "addon_download_failed", message: "Proxy connection refused." } });
  await [...h.timers.values()][0]();
  const failed = livenessManagementView(h.last());
  assert.equal(failed.error.code, "addon_download_failed");
  assert.equal(failed.error.message, "Proxy connection refused.");
  assert.equal(failed.showAction, true);
  assert.equal(failed.actionDisabled, false);
  assert.equal(h.timers.size, 0);
  await h.manager.enable();
  assert.equal(postCount, 1);
  assert.equal(h.last().status.state, "downloading");
  h.manager.stop();
  assert.equal(h.timers.size, 0);
});

test("failed polls do not misreport disabled state or lose an ongoing download", async () => {
  let disconnected = false;
  const h = harness({
    liveness: async () => {
      if (disconnected) throw new Error("Connection interrupted");
      return status({ state: "downloading" });
    },
  });
  await h.manager.start();
  disconnected = true;
  await [...h.timers.values()][0]();
  assert.equal(livenessManagementView(h.last()).runtimeLabel, "Liveness status unknown");
  assert.equal(h.timers.size, 1);
  disconnected = false;
  await [...h.timers.values()][0]();
  assert.equal(h.last().error, null);
  assert.equal(livenessManagementView(h.last()).runtimeLabel, "Liveness disabled");
  h.manager.stop();
});

test("an uncertain enable response is checked before retrying the operation", async () => {
  let postCount = 0;
  const h = harness({
    liveness: async () => status(),
    enableLiveness: async () => {
      if (++postCount === 1) throw new ApiError({ code: "network_error", message: "Connection interrupted" });
      return status({ state: "downloading" });
    },
  });
  await h.manager.start();
  assert.equal(await h.manager.enable(), false);
  assert.equal(h.last().error.code, "network_error");
  assert.equal(livenessManagementView(h.last()).showAction, false);
  await h.manager.refresh();
  assert.equal(h.last().error, null);
  assert.equal(livenessManagementView(h.last()).showAction, true);
  assert.equal(await h.manager.enable(), true);
  assert.equal(postCount, 2);
  h.manager.stop();
});

test("returning to System during an enable request refreshes status after it completes", async () => {
  let current = status();
  let reads = 0;
  let resolvePost;
  const h = harness({
    liveness: async () => { reads += 1; return current; },
    enableLiveness: () => new Promise((resolve) => { resolvePost = resolve; }),
  });
  await h.manager.start();
  const submitting = h.manager.enable();
  h.manager.stop();
  await h.manager.start();
  current = status({ installed: true, configured_enabled: true, restart_required: true, state: "ready" });
  resolvePost(current);
  await submitting;
  assert.equal(reads, 2);
  assert.equal(h.last().submitting, false);
  assert.equal(h.last().status.state, "ready");
  assert.equal(livenessManagementView(h.last()).showAction, false);
});

test("a status request from a previous page visit cannot overwrite a newer response", async () => {
  let resolveFirst;
  let reads = 0;
  const h = harness({
    liveness: () => ++reads === 1 ? new Promise((resolve) => { resolveFirst = resolve; })
      : Promise.resolve(status({ enabled: true, configured_enabled: true, installed: true })),
  });
  const first = h.manager.start();
  h.manager.stop();
  await h.manager.start();
  resolveFirst(status());
  await first;
  assert.equal(h.last().status.enabled, true);
});

test("all liveness management states provide translated copy in the nine supported languages", () => {
  const messages = new Set();
  const variants = [
    {}, { status: status() }, { status: status({ installed: true }) },
    { status: status(), submitting: true }, { status: status({ state: "downloading" }) },
    { status: status({ enabled: true, installed: true, configured_enabled: true }) },
    { status: status({ installed: true, configured_enabled: true, restart_required: true }) },
    { status: status({ enabled: true, installed: true, configured_enabled: false, restart_required: true }) },
  ];
  for (const variant of variants) {
    const view = livenessManagementView(variant);
    for (const key of ["runtimeLabel", "installedLabel", "afterRestartLabel", "actionLabel", "notice"]) {
      if (view[key]) messages.add(view[key]);
    }
  }
  for (const { code } of LANGUAGES.filter(({ code }) => code !== "en")) {
    for (const message of messages) assert.ok(hasTranslation(message, code), `${code}: ${message}`);
  }
  assert.equal(t("Liveness disabled", {}, "zh"), "活体检测已禁用");
  assert.match(t("The model is installed and the startup configuration is saved. Restart Server manually to enable liveness.", {}, "zh"), /手动重启/);
});

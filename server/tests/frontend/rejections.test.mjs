import assert from "node:assert/strict";
import test from "node:test";
import { t } from "../../frontend/i18n.mjs";
import { renderRejectionList } from "../../frontend/rejections.mjs";
import { livenessResultText } from "../../frontend/liveness.mjs";
import { formatScore } from "../../frontend/core.mjs";

// Only the DOM operations used by this renderer; no browser globals required.
function element(tag, { className = "", text = "" } = {}) {
  return {
    tag, className, textContent: t(text, {}, "zh"), children: [], hidden: false,
    append(...nodes) { this.children.push(...nodes); },
    replaceChildren(...nodes) { this.children = nodes; },
  };
}

function render(target, items, files = [], livenessText = (item) => `liveness: ${JSON.stringify(item.liveness)}`) {
  renderRejectionList(target, items, files, {
    element,
    t: (message, values) => t(message, values, "zh"),
    livenessText,
  });
}

test("enrollment renders localized input advice as text beside the independent business reason", () => {
  const inputReason = "Insufficient image area around the face for liveness detection. Move the face toward the center, step back from the camera, or use a less tightly cropped image.";
  const unknownReason = '<img src=x onerror="alert(1)">';
  for (const businessReason of ["liveness_input_rejected", "low_quality"]) {
    for (const reason of [inputReason, unknownReason, undefined]) {
      const target = element("div");
      const item = Object.freeze({ reason: businessReason, liveness: Object.freeze({ status: "input_rejected", is_live: null, live_score: null, reason }) });
      render(target, [item], [], (entry) => livenessResultText(entry.liveness, { translate: (message) => t(message, {}, "zh"), formatScore }));
      const [primary, supplemental] = target.children[1].children[1].children;
      assert.equal(primary.textContent, businessReason);
      assert.equal(supplemental.textContent, t(reason ?? "Liveness input rejected", {}, "zh"));
      assert.deepEqual(supplemental.children, []);
      assert.equal(item.liveness.reason, reason);
      assert.equal(item.reason, businessReason);
    }
  }
});

test("an enrollment rejection keeps its actual reason alongside any liveness result", () => {
  for (const [reason, liveness] of [
    ["low_quality", { status: "ok", is_live: true, live_score: 0.98 }],
    ["extreme_pose", { status: "ok", is_live: false, live_score: 0.12 }],
    ["low_quality", { status: "input_rejected", is_live: null, live_score: null }],
    ["liveness_fake", { status: "ok", is_live: false, live_score: 0.12 }],
    ["liveness_input_rejected", { status: "input_rejected", is_live: null, live_score: null }],
  ]) {
    const target = element("div");
    render(target, [{ index: 0, reason, liveness }], [{ name: "portrait.jpg" }]);
    assert.equal(target.hidden, false);
    assert.equal(target.children[0].textContent, "1 张图片被拒绝");
    const row = target.children[1];
    assert.equal(row.children[0].textContent, "portrait.jpg");
    const [primary, supplemental] = row.children[1].children;
    assert.equal(primary.tag, "strong");
    assert.equal(primary.textContent, reason);
    assert.equal(supplemental.className, "rejection-liveness");
    assert.equal(supplemental.textContent, `liveness: ${JSON.stringify(liveness)}`);
  }
});

test("enrollment without liveness preserves filename and error-code fallbacks", () => {
  const target = element("div");
  render(target, [
    { filename: "first.jpg", file_name: "unused.jpg", reason: "face_too_small", code: "unused" },
    { file_name: "second.jpg", code: "future_reason" },
    {},
  ]);
  assert.equal(target.children[0].textContent, "3 张图片被拒绝");
  assert.deepEqual(target.children.slice(1).map((row) => {
    assert.equal(row.children[1].children.length, 1);
    return [row.children[0].textContent, row.children[1].children[0].textContent];
  }), [["first.jpg", "face_too_small"], ["second.jpg", "future_reason"], ["图片 3", "已拒绝"]]);
});

test("rendering a later response clears stale rejection and liveness information", () => {
  const target = element("div");
  render(target, [{ reason: "low_quality", liveness: { status: "ok", is_live: true, live_score: 1 } }]);
  render(target, [{ reason: "face_not_found" }]);
  assert.equal(target.children.length, 2);
  assert.equal(target.children[1].children[1].children.length, 1);
  for (const empty of [[], undefined, null]) {
    render(target, empty);
    assert.equal(target.hidden, true);
    assert.deepEqual(target.children, []);
  }
});

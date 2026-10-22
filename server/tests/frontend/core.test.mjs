import assert from "node:assert/strict";
import test from "node:test";

import { ApiClient, ApiError } from "../../frontend/api.mjs";
import { LANGUAGES, detectLocale, normalizeLocale, t } from "../../frontend/i18n.mjs";
import { renderMarkdown } from "../../frontend/markdown.mjs";
import {
  DEFAULT_THRESHOLD,
  SEARCH_PROFILES,
  applySearchProfileAvailability,
  authenticationEnabledFromHealth,
  bestPersonMatches,
  clamp,
  faceLandmarks,
  facePixels,
  filterPeople,
  formatCosine,
  formatScore,
  listItems,
  normalizeThreshold,
  parseExternalEmbeddings,
  parseMetadata,
  searchProfilesFromSystem,
} from "../../frontend/core.mjs";

function profileSelect(value = "fp32_v1") {
  return {
    value,
    disabled: false,
    options: SEARCH_PROFILES.map((profile) => ({ value: profile, disabled: false })),
  };
}

test("locales match the public InsightFace website and normalize browser variants", () => {
  assert.deepEqual(LANGUAGES.map(({ code }) => code), ["en", "zh", "ja", "de", "es", "fr", "ru", "pt", "ko"]);
  assert.equal(normalizeLocale("zh-CN"), "zh");
  assert.equal(normalizeLocale("pt_BR"), "pt");
  assert.equal(normalizeLocale("it-IT"), "en");
  assert.equal(detectLocale({ stored: "ja", languages: ["de-DE"] }), "ja");
  assert.equal(detectLocale({ languages: ["it-IT", "fr-FR"] }), "fr");
  assert.equal(t("Dashboard", {}, "zh"), "仪表盘");
  assert.equal(t("Loading documentation…", {}, "ko"), "문서 로드 중…");
  assert.equal(t("Open-source model license", {}, "zh"), "开源模型许可");
  assert.equal(t("Camera monitoring", {}, "zh"), "摄像头监控");
  assert.equal(t("No file selected", {}, "zh"), "未选择文件");
  assert.equal(t("JPEG, PNG, WebP, or BMP", {}, "zh"), "JPEG、PNG、WebP 或 BMP");
  assert.match(t("Commercial use requires a separate license.", {}, "ja"), /別途ライセンス/);
  assert.equal(t(t("Detection {score}", { score: "0.998" }, "de"), {}, "zh"), "检测 0.998");
  assert.equal(t(t("People in {collection}", { collection: "lfw" }, "ja"), {}, "ko"), "lfw의 인물");
});

test("bundled Markdown renderer supports portable guide links without executing HTML", () => {
  const rendered = renderMarkdown("# Guide\n\n![Dashboard](docs/images/customer/dashboard-en.jpg)\n\n[API](api.zh-CN.md)\n\n[Maintainer](maintainer-guide.md)\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\n```html\n<script>alert(1)</script>\n```\n\n[unsafe](javascript:alert(1))");
  assert.match(rendered, /<h1 id="guide">Guide<\/h1>/);
  assert.match(rendered, /<table>/);
  assert.match(rendered, /&lt;script&gt;alert\(1\)&lt;\/script&gt;/);
  assert.doesNotMatch(rendered, /<script>/);
  assert.match(rendered, /href="#"/);
  assert.match(rendered, /href="api\.zh-CN\.md"/);
  assert.match(rendered, /href="maintainer-guide\.md"/);
  assert.match(rendered, /<img src="\/guide-images\/customer\/dashboard-en\.jpg" alt="Dashboard" loading="lazy">/);
});

test("numeric helpers clamp thresholds and format scores", () => {
  assert.equal(DEFAULT_THRESHOLD, 0.4);
  assert.equal(clamp(2, 0, 1), 1);
  assert.equal(clamp(-1, 0, 1), 0);
  assert.equal(normalizeThreshold("", 0.3), 0.3);
  assert.equal(normalizeThreshold(1.5), 1);
  assert.equal(normalizeThreshold(-1.5), 0);
  assert.equal(formatScore(0.8234, 3), "0.823");
  assert.equal(formatScore(0.9984), "0.998");
  assert.equal(formatScore(undefined), "—");
  assert.equal(formatCosine(0.8234), "0.8234");
  assert.equal(formatCosine(-0.125), "-0.1250");
  assert.equal(formatCosine(undefined), "—");
});

test("authentication controls follow public health capability with a secure legacy fallback", () => {
  assert.equal(authenticationEnabledFromHealth({ auth_enabled: false }), false);
  assert.equal(authenticationEnabledFromHealth({ auth_enabled: true }), true);
  assert.equal(authenticationEnabledFromHealth({}), true);
  assert.equal(authenticationEnabledFromHealth(null), true);
});

test("search profile capabilities preserve server order-independent public profiles", () => {
  assert.equal(searchProfilesFromSystem({}), null);
  assert.deepEqual(searchProfilesFromSystem({ search: { profiles: [] } }), []);
  assert.deepEqual(
    searchProfilesFromSystem({ search: { profiles: ["int8_x1000_v1", "int8_x736_v1", "fp32_v1", "private_profile", "fp32_v1"] } }),
    ["fp32_v1", "int8_x736_v1", "int8_x1000_v1"],
  );
});

test("create Collection profile control disables unsupported options and selects first supported", () => {
  const select = profileSelect("fp16_v1");
  assert.equal(applySearchProfileAvailability(select, ["bf16_v1", "int8_x736_v1", "int8_x1000_v1"]), "bf16_v1");
  assert.deepEqual(
    select.options.map((option) => [option.value, option.disabled]),
    [
      ["fp32_v1", true],
      ["fp16_v1", true],
      ["bf16_v1", false],
      ["int8_x736_v1", false],
      ["int8_x1000_v1", false],
    ],
  );
  assert.equal(select.disabled, false);

  assert.equal(applySearchProfileAvailability(select, ["int8_x1000_v1"]), "int8_x1000_v1");
  assert.equal(applySearchProfileAvailability(select, []), "");
  assert.equal(select.disabled, true);
});

test("unknown search capabilities keep create profiles available for older servers", () => {
  const select = profileSelect("bf16_v1");
  select.options.forEach((option) => { option.disabled = true; });
  assert.equal(applySearchProfileAvailability(select, null), "bf16_v1");
  assert.equal(select.disabled, false);
  assert.ok(select.options.every((option) => option.disabled === false));
});

test("metadata accepts only JSON objects", () => {
  assert.deepEqual(parseMetadata(""), {});
  assert.deepEqual(parseMetadata('{"department":"sales"}'), { department: "sales" });
  assert.throws(() => parseMetadata("not-json"), /valid JSON/);
  assert.throws(() => parseMetadata("[]"), /JSON object/);
  assert.throws(() => parseMetadata("null"), /JSON object/);
});

test("trusted external embeddings require one finite vector per image", () => {
  assert.deepEqual(
    parseExternalEmbeddings("[[1,0],[0,1]]", 2, 2),
    [[1, 0], [0, 1]],
  );
  assert.throws(() => parseExternalEmbeddings("no", 1, 2), /valid JSON/);
  assert.throws(() => parseExternalEmbeddings("{}", 1, 2), /array of vectors/);
  assert.throws(() => parseExternalEmbeddings("[[1,2]]", 2, 2), /one external embedding for each image/);
  assert.throws(() => parseExternalEmbeddings("[[]]", 1, 2), /non-empty array/);
  assert.throws(() => parseExternalEmbeddings("[[1]]", 1, 2), /contain 2 values/);
  assert.throws(() => parseExternalEmbeddings('[[1,"2"]]', 1, 2), /non-finite or non-numeric/);
  assert.throws(() => parseExternalEmbeddings("[[0.5,0.5]]", 1, 2), /L2-normalized/);
});

test("listItems supports direct lists and conventional response keys", () => {
  assert.deepEqual(listItems([1, 2]), [1, 2]);
  assert.deepEqual(listItems({ collections: [{ id: "employees" }] }, ["collections"]), [{ id: "employees" }]);
  assert.deepEqual(listItems({ items: [3] }), [3]);
  assert.deepEqual(listItems(null), []);
});

test("facePixels handles pixel and normalized boxes", () => {
  assert.deepEqual(facePixels({ bbox: { pixels: { x: 10, y: 20, width: 30, height: 40 } } }, 200, 100), { x: 10, y: 20, width: 30, height: 40 });
  assert.deepEqual(facePixels({ bbox: { normalized: { left: 0.1, top: 0.2, width: 0.3, height: 0.4 } } }, 200, 100), { x: 20, y: 20, width: 60, height: 40 });
  assert.equal(facePixels({}, 200, 100), null);
});

test("faceLandmarks normalizes tuple and object points", () => {
  assert.deepEqual(faceLandmarks({ landmarks: [[1, 2], { x: 3, y: 4 }, null] }), [{ x: 1, y: 2 }, { x: 3, y: 4 }]);
  assert.deepEqual(faceLandmarks({}), []);
});

test("bestPersonMatches keeps the best face per person and sorts descending", () => {
  const samples = [
    { person: { id: "alice" }, similarity: 0.72, matched_face_id: "a1" },
    { person: { id: "bob" }, similarity: 0.81, matched_face_id: "b1" },
    { person: { id: "alice" }, similarity: 0.91, matched_face_id: "a2" },
    { person: { id: "carol" }, similarity: 0.4, matched_face_id: "c1" },
  ];
  assert.deepEqual(bestPersonMatches(samples, 0.68, 2).map((item) => item.matched_face_id), ["a2", "b1"]);
});

test("filterPeople searches ID, name, and external ID without changing input", () => {
  const people = [
    { id: "employee-001", name: "Alice", external_id: "HR-1001" },
    { id: "employee-002", name: "Bob", external_id: "HR-1002" },
  ];
  assert.deepEqual(filterPeople(people, "alice"), [people[0]]);
  assert.deepEqual(filterPeople(people, "1002"), [people[1]]);
  assert.notEqual(filterPeople(people, ""), people);
});

function fakeResponse(status, payload, headers = {}) {
  return {
    status,
    ok: status >= 200 && status < 300,
    headers: new Headers(headers),
    text: async () => payload === undefined ? "" : JSON.stringify(payload),
  };
}

test("ApiClient keeps authentication in memory and creates multipart detect requests", async () => {
  const calls = [];
  const client = new ApiClient("http://server.test", {
    fetchFn: async (url, options) => {
      calls.push({ url, options });
      return fakeResponse(200, { faces: [] }, { "x-request-id": "request-1" });
    },
  });
  client.setApiKey("secret-value");
  const image = new Blob(["image"], { type: "image/jpeg" });
  const result = await client.detect(image, { maxFaces: 4, collectionId: "employees" });
  assert.equal(result.request_id, "request-1");
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url.pathname, "/v1/detect");
  assert.equal(calls[0].options.method, "POST");
  assert.equal(calls[0].options.headers.get("Authorization"), "Bearer secret-value");
  assert.equal(calls[0].options.headers.get("Content-Type"), null, "browser must set the multipart boundary");
  assert.equal(calls[0].options.body.get("max_faces"), "4");
  assert.equal(calls[0].options.body.get("collection_id"), "employees");
  client.clearApiKey();
  assert.equal(client.hasApiKey, false);
});

test("ApiClient uses repeated images for multi-photo registration", async () => {
  let body;
  const client = new ApiClient("http://server.test", {
    fetchFn: async (_url, options) => { body = options.body; return fakeResponse(201, { person: { id: "alice" }, faces: [] }); },
  });
  const first = new Blob(["one"], { type: "image/jpeg" });
  const second = new Blob(["two"], { type: "image/png" });
  await client.createPerson("employees", { id: "alice", metadata: { team: "sales" }, images: [first, second], reviewMode: "strict" });
  assert.equal(body.getAll("images").length, 2);
  assert.equal(body.get("metadata"), '{"team":"sales"}');
  assert.equal(body.get("review_mode"), "strict");
  assert.equal(body.get("embedding_mode"), "server");
});

test("ApiClient sends trusted embeddings and Collection contract only when selected", async () => {
  const bodies = [];
  const client = new ApiClient("http://server.test", {
    fetchFn: async (_url, options) => { bodies.push(options.body); return fakeResponse(201, { faces: [], rejected_images: [] }); },
  });
  const images = [new Blob(["one"]), new Blob(["two"])];
  await client.createPerson("employees", {
    images,
    embeddingMode: "external_trusted",
    externalEmbeddings: [[0.1, -0.2], [0.3, 0.4]],
    embeddingContractId: "contract-v1",
  });
  await client.addFaces("employees", "alice", [images[0]], {
    embeddingMode: "external_trusted",
    externalEmbeddings: [[0.1, -0.2]],
    embeddingContractId: "contract-v1",
  });

  for (const body of bodies) {
    assert.equal(body.get("embedding_mode"), "external_trusted");
    assert.equal(body.get("embedding_contract_id"), "contract-v1");
  }
  assert.equal(bodies[0].get("external_embeddings"), "[[0.1,-0.2],[0.3,0.4]]");
});

test("ApiClient defaults additional FaceSample review to off", async () => {
  let body;
  const client = new ApiClient("http://server.test", {
    fetchFn: async (_url, options) => { body = options.body; return fakeResponse(201, { faces: [], rejected_images: [] }); },
  });
  await client.addFaces("employees", "alice", [new Blob(["face"], { type: "image/jpeg" })]);
  assert.equal(body.get("review_mode"), "off");
  assert.equal(body.get("embedding_mode"), "server");
});

test("ApiClient downloads a face crop with in-memory Bearer authentication", async () => {
  let request;
  const client = new ApiClient("http://server.test", {
    fetchFn: async (url, options) => {
      request = { url, options };
      return new Response(new Blob(["jpeg-crop"], { type: "image/jpeg" }), {
        status: 200,
        headers: { "content-type": "image/jpeg", "x-request-id": "crop-request" },
      });
    },
  });
  client.setApiKey("secret-value");

  const crop = await client.getFaceCrop("team a", "alice/b", "face/1");

  assert.equal(await crop.text(), "jpeg-crop");
  assert.equal(request.url.pathname, "/v1/collections/team%20a/persons/alice%2Fb/faces/face%2F1/image");
  assert.equal(request.options.headers.get("Authorization"), "Bearer secret-value");
  assert.equal(request.options.headers.get("Accept"), "image/jpeg");
});

test("ApiClient creates, polls, updates, and deletes persistent Monitors", async () => {
  const calls = [];
  const responses = [
    fakeResponse(201, { monitor: { id: "front-gate" } }),
    fakeResponse(200, { state: { monitor_id: "front-gate", status: "running" } }),
    fakeResponse(200, { events: [], next_cursor: "cursor-1" }),
    fakeResponse(200, { monitor: { id: "front-gate", preview_enabled: true } }),
    fakeResponse(204),
  ];
  const client = new ApiClient("http://server.test", {
    fetchFn: async (url, options) => {
      calls.push({ url, options });
      return responses.shift();
    },
  });

  await client.createMonitor({
    id: "front-gate",
    name: "Front gate",
    url: "rtsp://camera.test/live",
    collectionId: "employees",
    inferenceFps: 2.5,
    matchThreshold: 0.45,
    previewEnabled: false,
  });
  await client.monitorState("front-gate");
  await client.monitorEvents("front-gate", { cursor: "old", limit: 25 });
  await client.updateMonitor("front-gate", { preview_enabled: true });
  await client.deleteMonitor("front-gate");

  assert.equal(calls[0].url.pathname, "/v1/monitors");
  assert.deepEqual(JSON.parse(calls[0].options.body), {
    id: "front-gate",
    name: "Front gate",
    description: "",
    enabled: true,
    source: { type: "rtsp", url: "rtsp://camera.test/live" },
    collection_id: "employees",
    inference_fps: 2.5,
    match_threshold: 0.45,
    event_buffer_size: 1000,
    event_policy: {
      confirm_frames: 3,
      absence_timeout_seconds: 3,
      cooldown_seconds: 10,
      emit_unknown: true,
    },
    preview_enabled: false,
  });
  assert.equal(calls[1].url.pathname, "/v1/monitors/front-gate/state");
  assert.equal(calls[1].options.method, "GET");
  assert.equal(calls[2].url.searchParams.get("cursor"), "old");
  assert.equal(calls[2].url.searchParams.get("limit"), "25");
  assert.equal(calls[3].options.method, "PATCH");
  assert.equal(calls[4].options.method, "DELETE");
});

test("ApiClient preserves collection-default threshold when creating a Monitor", async () => {
  let request;
  const client = new ApiClient("http://server.test", {
    fetchFn: async (url, options) => {
      request = { url: new URL(url), options };
      return fakeResponse(201, { monitor: { id: "front-gate" } });
    },
  });

  await client.createMonitor({
    id: "front-gate",
    name: "Front gate",
    url: "rtsp://camera.test/live",
    collectionId: "employees",
    matchThreshold: null,
  });

  assert.equal(JSON.parse(request.options.body).match_threshold, null);
});

test("ApiClient converts standard API errors and preserves request ID", async () => {
  const client = new ApiClient("http://server.test", {
    fetchFn: async () => fakeResponse(422, {
      error: { code: "face_not_found", message: "No usable face was detected.", details: { field: "image" } },
      request_id: "request-body-id",
    }),
  });
  await assert.rejects(
    client.health(),
    (error) => error instanceof ApiError && error.status === 422 && error.code === "face_not_found" && error.requestId === "request-body-id",
  );
});

test("ApiClient handles empty 204 responses", async () => {
  const client = new ApiClient("http://server.test", { fetchFn: async () => fakeResponse(204) });
  assert.equal(await client.deletePerson("employees", "alice"), null);
});

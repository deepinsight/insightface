import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { hasTranslation, LANGUAGES } from "../../frontend/i18n.mjs";

const frontend = new URL("../../frontend/", import.meta.url);
const server = new URL("../../", import.meta.url);

async function source(name) {
  return readFile(new URL(name, frontend), "utf8");
}

function decodeEntities(value) {
  return value
    .replaceAll("&amp;", "&")
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">")
    .replaceAll("&quot;", '"')
    .replaceAll("&#39;", "'")
    .replaceAll("&nbsp;", " ");
}

function staticUiStrings(html) {
  const visible = html
    .replace(/<!--.*?-->/gs, "")
    .replace(/<(script|style|code|pre|textarea)\b[^>]*>[\s\S]*?<\/\1>/gi, "<ignored></ignored>");
  const messages = new Set();
  for (const match of visible.matchAll(/\b(?:aria-label|placeholder|title)="([^"]+)"/g)) {
    messages.add(decodeEntities(match[1]).trim());
  }
  for (const part of visible.split(/<[^>]+>/g)) {
    const message = decodeEntities(part).replace(/\s+/g, " ").trim();
    if (message) messages.add(message);
  }
  return [...messages];
}

function translatedLiteralStrings(script) {
  const messages = new Set();
  const patterns = [
    /\bt\(\s*("(?:[^"\\]|\\.)*")/g,
    /\btext:\s*("(?:[^"\\]|\\.)*")/g,
    /\bstatusRow\(\s*("(?:[^"\\]|\\.)*")/g,
    /\btoast\(\s*("(?:[^"\\]|\\.)*")/g,
  ];
  for (const pattern of patterns) {
    for (const match of script.matchAll(pattern)) messages.add(JSON.parse(match[1]));
  }
  return messages;
}

test("main page exposes every requested product surface", async () => {
  const html = await source("index.html");
  for (const page of ["dashboard", "collections", "people", "detect", "compare", "search", "video", "system", "help"]) {
    assert.match(html, new RegExp(`data-page="${page}"`));
  }
  const primaryNavigation = html.match(/<nav class="nav-list">([\s\S]*?)<\/nav>/)?.[1] || "";
  assert.doesNotMatch(primaryNavigation, /href="\/docs"/);
  assert.match(html, /href="\/docs"/);
  assert.match(html, /DEVELOPER TOOLS/);
  assert.match(html, /Developer schema/);
  assert.match(html, /Content-Security-Policy/);
});

test("console and API reference expose the official nine-language selector", async () => {
  const index = await source("index.html");
  const docs = await source("openapi.html");
  for (const code of ["en", "zh", "ja", "de", "es", "fr", "ru", "pt", "ko"]) {
    assert.match(index, new RegExp(`<option value="${code}">`));
    assert.match(docs, new RegExp(`<option value="${code}">`));
  }
  assert.match(await source("app.mjs"), /guide-content\/\$\{documentLocale\}/);
  assert.match(await source("openapi.js"), /initializeI18n/);
});

test("repository README and bundled help sources exist for every supported locale", async () => {
  const suffixes = ["", ".zh-CN", ".ja", ".de", ".es", ".fr", ".ru", ".pt", ".ko"];
  const retiredDocumentation = [
    "deployment.md",
    "sdk.md",
    "external-embeddings.md",
    "models.md",
    "gpu-support.md",
    "security.md",
    "search.md",
    "development.md",
    "container-versioning.md",
  ];
  const requiredReadmeMarkers = [
    "Rekognition",
    "0.3.1-cpu",
    "0.3.1-cuda12",
    "dashboard-en.jpg",
    "collections-en.jpg",
    "monitoring-en.jpg",
    "models install buffalo_l",
    "--accept-license",
    "buffalo_m",
    "buffalo_s",
    "buffalo_sc",
    "antelopev2",
    "raccoon_s",
    "raccoon_l",
    "MODEL.LICENSE",
    "LICENSING.md",
    "external_trusted",
    "review_mode",
    "center_largest",
    "RTX 5090",
    "INT8",
    "50M+",
    "Top-5",
    "/v1/embeddings",
    "preview.mjpeg",
    "collection_model_mismatch",
    "x-request-id",
    "server/.models",
    "make -C server build-cpu",
    "make -C server build-cuda12",
    "--pull never",
    "INSIGHTFACE_AUTH_ENABLED",
    "0.4",
    "112x112",
    "bounding-box",
  ];
  for (const suffix of suffixes) {
    const readme = await readFile(new URL(`README${suffix}.md`, server), "utf8");
    const guide = await readFile(new URL(`docs/user-guide${suffix}.md`, server), "utf8");
    const api = await readFile(new URL(`docs/api${suffix}.md`, server), "utf8");
    assert.match(readme, /^# InsightFace Server/m);
    assert.match(guide, /^# .*InsightFace Server|^# InsightFace Server/m);
    assert.match(api, /^# .*InsightFace Server|^# InsightFace Server/m);
    assert.ok(readme.length > 1000, `README${suffix}.md is unexpectedly short`);
    assert.ok(
      readme.split(/\r?\n/).length <= 320,
      `README${suffix}.md must remain an overview rather than duplicate detailed guides`,
    );
    assert.equal(
      (readme.match(/^## /gm) || []).length,
      9,
      `README${suffix}.md must use the shared nine-section information architecture`,
    );
    assert.ok(guide.length > 3500, `user-guide${suffix}.md is unexpectedly short`);
    assert.ok(api.length > 7000, `api${suffix}.md is unexpectedly short`);
    assert.equal((api.match(/^### `(?:GET|POST|PATCH|DELETE) \/v1\//gm) || []).length, 31);
    for (const marker of requiredReadmeMarkers) {
      assert.ok(
        readme.includes(marker),
        `README${suffix}.md does not cover required marker: ${marker}`,
      );
    }
    for (const value of [
      /15[.,]8M/,
      /30[.,]7M/,
      /58[.,]9M/,
      /3[.,]84 ms/,
      /260[.,]81/,
    ]) {
      assert.match(
        readme,
        value,
        `README${suffix}.md is missing a published RTX 5090 benchmark value`,
      );
    }
    assert.ok(readme.includes(`docs/user-guide${suffix}.md`));
    assert.ok(readme.includes(`docs/api${suffix}.md`));
    assert.ok(readme.includes("docs/maintainer-guide.md"));
    assert.ok(!readme.includes("LICENSE-NOTICE.md"));
    assert.ok(!readme.includes("[LICENSE](LICENSE)"));
    for (const retired of retiredDocumentation) {
      assert.ok(
        !readme.includes(`docs/${retired}`),
        `README${suffix}.md still links retired fragmented documentation: ${retired}`,
      );
    }
    for (const marker of [
      "0.4",
      "center_largest",
      "external_trusted",
      "bounding-box crop",
      "INSIGHTFACE_AUTH_ENABLED",
      "models verify",
      "MODEL.LICENSE",
      "server/config/server.toml",
      "max_concurrency",
      "fp32_v1",
      "int8_x736_v1",
      "CUDA Runtime 12.9.1",
      "docker compose down -v",
    ]) {
      assert.ok(
        guide.includes(marker),
        `user-guide${suffix}.md is missing the shared behavior marker: ${marker}`,
      );
    }
    assert.ok(
      readme.indexOf("https://www.insightface.ai") <
        readme.indexOf("dashboard-en.jpg"),
      `README${suffix}.md must show the commercial model-license notice near the top`,
    );
  }
  const maintainer = await readFile(new URL("docs/maintainer-guide.md", server), "utf8");
  const licensing = await readFile(new URL("LICENSING.md", server), "utf8");
  assert.match(licensing, /source code and its Python SDK are licensed under MIT/i);
  assert.match(licensing, /non-commercial\s+academic research only/i);
  assert.match(licensing, /https:\/\/www\.insightface\.ai/);
  assert.match(licensing, /MODEL\.LICENSE/);
  assert.doesNotMatch(licensing, /Permission is hereby granted/);
  assert.ok(maintainer.length > 12000);
  for (const marker of [
    "Maintainer reference — English only",
    "durable barrier",
    "int8_x736_v1",
    "device-resident",
    "Public API documentation change gate",
    "Container versioning and release",
  ]) {
    assert.ok(maintainer.includes(marker), `maintainer-guide.md is missing: ${marker}`);
  }
  for (const retired of retiredDocumentation) {
    await assert.rejects(readFile(new URL(`docs/${retired}`, server), "utf8"));
  }
  const index = await source("index.html");
  const app = await source("app.mjs");
  const markdown = await source("markdown.mjs");
  assert.match(index, /id="documentation-switcher"/);
  assert.match(index, /id="documentation-toc"/);
  assert.match(index, /id="documentation-filter"/);
  assert.doesNotMatch(index, /data-document=/);
  assert.doesNotMatch(index, />README</);
  assert.match(app, /\{ name: "user-guide", label: "User guide", localized: true \}/);
  assert.match(app, /\{ name: "api", label: "API guide", localized: true \}/);
  assert.match(app, /\{ name: "maintainer", label: "Maintainer guide", localized: false \}/);
  assert.match(app, /documentDefinition\.localized \? locale\(\) : "en"/);
  assert.match(app, /documentationTarget/);
  assert.match(markdown, /maintainer-guide/);
  assert.doesNotMatch(app, /name: "readme"/);
  assert.match(app, /renderDocumentationToc/);
});

test("frontend assets use the server's /assets mount and no external CDN", async () => {
  const index = await source("index.html");
  const docs = await source("openapi.html");
  assert.doesNotMatch(`${index}\n${docs}`, /\/static\//);
  assert.match(index, /href="\/assets\/styles\.css\?v=0\.3\.1-r1"/);
  assert.match(index, /src="\/assets\/app\.mjs\?v=0\.3\.1-r1"/);
  assert.match(docs, /href="\/assets\/openapi\.css"/);
  assert.match(docs, /src="\/assets\/openapi\.js\?v=0\.3\.1-r1"/);
  const externalReferences = [...`${index}\n${docs}`.matchAll(/(?:src|href)="(https?:\/\/[^\"]+)"/g)].map((match) => match[1]);
  assert.deepEqual(externalReferences, ["https://www.insightface.ai"]);
});

test("System page renders the runtime model license and keeps the commercial link", async () => {
  const html = await source("index.html");
  const app = await source("app.mjs");
  assert.match(html, /Commercial use requires a separate license/);
  assert.match(html, /href="https:\/\/www\.insightface\.ai"/);
  assert.match(html, /id="system-license-notice"[^>]*hidden/);
  assert.match(html, /id="system-license-summary"/);
  assert.match(app, /modelsPayload\?\.license/);
  assert.match(app, /licenseNotice\.hidden = false/);
  assert.match(app, /license\.commercial_use_permitted === true/);
  assert.doesNotMatch(app, /license\.status/);
});

test("console and API reference use the InsightFace.ai brand system", async () => {
  const index = await source("index.html");
  const docs = await source("openapi.html");
  const styles = `${await source("styles.css")}\n${await source("openapi.css")}`;
  const mark = await source("insightface-mark.svg");

  assert.match(index, /InsightFace Server dashboard/);
  assert.match(`${index}\n${docs}`, /\/assets\/insightface-mark\.svg/);
  assert.match(mark, /fill="#6366f1"/);
  assert.match(mark, /<circle cx="12" cy="8" r="4"/);
  for (const token of ["#09090b", "#111113", "#6366f1", "#818cf8", "#22d3ee"]) {
    assert.match(styles, new RegExp(token));
  }
});

test("API key and images are not persisted by browser code", async () => {
  const scripts = `${await source("app.mjs")}\n${await source("api.mjs")}\n${await source("rejections.mjs")}\n${await source("liveness.mjs")}`;
  assert.doesNotMatch(scripts, /localStorage|sessionStorage|indexedDB/);
  assert.doesNotMatch(scripts, /console\.(?:log|info|debug)/);
});

test("API key UI is shown only when the public health response enables authentication", async () => {
  const html = await source("index.html");
  const app = await source("app.mjs");
  assert.match(html, /class="sidebar-footer" data-auth-ui hidden/);
  assert.match(html, /id="auth-state" data-auth-ui hidden/);
  assert.match(html, /class="metric-card" data-auth-ui hidden><span>API key in UI<\/span><strong class="metric-text" id="system-auth"/);
  assert.match(html, /id="api-key-dialog" data-auth-ui hidden/);
  assert.match(app, /authenticationEnabledFromHealth/);
  assert.match(app, /\$\$\('\[data-auth-ui\]'\)/);
  assert.match(app, /node\.hidden = !enabled/);
});

test("System page shows startup-only detector configuration", async () => {
  const app = await source("app.mjs");
  const api = await source("api.mjs");
  const html = await source("index.html");
  assert.match(app, /Detector input sizes/);
  assert.match(app, /Detection threshold/);
  assert.match(app, /Single-face selection/);
  assert.match(`${app}\n${html}`, /center_largest/);
  assert.match(app, /Startup config/);
  assert.doesNotMatch(api, /\/v1\/system\/config/);
  assert.doesNotMatch(app, /updateSystemConfig|getSystemConfig/);
});

test("enrollment forms expose review modes with off selected by default", async () => {
  const html = await source("index.html");
  const app = await source("app.mjs");
  for (const value of ["off", "standard", "strict"]) {
    assert.match(`${html}\n${app}`, new RegExp(`<option value="${value}"`));
  }
  assert.match(html, /<select name="review_mode"><option value="off" selected>/);
  assert.match(app, /<select name="review_mode"><option value="off" selected>/);
  assert.match(app, /reviewMode: form\.elements\.review_mode\.value/);
});

test("enrollment UI makes trusted external features explicit and shows the contract", async () => {
  const html = await source("index.html");
  const app = await source("app.mjs");
  const api = await source("api.mjs");
  assert.match(html, /<select name="embedding_mode"><option value="server" selected>/);
  assert.match(`${html}\n${app}`, /value="external_trusted"/);
  assert.match(`${html}\n${app}`, /name="external_embeddings"/);
  assert.match(app, /collection\?\.embedding_contract_id/);
  assert.match(app, /parseExternalEmbeddings/);
  assert.match(api, /form\.append\("embedding_mode", embeddingMode\)/);
  assert.match(api, /form\.append\("embedding_contract_id", embeddingContractId/);
});

test("camera monitoring uses persistent Monitors, memory events, and raw lazy preview", async () => {
  const html = await source("index.html");
  const app = await source("app.mjs");
  const api = await source("api.mjs");
  assert.match(html, /id="rtsp-url" type="url"/);
  assert.match(html, /id="rtsp-preview"/);
  assert.match(html, /legend-swatch recognized/);
  assert.match(html, /legend-swatch not-enrolled/);
  assert.doesNotMatch(html, /type="file" accept="video\//);
  assert.doesNotMatch(app, /mediaDevices\.getUserMedia/);
  assert.doesNotMatch(app, /new VideoFrameController/);
  assert.match(html, /name="preview_enabled" type="checkbox"/);
  assert.doesNotMatch(html, /name="preview_enabled"[^>]*checked/);
  assert.match(html, /id="rtsp-overlay"/);
  assert.match(html, /id="monitor-edit"/);
  assert.match(app, /client\.createMonitor/);
  assert.match(app, /client\.updateMonitor\(editingMonitorId, common\)/);
  assert.match(app, /Leave blank to keep the current RTSP URL/);
  assert.match(app, /client\.monitorState/);
  assert.match(app, /client\.monitorEvents/);
  assert.match(app, /client\.monitorPreview/);
  assert.match(app, /window\.setTimeout\(\(\) => void pollMonitor/);
  assert.match(api, /\/v1\/monitors/);
  assert.doesNotMatch(api, /\/v1\/streams\/rtsp/);
  assert.match(app, /URL\.createObjectURL\(new Blob\(\[jpeg\]/);
  assert.match(app, /drawMonitorOverlay/);
});

test("image uploads advertise JPEG, PNG, WebP, and BMP and search fields share a baseline", async () => {
  const html = await source("index.html");
  const app = await source("app.mjs");
  const styles = await source("styles.css");
  assert.equal([...html.matchAll(/accept="image\/jpeg,image\/png,image\/webp,image\/bmp,image\/x-ms-bmp,\.bmp"/g)].length, 5);
  assert.match(app, /accept="image\/jpeg,image\/png,image\/webp,image\/bmp,image\/x-ms-bmp,\.bmp"/);
  assert.doesNotMatch(`${html}\n${app}`, /accept="image\/jpeg,image\/png"/);
  assert.match(html, /class="form-grid two search-parameters"/);
  assert.match(styles, /\.search-parameters \{ align-items: end; \}/);
});

test("form-grid controls stay top-aligned when a sibling has helper text", async () => {
  const html = await source("index.html");
  const styles = await source("styles.css");
  assert.match(styles, /\.form-grid > label \{[\s\S]*?align-self: start;/);
  assert.match(styles, /grid-template-rows: minmax\(2\.65em, auto\) auto auto/);
  assert.match(styles, /\.field-caption \{[^}]*align-self: end/s);
  assert.ok([...html.matchAll(/class="field-caption"/g)].length >= 20);
  assert.match(styles, /\.compare-action-bar \{[^}]*grid-template-columns:[^;}]*minmax\(190px,[^;}]*minmax\(250px/s);
});

test("all static console and API-reference copy is translated in every supported locale", async () => {
  const messages = staticUiStrings(`${await source("index.html")}\n${await source("openapi.html")}`);
  for (const language of LANGUAGES.filter(({ code }) => code !== "en")) {
    const missing = messages.filter((message) => !hasTranslation(message, language.code));
    assert.deepEqual(missing, [], `${language.code} is missing: ${missing.join(" | ")}`);
  }
});

test("dynamic result, status, and action copy is translated in every supported locale", async () => {
  const app = await source("app.mjs");
  const messages = translatedLiteralStrings(`${app}\n${await source("openapi.js")}\n${await source("rejections.mjs")}\n${await source("liveness.mjs")}`);
  for (const template of app.matchAll(/innerHTML\s*=\s*`([\s\S]*?)`/g)) {
    for (const message of staticUiStrings(template[1])) messages.add(message);
  }
  for (const message of [
    "Detection failed", "Detecting…", "Comparison failed", "Comparing…", "Search failed", "Searching…",
    "Could not load collections", "Could not load people", "Could not load face samples", "Could not load system diagnostics", "Refreshing…",
    "Collection could not be created", "Creating…", "Collection could not be updated", "Saving…", "Person could not be registered", "Registering…",
    "Face samples could not be added", "Uploading…", "Face sample could not be deleted", "Video frame failed", "Video could not start",
  ]) messages.add(message);
  for (const language of LANGUAGES.filter(({ code }) => code !== "en")) {
    const missing = [...messages].filter((message) => !hasTranslation(message, language.code));
    assert.deepEqual(missing, [], `${language.code} is missing dynamic copy: ${missing.join(" | ")}`);
  }
});

test("self-hosted schema explorer reads the local OpenAPI document", async () => {
  const html = await source("openapi.html");
  const script = await source("openapi.js");
  assert.match(html, /OpenAPI Schema/);
  assert.match(html, /DEVELOPER TOOL · OPENAPI 3/);
  assert.match(script, /fetch\("\/openapi\.json"/);
  assert.match(script, /components\?\.schemas/);
});

test("Collection creation follows runtime search profiles without changing the pinned edit field", async () => {
  const html = await source("index.html");
  const app = await source("app.mjs");
  assert.match(html, /<select name="search_profile" required>/);
  for (const profile of ["fp32_v1", "fp16_v1", "bf16_v1", "int8_x736_v1", "int8_x1000_v1"]) {
    assert.match(html, new RegExp(`<option value="${profile}">`));
  }
  assert.match(html, /<input name="search_profile" readonly>/);
  assert.match(app, /searchProfilesFromSystem\(state\.system\)/);
  assert.match(app, /state\.system = result\.system;[\s\S]*?updateCreateCollectionProfiles\(\);[\s\S]*?renderSystem\(\);/);
});

test("Collection crop storage is opt-in and previews use revocable authenticated blobs", async () => {
  const html = await source("index.html");
  const app = await source("app.mjs");
  const api = await source("api.mjs");
  assert.match(html, /name="save_face_crops" type="checkbox"/);
  assert.doesNotMatch(html, /name="save_face_crops"[^>]*checked/);
  assert.match(app, /save_face_crops: form\.elements\.save_face_crops\.checked/);
  assert.match(app, /state\.system\?\.safe_config\?\.save_face_crops/);
  assert.match(app, /face\.has_crop/);
  assert.match(api, /getFaceCrop\(/);
  assert.match(api, /Authorization/);
  assert.match(app, /URL\.createObjectURL\(blob\)/);
  assert.match(app, /URL\.revokeObjectURL\(url\)/);
});

test("model surfaces identify packages by name without a separate model version", async () => {
  const app = await source("app.mjs");
  const html = await source("index.html");
  assert.doesNotMatch(`${app}\n${html}`, /model_version|metric-model-version/);
  assert.match(html, /id="metric-model"/);
  assert.match(app, /collection\.model_id/);
});

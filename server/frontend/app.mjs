import { ApiClient, ApiError } from "./api.mjs?v=0.3.1-r1";
import { initializeI18n, locale, setLocale, t, translateTree } from "./i18n.mjs?v=0.3.1-r1";
import { documentationLink, renderMarkdown } from "./markdown.mjs?v=0.3.1-r1";
import { renderRejectionList } from "./rejections.mjs?v=0.3.1-r1";
import { createLivenessManager, livenessErrorText, livenessManagementView, livenessMessage, livenessResultText, livenessRuntimeLabel } from "./liveness.mjs?v=0.3.1-r1";
import {
  applySearchProfileAvailability,
  authenticationEnabledFromHealth,
  faceLandmarks,
  facePixels,
  filterPeople,
  formatCosine,
  formatDuration,
  formatScore,
  listItems,
  parseExternalEmbeddings,
  parseMetadata,
  searchProfilesFromSystem,
} from "./core.mjs?v=0.3.1-r1";

const client = new ApiClient(window.location.origin);
const state = {
  route: "dashboard",
  collections: [],
  people: [],
  selectedCollection: "",
  selectedPerson: null,
  system: null,
  models: [],
  modelLicense: null,
  livenessManagement: null,
  errors: [],
  serverErrors: [],
  monitors: [],
  selectedMonitorId: "",
  editingMonitorId: "",
  monitorPollTimer: null,
  monitorEventCursor: "",
  monitorEvents: [],
  previewAbortController: null,
  previewObjectUrl: "",
  videoStatus: null,
  cropObjectUrls: new Set(),
  authEnabled: null,
  helpDocument: "user-guide",
  helpPendingAnchor: null,
  helpLoadRevision: 0,
};

const helpDocuments = [
  { name: "user-guide", label: "User guide", localized: true },
  { name: "api", label: "API guide", localized: true },
  { name: "maintainer", label: "Maintainer guide", localized: false },
];

const pageTitles = {
  dashboard: "Dashboard",
  collections: "Collections",
  people: "People",
  detect: "Detect",
  compare: "Compare",
  search: "Search",
  video: "Camera monitoring",
  system: "System",
  help: "Help",
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const livenessManager = createLivenessManager({
  client,
  onChange(snapshot) {
    state.livenessManagement = snapshot;
    renderLivenessManagement();
  },
});

function element(tag, { className = "", text = "", title = "" } = {}) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== "") node.textContent = t(text);
  if (title) node.title = t(title);
  return node;
}

function firstValue(...values) {
  return values.find((value) => value !== undefined && value !== null && value !== "");
}

function valueAt(object, path) {
  return path.split(".").reduce((value, key) => value?.[key], object);
}

function readAny(object, ...paths) {
  return firstValue(...paths.map((path) => valueAt(object, path)));
}

function unwrap(payload, key) {
  return payload?.[key] ?? payload;
}

function selectedCollection() {
  return state.collections.find((collection) => collection.id === state.selectedCollection);
}

function configureEmbeddingFields(form) {
  const mode = form?.elements?.embedding_mode?.value ?? "server";
  const fields = $(".external-embedding-fields", form);
  if (!fields) return;
  fields.hidden = mode !== "external_trusted";
  const collection = selectedCollection();
  const contract = collection?.embedding_contract_id || "Unavailable";
  $$(".embedding-contract", form).forEach((target) => { target.textContent = contract; });
}

function enrollmentEmbeddingInput(form, images) {
  const embeddingMode = form.elements.embedding_mode?.value ?? "server";
  if (embeddingMode !== "external_trusted") {
    return { embeddingMode: "server" };
  }
  const collection = selectedCollection();
  const embeddingContractId = collection?.embedding_contract_id;
  if (!embeddingContractId) {
    throw new Error("The selected Collection does not report an embedding contract. Refresh it or use server extraction.");
  }
  return {
    embeddingMode,
    embeddingContractId,
    externalEmbeddings: parseExternalEmbeddings(
      form.elements.external_embeddings.value,
      images.length,
      collection.embedding_dimension,
    ),
  };
}

function setText(selector, text) {
  const target = $(selector);
  if (target) target.textContent = text ?? "—";
}

function releaseCropObjectUrl(url) {
  if (!url || !state.cropObjectUrls.delete(url)) return;
  URL.revokeObjectURL(url);
}

function releaseAllCropObjectUrls() {
  for (const url of state.cropObjectUrls) URL.revokeObjectURL(url);
  state.cropObjectUrls.clear();
}

function setBusy(control, busy, busyText = "Working…") {
  if (!control) return;
  if (busy) {
    control.dataset.originalText = control.textContent;
    control.textContent = t(busyText);
    control.disabled = true;
  } else {
    control.textContent = control.dataset.originalText || control.textContent;
    control.disabled = false;
    delete control.dataset.originalText;
  }
}

function toast(title, message = "", kind = "success") {
  const region = $("#toast-region");
  const item = element("div", { className: `toast ${kind === "error" ? "error" : ""}` });
  item.append(element("strong", { text: title }));
  if (message) item.append(element("span", { text: message }));
  region.append(item);
  window.setTimeout(() => item.remove(), 5000);
}

function describeError(error) {
  if (error instanceof ApiError) return livenessErrorText(error, { translate: t, formatScore });
  return t(error?.message || "Unexpected error.");
}

function handleError(error, context = "Request failed") {
  const message = describeError(error);
  const requestId = error?.requestId ? ` ${t("Request: {id}", { id: error.requestId })}` : "";
  state.errors.unshift({ context, message, requestId, at: new Date() });
  state.errors = state.errors.slice(0, 8);
  renderRecentErrors();
  toast(context, `${message}${requestId}`, "error");
}

async function guarded(control, operation, context, busyText) {
  setBusy(control, true, busyText);
  try {
    return await operation();
  } catch (error) {
    handleError(error, context);
    return undefined;
  } finally {
    setBusy(control, false);
  }
}

function renderRecentErrors() {
  const container = $("#recent-errors");
  container.replaceChildren();
  const serverErrors = (state.serverErrors ?? []).map((item) => ({
    context: item.code || item.error_code || "Server error",
    message: item.message || item.detail || item.path || "An error was recorded by the server.",
    requestId: item.request_id ? ` ${t("Request: {id}", { id: item.request_id })}` : "",
    at: item.timestamp ? new Date(item.timestamp) : null,
  }));
  const errors = [...state.errors, ...serverErrors].slice(0, 8);
  if (!errors.length) {
    container.className = "empty-state compact";
    container.append(element("strong", { text: "No recent errors" }), element("span", { text: "API failures from this browser tab will appear here." }));
    return;
  }
  container.className = "";
  for (const error of errors) {
    const entry = element("div", { className: "error-entry" });
    entry.append(
      element("strong", { text: error.context }),
      element("span", { text: `${error.message}${error.requestId}` }),
      element("time", { text: error.at && !Number.isNaN(error.at.getTime()) ? error.at.toLocaleTimeString() : t("Server") }),
    );
    container.append(entry);
  }
}

async function refreshHealth() {
  const chip = $("#sidebar-health");
  try {
    const health = await client.health();
    applyAuthenticationMode(health);
    const status = String(firstValue(health?.status, health?.state, "healthy"));
    chip.className = `server-chip ${/ok|healthy|ready/i.test(status) ? "healthy" : "error"}`;
    chip.lastElementChild.textContent = t(status);
    return health;
  } catch (error) {
    chip.className = "server-chip error";
    chip.lastElementChild.textContent = t("Unavailable");
    throw error;
  }
}

function applyAuthenticationMode(health) {
  const enabled = authenticationEnabledFromHealth(health);
  state.authEnabled = enabled;
  if (!enabled) {
    client.clearApiKey();
    const dialog = $("#api-key-dialog");
    if (dialog?.open) dialog.close();
  }
  $$('[data-auth-ui]').forEach((node) => { node.hidden = !enabled; });
  updateAuthState();
}

function navigate(route) {
  if (!pageTitles[route]) return;
  if (state.route === "video" && route !== "video") stopMonitorView();
  if (state.route === "system" && route !== "system") livenessManager.stop();
  state.route = route;
  $$("[data-page]").forEach((page) => { page.hidden = page.dataset.page !== route; });
  $$(".nav-item[data-route]").forEach((item) => {
    const selected = item.dataset.route === route;
    item.classList.toggle("active", selected);
    if (selected) item.setAttribute("aria-current", "page");
    else item.removeAttribute("aria-current");
  });
  setText("#page-title", t(pageTitles[route]));
  document.body.classList.remove("menu-open");
  $("#menu-toggle").setAttribute("aria-expanded", "false");
  document.title = `${t(pageTitles[route])} · InsightFace Server`;
  window.history.replaceState(null, "", `#${route}`);
  void refreshRoute(route);
}

async function refreshRoute(route = state.route) {
  if (route === "dashboard") await loadDashboard();
  if (route === "collections") await loadCollections();
  if (route === "people") {
    await loadCollections();
    if (state.selectedCollection) await loadPeople();
  }
  if (["detect", "compare", "search", "video"].includes(route)) await loadCollections();
  if (route === "video") await loadMonitors();
  if (route === "system") await loadSystem();
  if (route === "help") await loadDocumentation();
}

function documentationTarget(href) {
  return documentationLink(href, state.helpDocument, locale());
}

function scrollDocumentation(chapter, anchor) {
  const heading = [...chapter.querySelectorAll("[data-anchor]")]
    .find((node) => node.dataset.anchor === anchor);
  (heading ?? chapter).scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderHelpDocumentSwitcher() {
  const switcher = $("#documentation-switcher");
  switcher.replaceChildren();
  for (const documentDefinition of helpDocuments) {
    const button = element("button", { className: "documentation-switch" });
    button.type = "button";
    button.dataset.helpDocument = documentDefinition.name;
    button.setAttribute("aria-pressed", String(state.helpDocument === documentDefinition.name));
    button.classList.toggle("active", state.helpDocument === documentDefinition.name);
    button.append(element("span", { text: documentDefinition.label }));
    if (!documentDefinition.localized) {
      button.append(element("small", { text: "English only" }));
    }
    button.addEventListener("click", () => {
      if (state.helpDocument === documentDefinition.name) {
        $("#markdown-document")?.scrollIntoView({ behavior: "smooth", block: "start" });
        return;
      }
      state.helpDocument = documentDefinition.name;
      renderHelpDocumentSwitcher();
      void loadDocumentation();
    });
    switcher.append(button);
  }
}

function embeddedDocumentationChapter(documentName, markdown, documentLocale) {
  const chapter = document.createElement("section");
  chapter.className = "documentation-chapter";
  chapter.dataset.document = documentName;
  chapter.setAttribute("translate", "no");
  chapter.lang = documentLocale;
  chapter.innerHTML = renderMarkdown(markdown);

  const languageNavigation = chapter.querySelector("p");
  const languageLinks = [...(languageNavigation?.querySelectorAll("a[href]") ?? [])]
    .filter((link) => documentationTarget(link.getAttribute("href")));
  if (languageLinks.length >= 4) {
    languageNavigation.remove();
  }

  [...chapter.querySelectorAll("h1, h2, h3, h4, h5")].forEach((heading, index) => {
    const currentLevel = Number(heading.tagName.slice(1));
    const replacement = document.createElement(`h${Math.min(currentLevel + 1, 6)}`);
    replacement.id = index === 0 ? `help-${documentName}` : `help-${documentName}-${heading.id}`;
    replacement.dataset.anchor = heading.id;
    replacement.replaceChildren(...heading.childNodes);
    heading.replaceWith(replacement);
  });
  for (const link of chapter.querySelectorAll("a[href]")) {
    const target = documentationTarget(link.getAttribute("href"));
    if (!target) continue;
    link.setAttribute("href", "#help");
    link.removeAttribute("target");
    link.removeAttribute("rel");
    link.addEventListener("click", (event) => {
      event.preventDefault();
      const currentDocumentLocale = state.helpDocument === "maintainer" ? "en" : locale();
      if (state.helpDocument === target.document && currentDocumentLocale === target.locale) {
        scrollDocumentation(chapter, target.anchor);
        return;
      }
      state.helpDocument = target.document;
      if (target.document !== "maintainer" && target.locale !== locale()) {
        state.helpPendingAnchor = target.anchor;
        setLocale(target.locale);
        return;
      }
      renderHelpDocumentSwitcher();
      void loadDocumentation(target.anchor);
    });
  }
  return chapter;
}

function filterDocumentationToc() {
  const query = $("#documentation-filter").value.trim().toLocaleLowerCase();
  $$(".documentation-toc-link", $("#documentation-toc")).forEach((button) => {
    button.hidden = Boolean(query) && !button.textContent.toLocaleLowerCase().includes(query);
  });
  $$(".documentation-toc-group", $("#documentation-toc")).forEach((group) => {
    group.hidden = $$(".documentation-toc-link", group).every((button) => button.hidden);
  });
}

function renderDocumentationToc(chapters) {
  const navigation = $("#documentation-toc");
  navigation.replaceChildren();
  for (const chapter of chapters) {
    const group = element("div", { className: "documentation-toc-group" });
    group.setAttribute("translate", "no");
    group.lang = chapter.lang;
    for (const heading of chapter.querySelectorAll("h2, h3, h4")) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `documentation-toc-link level-${heading.tagName.slice(1)}`;
      button.textContent = heading.textContent;
      button.addEventListener("click", () => {
        heading.scrollIntoView({ behavior: "smooth", block: "start" });
      });
      group.append(button);
    }
    navigation.append(group);
  }
  filterDocumentationToc();
}

async function loadDocumentation(anchor = null) {
  const revision = ++state.helpLoadRevision;
  const documentDefinition = helpDocuments.find(({ name }) => name === state.helpDocument)
    ?? helpDocuments[0];
  renderHelpDocumentSwitcher();
  const root = $("#markdown-document");
  root.replaceChildren();
  root.setAttribute("aria-busy", "true");
  const loading = element("div", { className: "loading-document" });
  loading.append(element("span"), element("strong", { text: "Loading documentation…" }));
  root.append(loading);
  $("#documentation-toc").replaceChildren();
  $("#documentation-filter").value = "";
  try {
    const documentLocale = documentDefinition.localized ? locale() : "en";
    const response = await fetch(`/guide-content/${documentLocale}/${documentDefinition.name}.md`, {
      headers: { Accept: "text/markdown" },
      cache: "no-store",
    });
    if (!response.ok) throw new Error(`${documentDefinition.name}: HTTP ${response.status}`);
    const markdown = await response.text();
    if (revision !== state.helpLoadRevision) return;
    const chapter = embeddedDocumentationChapter(documentDefinition.name, markdown, documentLocale);
    root.replaceChildren(chapter);
    root.setAttribute("aria-busy", "false");
    renderDocumentationToc([chapter]);
    if (anchor !== null) scrollDocumentation(chapter, anchor);
  } catch (error) {
    if (revision !== state.helpLoadRevision) return;
    root.setAttribute("aria-busy", "false");
    root.replaceChildren(
      element("div", { className: "empty-state" }),
    );
    root.firstElementChild.append(
      element("strong", { text: "Could not load documentation." }),
      element("span", { text: error.message }),
    );
  }
}

function statusRow(label, value, className = "") {
  const row = element("div");
  row.append(element("dt", { text: label }));
  const description = element("dd", { text: String(value ?? "—"), className });
  row.append(description);
  return row;
}

function systemProvider(system) {
  return readAny(system, "execution_provider", "provider", "onnx_runtime.provider", "runtime.execution_provider") ?? "Unknown";
}

function systemModels(system) {
  const fromSystem = listItems(system, ["models"]);
  return fromSystem.length ? fromSystem : state.models;
}

function updateCreateCollectionProfiles() {
  const select = $("#collection-create-form select[name=search_profile]");
  return applySearchProfileAvailability(select, searchProfilesFromSystem(state.system));
}

function newCollectionCropDefault() {
  return Boolean(state.system?.safe_config?.save_face_crops);
}

function systemDetectionDefaults() {
  return state.system?.safe_config?.detection ?? {
    input_sizes: [[96, 96], [512, 512]],
    threshold: 0.5,
    nms_threshold: 0.4,
    single_face_selection: "largest",
  };
}

function formatDetectorInputSizes(value) {
  return (value ?? []).map((size) => `${size[0]}x${size[1]}`).join(", ");
}

function parseDetectorInputSizes(value) {
  const parts = String(value ?? "").split(",").map((part) => part.trim()).filter(Boolean);
  if (!parts.length) throw new Error("At least one detector input size is required.");
  return parts.map((part) => {
    const match = part.match(/^(\d+)\s*[x×]\s*(\d+)$/i);
    if (!match) throw new Error(`Invalid detector input size: ${part}`);
    return [Number(match[1]), Number(match[2])];
  });
}

async function loadDashboard() {
  const [healthResult, systemResult, collectionResult, modelResult] = await Promise.allSettled([
    refreshHealth(), client.system(), client.listCollections({ limit: 100 }), client.models(),
  ]);
  if (systemResult.status === "fulfilled") {
    state.system = unwrap(systemResult.value, "system");
    updateCreateCollectionProfiles();
  }
  if (collectionResult.status === "fulfilled") {
    state.collections = listItems(collectionResult.value, ["collections"]);
    updateCollectionSelects();
  }
  if (modelResult.status === "fulfilled") state.models = listItems(modelResult.value, ["models"]);

  const system = state.system ?? {};
  state.serverErrors = Array.isArray(system.recent_errors) ? system.recent_errors : [];
  renderRecentErrors();
  const collections = state.collections;
  const personCount = Number(system.stats?.person_count ?? collections.reduce((sum, item) => sum + Number(item.person_count ?? 0), 0));
  const faceCount = Number(system.stats?.face_count ?? collections.reduce((sum, item) => sum + Number(item.face_count ?? 0), 0));
  const models = systemModels(system);
  const model = models[0] ?? {};
  const provider = systemProvider(system);
  setText("#dashboard-provider", provider);
  setText("#metric-collections", String(system.stats?.collection_count ?? collections.length));
  setText("#metric-people", String(personCount));
  setText("#metric-faces", String(faceCount));
  setText("#metric-model", firstValue(model.model_id, model.id, readAny(system, "model.model_id"), "Not loaded"));
  const livenessIndicator = $("#metric-model-liveness");
  const livenessEnabled = systemResult.status === "fulfilled"
    ? system.safe_config?.addons?.includes("liveness") === true : null;
  livenessIndicator.textContent = t(livenessRuntimeLabel(livenessEnabled));

  const status = $("#dashboard-status");
  status.replaceChildren(
    statusRow("API readiness", healthResult.status === "fulfilled" ? firstValue(healthResult.value?.status, "ready") : "unavailable"),
    statusRow("Database", readAny(system, "database.quick_check", "database.status", "database_state") ?? "unknown"),
    statusRow("Execution provider", provider),
    statusRow("ONNX Runtime", readAny(system, "onnx_runtime_version", "onnx_runtime.version", "runtime.onnx_runtime_version") ?? "—"),
  );
}

async function loadCollections() {
  const capabilityRequest = state.system === null
    ? client.system().catch((error) => {
      handleError(error, "Could not load search capabilities");
      return null;
    })
    : Promise.resolve(null);
  const [payload, systemPayload] = await Promise.all([
    guarded(null, () => client.listCollections({ limit: 100 }), "Could not load collections"),
    capabilityRequest,
  ]);
  if (systemPayload !== null) {
    state.system = unwrap(systemPayload, "system");
    updateCreateCollectionProfiles();
  }
  if (payload === undefined) return;
  state.collections = listItems(payload, ["collections"]);
  renderCollections();
  updateCollectionSelects();
  configureEmbeddingFields($("#person-create-form"));
}

function collectionLabel(collection) {
  return collection.name || collection.id || "Unnamed collection";
}

function updateCollectionSelects() {
  for (const select of [$("#people-collection"), $("#search-collection"), $("#video-collection"), $("#detect-collection"), $("#compare-collection")]) {
    if (!select) continue;
    const current = select.value;
    const prompt = select.options[0]?.textContent ?? "Select a collection";
    select.replaceChildren(new Option(prompt, ""));
    for (const collection of state.collections) select.add(new Option(`${collectionLabel(collection)} (${collection.id})`, collection.id));
    if (state.collections.some((collection) => collection.id === current)) select.value = current;
  }
  const peopleSelect = $("#people-collection");
  if (state.selectedCollection && state.collections.some((item) => item.id === state.selectedCollection)) peopleSelect.value = state.selectedCollection;
}

function renderCollections() {
  const table = $("#collection-table");
  const query = $("#collection-filter").value.trim().toLocaleLowerCase();
  const collections = state.collections.filter((item) => !query || [item.id, item.name, item.description].some((value) => String(value ?? "").toLocaleLowerCase().includes(query)));
  table.replaceChildren();
  $("#collection-empty").hidden = collections.length > 0;
  $("#collection-count").textContent = t(
    collections.length === 1 ? "{count} collection" : "{count} collections",
    { count: collections.length },
  );
  for (const collection of collections) {
    const row = document.createElement("tr");
    const identity = document.createElement("td");
    identity.append(element("span", { className: "table-title", text: collectionLabel(collection) }), element("span", { className: "table-subtitle", text: collection.id }));
    const threshold = document.createElement("td");
    threshold.append(element("span", { className: "badge", text: formatCosine(firstValue(collection.default_threshold, collection.threshold, 0)) }));
    const profile = document.createElement("td");
    profile.append(element("span", { className: "badge", text: firstValue(collection.search_profile, "fp32_v1") }));
    const crops = document.createElement("td");
    crops.append(element("span", { className: "badge", text: collection.save_face_crops ? "On" : "Off" }));
    const model = document.createElement("td");
    model.append(element("span", { className: "table-title", text: firstValue(collection.model_id, "—") }));
    const people = element("td", { text: String(collection.person_count ?? 0) });
    const faces = element("td", { text: String(collection.face_count ?? 0) });
    const actions = element("td", { className: "table-actions" });
    const manage = element("button", { className: "button secondary small", text: "People" });
    manage.type = "button";
    manage.addEventListener("click", () => {
      state.selectedCollection = collection.id;
      navigate("people");
    });
    const edit = element("button", { className: "button secondary small", text: "Edit" });
    edit.type = "button";
    edit.addEventListener("click", () => openCollectionEdit(collection));
    const remove = element("button", { className: "button danger small", text: "Delete" });
    remove.type = "button";
    remove.addEventListener("click", () => void deleteCollection(collection));
    actions.append(manage, edit, remove);
    row.append(identity, threshold, profile, crops, model, people, faces, actions);
    table.append(row);
  }
}

function openCollectionEdit(collection) {
  const form = $("#collection-edit-form");
  form.elements.id.value = collection.id;
  form.elements.name.value = collection.name ?? "";
  form.elements.description.value = collection.description ?? "";
  form.elements.threshold.value = firstValue(collection.default_threshold, collection.threshold, 0.4);
  form.elements.search_profile.value = firstValue(collection.search_profile, "fp32_v1");
  form.elements.capacity_rows.value = firstValue(collection.capacity_rows, 100000);
  form.elements.max_faces_per_person.value = firstValue(collection.max_faces_per_person, 20);
  form.elements.load_policy.value = firstValue(collection.load_policy, "lazy");
  form.elements.save_face_crops.checked = Boolean(collection.save_face_crops);
  const detection = collection.detection ?? systemDetectionDefaults();
  form.elements.detector_input_sizes.value = formatDetectorInputSizes(detection.input_sizes);
  form.elements.detector_threshold.value = firstValue(detection.threshold, 0.5);
  form.elements.detector_nms_threshold.value = firstValue(detection.nms_threshold, 0.4);
  form.elements.single_face_selection.value = firstValue(detection.single_face_selection, "largest");
  form.elements.metadata.value = JSON.stringify(collection.metadata ?? {}, null, 2);
  $("#collection-edit-dialog").showModal();
}

async function deleteCollection(collection) {
  const nonEmpty = Number(collection.person_count ?? 0) > 0 || Number(collection.face_count ?? 0) > 0;
  const prompt = nonEmpty
    ? t("Collection “{name}” is not empty. Delete it and all people and face samples?", { name: collectionLabel(collection) })
    : t("Delete collection “{name}”?", { name: collectionLabel(collection) });
  if (!window.confirm(prompt)) return;
  const result = await guarded(null, () => client.deleteCollection(collection.id, nonEmpty), "Collection could not be deleted");
  if (result === undefined) return;
  toast("Collection deleted", collection.id);
  if (state.selectedCollection === collection.id) {
    state.selectedCollection = "";
    state.selectedPerson = null;
  }
  await loadCollections();
}

async function loadPeople() {
  const collectionId = state.selectedCollection || $("#people-collection").value;
  state.selectedCollection = collectionId;
  if (!collectionId) {
    state.people = [];
    state.selectedPerson = null;
    renderPeople();
    return;
  }
  const payload = await guarded(null, () => client.listPeople(collectionId, { limit: 100 }), "Could not load people");
  if (payload === undefined) return;
  state.people = listItems(payload, ["persons", "people"]);
  if (state.selectedPerson && !state.people.some((person) => person.id === state.selectedPerson.id)) state.selectedPerson = null;
  renderPeople();
}

function initials(person) {
  const text = String(person.name || person.id || "?").trim();
  const words = text.split(/\s+/).filter(Boolean);
  return (words.length > 1 ? `${words[0][0]}${words.at(-1)[0]}` : text.slice(0, 2)).toUpperCase();
}

function renderPeople() {
  const list = $("#person-list");
  const people = filterPeople(state.people, $("#people-filter").value);
  list.replaceChildren();
  setText("#people-count", t(people.length === 1 ? "{count} person" : "{count} people", { count: people.length }));
  setText("#people-list-title", state.selectedCollection ? t("People in {collection}", { collection: state.selectedCollection }) : t("People"));
  $("#people-empty").hidden = people.length > 0;
  if (!people.length) {
    const title = state.selectedCollection ? "No people found" : "Select a collection";
    const message = state.selectedCollection ? "Register a person or change the search." : "Its enrolled people will appear here.";
    $("#people-empty").replaceChildren(element("strong", { text: title }), element("span", { text: message }));
  }
  for (const person of people) {
    const button = element("button", { className: `person-row ${state.selectedPerson?.id === person.id ? "selected" : ""}` });
    button.type = "button";
    const identity = element("span");
    identity.append(element("strong", { text: person.name || person.id }), element("small", { text: person.external_id || person.id }));
    const faceCount = Number(person.face_count ?? 0);
    button.append(
      element("span", { className: "avatar", text: initials(person) }),
      identity,
      element("span", { className: "face-count", text: t(faceCount === 1 ? "{count} face" : "{count} faces", { count: faceCount }) }),
    );
    button.addEventListener("click", () => void selectPerson(person));
    list.append(button);
  }
  if (!state.selectedPerson) renderPersonDetail();
}

async function selectPerson(person) {
  const payload = await guarded(null, () => client.getPerson(state.selectedCollection, person.id), "Could not load person");
  if (payload === undefined) return;
  state.selectedPerson = unwrap(payload, "person");
  renderPeople();
  await renderPersonDetail();
}

function renderRejections(target, rejected, files = []) {
  renderRejectionList(target, rejected, files, { element, t, livenessText });
}

async function renderPersonDetail() {
  const panel = $("#person-detail");
  const person = state.selectedPerson;
  if (!person) {
    panel.replaceChildren();
    const empty = element("div", { className: "empty-state" });
    empty.append(element("strong", { text: "No person selected" }), element("span", { text: "Choose a person to view details and face samples." }));
    panel.append(empty);
    return;
  }

  panel.innerHTML = `
    <div class="person-detail-header">
      <div><p class="eyebrow">PERSON</p><h3 class="person-name"></h3><p class="person-identity"></p></div>
      <div class="table-actions"><button class="button secondary small edit-person" type="button">Edit</button><button class="button danger small delete-person" type="button">Delete</button></div>
    </div>
    <div class="person-detail-body">
      <form class="person-edit-form" hidden>
        <div class="form-grid two"><label><span class="field-caption">Name</span><input name="name" maxlength="200"></label><label><span class="field-caption">External ID</span><input name="external_id" maxlength="200"></label></div>
        <label>Metadata (JSON)<textarea name="metadata" rows="3" spellcheck="false"></textarea></label>
        <div class="form-actions"><button class="button primary small" type="submit">Save</button><button class="button secondary small cancel-edit" type="button">Cancel</button></div>
      </form>
      <div class="person-summary"><p class="eyebrow">METADATA</p><pre class="person-metadata"></pre></div>
      <div><div class="panel-heading"><div><p class="eyebrow">FACE SAMPLES</p><h3 class="faces-heading">Registered faces</h3></div></div>
        <form class="add-faces-form"><label class="drop-zone compact-drop"><input name="images" type="file" accept="image/jpeg,image/png,image/webp,image/bmp,image/x-ms-bmp,.bmp" multiple required><strong>Add registration photos</strong><span>JPEG, PNG, WebP, or BMP</span><span class="file-summary">No files selected</span></label><div class="form-grid two"><label><span class="field-caption">Enrollment review</span><select name="review_mode"><option value="off" selected>Off · collection selection</option><option value="standard">Standard quality</option><option value="strict">Strict identity</option></select></label><label><span class="field-caption">Embedding source</span><select name="embedding_mode"><option value="server" selected>Extract from image</option><option value="external_trusted">Trusted external feature</option></select></label></div><p class="hint">Off uses the Collection single-face selection strategy; standard and strict require exactly one.</p><p class="hint embedding-contract-summary">Collection embedding contract: <code class="embedding-contract"></code></p><div class="external-embedding-fields" hidden><p class="hint">The server trusts the image/vector pairing and does not re-extract the feature.</p><label>External embeddings (JSON, one vector per image)<textarea name="external_embeddings" rows="5" spellcheck="false" placeholder="[[0.0123, -0.0456, ...]]"></textarea></label></div><div class="form-actions"><button class="button secondary small" type="submit">Add face samples</button><span class="form-status" role="status"></span></div></form>
        <div class="rejection-list face-rejections" hidden></div><div class="face-sample-grid"></div>
      </div>
    </div>`;

  translateTree(panel);
  $(".person-name", panel).textContent = person.name || person.id;
  $(".person-identity", panel).textContent = [person.id, person.external_id].filter(Boolean).join(" · ");
  $(".person-metadata", panel).textContent = JSON.stringify(person.metadata ?? {}, null, 2);
  const editForm = $(".person-edit-form", panel);
  editForm.elements.name.value = person.name ?? "";
  editForm.elements.external_id.value = person.external_id ?? "";
  editForm.elements.metadata.value = JSON.stringify(person.metadata ?? {}, null, 2);
  $(".edit-person", panel).addEventListener("click", () => { editForm.hidden = false; });
  $(".cancel-edit", panel).addEventListener("click", () => { editForm.hidden = true; });
  $(".delete-person", panel).addEventListener("click", () => void deleteSelectedPerson());
  editForm.addEventListener("submit", (event) => void updateSelectedPerson(event));
  const addForm = $(".add-faces-form", panel);
  setupFileInput($("input[type=file]", addForm));
  addForm.elements.embedding_mode.addEventListener("change", () => configureEmbeddingFields(addForm));
  configureEmbeddingFields(addForm);
  addForm.addEventListener("submit", (event) => void addPersonFaces(event));

  const facePayload = await guarded(null, () => client.listFaces(state.selectedCollection, person.id, { limit: 100 }), "Could not load face samples");
  if (facePayload === undefined || state.selectedPerson?.id !== person.id) return;
  renderFaceSamples(listItems(facePayload, ["faces", "face_samples"]));
}

function renderFaceSamples(faces) {
  const panel = $("#person-detail");
  const grid = $(".face-sample-grid", panel);
  if (!grid) return;
  releaseAllCropObjectUrls();
  grid.replaceChildren();
  $(".faces-heading", panel).textContent = t("Registered faces ({count})", { count: faces.length });
  if (!faces.length) {
    const empty = element("div", { className: "empty-state compact" });
    empty.append(element("strong", { text: "No face samples" }), element("span", { text: "Add one or more clear photos." }));
    grid.append(empty);
    return;
  }
  for (const face of faces) {
    const card = element("article", { className: "face-card" });
    card.append(element("strong", { text: face.id }));
    const details = document.createElement("dl");
    details.append(
      statusRow("Detection", formatScore(firstValue(face.detection_score, face.confidence))),
      statusRow("Quality", formatScore(firstValue(face.quality?.score, face.quality_score))),
      statusRow("Embedding", face.embedding_source === "external_trusted" ? "External trusted" : "Server extracted"),
      statusRow("Stored crop", face.has_crop ? "Available" : "Not saved"),
      statusRow("Created", formatTimestamp(face.created_at)),
    );
    if (face.liveness) details.append(statusRow("Liveness", livenessText(face)));
    const crop = document.createElement("img");
    crop.alt = t("Saved crop for face sample {id}", { id: face.id });
    crop.hidden = true;
    if (face.has_crop) {
      const view = element("button", { className: "button secondary small full-width", text: "View crop" });
      view.type = "button";
      view.addEventListener("click", () => void toggleFaceCrop(face, view, crop));
      card.append(view, crop);
    }
    const remove = element("button", { className: "button danger small full-width", text: "Delete sample" });
    remove.type = "button";
    remove.addEventListener("click", () => void deleteFaceSample(face));
    card.append(details, remove);
    grid.append(card);
  }
}

async function toggleFaceCrop(face, button, image) {
  if (!image.hidden) {
    releaseCropObjectUrl(image.dataset.cropUrl);
    delete image.dataset.cropUrl;
    image.hidden = true;
    image.removeAttribute("src");
    button.textContent = t("View crop");
    return;
  }
  const blob = await guarded(
    button,
    () => client.getFaceCrop(state.selectedCollection, state.selectedPerson.id, face.id),
    "Face crop could not be loaded",
    "Loading…",
  );
  if (blob === undefined || !button.isConnected) return;
  const url = URL.createObjectURL(blob);
  state.cropObjectUrls.add(url);
  image.dataset.cropUrl = url;
  const release = () => {
    releaseCropObjectUrl(url);
    if (image.dataset.cropUrl === url) delete image.dataset.cropUrl;
  };
  image.addEventListener("load", release, { once: true });
  image.addEventListener("error", () => {
    release();
    image.hidden = true;
    image.removeAttribute("src");
    button.textContent = t("View crop");
    handleError(new Error("The stored face crop could not be decoded."), "Face crop could not be displayed");
  }, { once: true });
  image.src = url;
  image.hidden = false;
  button.textContent = t("Hide crop");
}

function formatTimestamp(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}

async function updateSelectedPerson(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const submit = $("button[type=submit]", form);
  let metadata;
  try { metadata = parseMetadata(form.elements.metadata.value); } catch (error) { handleError(error, "Invalid person metadata"); return; }
  const payload = await guarded(submit, () => client.updatePerson(state.selectedCollection, state.selectedPerson.id, {
    name: form.elements.name.value.trim() || null,
    external_id: form.elements.external_id.value.trim() || null,
    metadata,
  }), "Person could not be updated", "Saving…");
  if (payload === undefined) return;
  state.selectedPerson = unwrap(payload, "person");
  toast("Person updated", state.selectedPerson.id);
  await loadPeople();
  const refreshed = state.people.find((item) => item.id === state.selectedPerson?.id);
  if (refreshed) await selectPerson(refreshed);
}

async function deleteSelectedPerson() {
  const person = state.selectedPerson;
  if (!person || !window.confirm(t("Delete “{name}” and all registered face samples?", { name: person.name || person.id }))) return;
  const result = await guarded(null, () => client.deletePerson(state.selectedCollection, person.id), "Person could not be deleted");
  if (result === undefined) return;
  toast("Person deleted", person.id);
  state.selectedPerson = null;
  await loadPeople();
}

async function addPersonFaces(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const button = $("button[type=submit]", form);
  const images = [...form.elements.images.files];
  if (!images.length) return;
  let embeddingInput;
  try { embeddingInput = enrollmentEmbeddingInput(form, images); } catch (error) { handleError(error, "Invalid external embeddings"); return; }
  const payload = await guarded(button, () => client.addFaces(state.selectedCollection, state.selectedPerson.id, images, { reviewMode: form.elements.review_mode.value, ...embeddingInput }), "Face samples could not be added", "Uploading…");
  if (payload === undefined) return;
  renderRejections($(".face-rejections", $("#person-detail")), payload.rejected_images ?? payload.rejected ?? [], images);
  const addedCount = listItems(payload, ["faces", "face_samples"]).length;
  toast("Registration complete", t(addedCount === 1 ? "{count} face sample added." : "{count} face samples added.", { count: addedCount }));
  form.elements.external_embeddings.value = "";
  const id = state.selectedPerson.id;
  await loadPeople();
  const person = state.people.find((item) => item.id === id);
  if (person) await selectPerson(person);
}

async function deleteFaceSample(face) {
  if (!window.confirm(t("Delete face sample {id}?", { id: face.id }))) return;
  const result = await guarded(null, () => client.deleteFace(state.selectedCollection, state.selectedPerson.id, face.id), "Face sample could not be deleted");
  if (result === undefined) return;
  toast("Face sample deleted", face.id);
  const person = state.selectedPerson;
  await selectPerson(person);
}

function fileSummary(files) {
  if (!files?.length) return t("No files selected");
  if (files.length === 1) return files[0].name;
  return t("{count} files selected", { count: files.length });
}

function setupFileInput(input) {
  if (!input || input.dataset.ready) return;
  input.dataset.ready = "true";
  const zone = input.closest(".drop-zone");
  const summary = $(".file-summary", zone);
  const update = () => { if (summary) summary.textContent = fileSummary(input.files); };
  input.addEventListener("change", update);
  for (const eventName of ["dragenter", "dragover"]) zone.addEventListener(eventName, () => zone.classList.add("dragging"));
  for (const eventName of ["dragleave", "drop"]) zone.addEventListener(eventName, () => zone.classList.remove("dragging"));
}

function setupImagePreview(input, canvas, onReset = () => {}) {
  input.addEventListener("change", () => {
    const file = input.files[0];
    onReset();
    if (!file) {
      canvas.closest(".canvas-stage")?.classList.add("empty");
      return;
    }
    void renderImageFaces(file, canvas).catch((error) => handleError(error, "Image preview failed"));
  });
}

function imageFromFile(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => { URL.revokeObjectURL(url); resolve(image); };
    image.onerror = () => { URL.revokeObjectURL(url); reject(new Error("The selected image could not be decoded.")); };
    image.src = url;
  });
}

function drawFaceOverlays(canvas, faces, { labels = [] } = {}) {
  const context = canvas.getContext("2d");
  const lineWidth = Math.max(2, Math.round(canvas.width / 600 * 2));
  context.lineWidth = lineWidth;
  context.font = `${Math.max(12, Math.round(canvas.width / 60))}px ui-sans-serif, sans-serif`;
  context.textBaseline = "bottom";
  faces.forEach((face, index) => {
    const box = facePixels(face, canvas.width, canvas.height);
    if (!box) return;
    const color = index === 0 ? "#818cf8" : "#22d3ee";
    context.strokeStyle = color;
    context.fillStyle = color;
    context.strokeRect(box.x, box.y, box.width, box.height);
    for (const point of faceLandmarks(face)) {
      const x = Math.abs(point.x) <= 1 ? point.x * canvas.width : point.x;
      const y = Math.abs(point.y) <= 1 ? point.y * canvas.height : point.y;
      context.beginPath();
      context.arc(x, y, Math.max(2, lineWidth * 1.2), 0, Math.PI * 2);
      context.fill();
    }
    const label = labels[index];
    if (label) {
      const metrics = context.measureText(label);
      const textHeight = Math.max(17, Math.round(canvas.width / 45));
      const top = Math.max(0, box.y - textHeight);
      context.fillStyle = "rgba(9, 9, 11, .9)";
      context.fillRect(box.x, top, metrics.width + 12, textHeight);
      context.fillStyle = color;
      context.fillText(label, box.x + 6, top + textHeight - 2);
    }
  });
}

async function renderImageFaces(file, canvas, faces = [], options = {}) {
  const image = await imageFromFile(file);
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const context = canvas.getContext("2d");
  context.drawImage(image, 0, 0);
  drawFaceOverlays(canvas, faces, options);
  canvas.closest(".canvas-stage")?.classList.remove("empty");
}

function resultFaces(payload) {
  return listItems(payload, ["faces", "detections"]);
}

function livenessText(face, options = {}) {
  return livenessResultText(face?.liveness, { translate: t, formatScore, ...options });
}

function renderFaceResults(target, faces) {
  target.replaceChildren();
  faces.forEach((face, index) => {
    const quality = face.quality ?? {};
    const row = element("div", { className: "face-result" });
    const details = element("div", { className: "quality-bars" });
    details.append(
      element("span", { text: t("Detection {score}", { score: formatScore(firstValue(face.detection_score, face.confidence)) }) }),
      element("span", { text: t("Quality {score}", { score: formatScore(quality.score) }) }),
      element("span", { text: t("Sharpness {score}", { score: formatScore(quality.sharpness) }) }),
    );
    if (face.liveness) details.append(element("span", { text: livenessText(face) }));
    row.append(element("span", { className: "face-index", text: String(index + 1) }), details, element("strong", { text: t("Brightness {score}", { score: formatScore(quality.brightness) }) }));
    target.append(row);
  });
}

async function submitDetect(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const button = $("button[type=submit]", form);
  const file = form.elements.image.files[0];
  $("#detect-results").replaceChildren();
  $("#detect-summary").textContent = "";
  await renderImageFaces(file, $("#detect-canvas"));
  const payload = await guarded(button, () => client.detect(file, { maxFaces: form.elements.max_faces.value, collectionId: form.elements.collection.value }), "Detection failed", "Detecting…");
  if (payload === undefined) return;
  const faces = resultFaces(payload);
  await renderImageFaces(file, $("#detect-canvas"), faces, { labels: faces.map((face, index) => `#${index + 1} · ${formatScore(firstValue(face.detection_score, face.confidence))}`) });
  renderFaceResults($("#detect-results"), faces);
  $("#detect-summary").textContent = t(
    faces.length === 1 ? "{count} face detected · {duration}" : "{count} faces detected · {duration}",
    { count: faces.length, duration: formatDuration(payload.processing_ms) },
  );
}

async function submitCompare(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const button = $("button[type=submit]", form);
  const source = form.elements.source.files[0];
  const target = form.elements.target.files[0];
  const threshold = form.elements.threshold.value;
  resetCompareVerdict();
  await Promise.all([
    renderImageFaces(source, $("#compare-source-canvas")),
    renderImageFaces(target, $("#compare-target-canvas")),
  ]);
  const payload = await guarded(button, () => client.compare(source, target, { threshold, collectionId: form.elements.collection.value }), "Comparison failed", "Comparing…");
  if (payload === undefined) {
    $("#compare-verdict strong").textContent = t("Comparison not performed");
    return;
  }
  await Promise.all([
    renderImageFaces(source, $("#compare-source-canvas"), [payload.source_face].filter(Boolean)),
    renderImageFaces(target, $("#compare-target-canvas"), [payload.target_face].filter(Boolean)),
  ]);
  const verdict = $("#compare-verdict");
  verdict.className = `compare-verdict ${payload.matched ? "match" : "no-match"}`;
  $(".verdict-mark", verdict).textContent = payload.matched ? "✓" : "×";
  $("strong", verdict).textContent = t(payload.matched ? "Match" : "No match");
  $("small", verdict).textContent = t("{similarity} similarity · threshold {threshold} · {duration}", {
    similarity: formatCosine(payload.similarity),
    threshold: formatCosine(payload.threshold),
    duration: formatDuration(payload.processing_ms),
  });
  const liveness = [
    livenessText(payload.source_face, { side: "source" }),
    livenessText(payload.target_face, { side: "target" }),
  ].filter(Boolean);
  if (liveness.length) $("small", verdict).textContent += ` · ${liveness.join(" / ")}`;
}

function renderMatches(target, matches) {
  target.replaceChildren();
  if (!matches.length) {
    const empty = element("div", { className: "empty-state compact" });
    empty.append(element("strong", { text: "No matches above threshold" }), element("span", { text: "Try a clearer image or review the collection threshold." }));
    target.append(empty);
    return;
  }
  matches.forEach((match, index) => {
    const person = match.person ?? {};
    const card = element("div", { className: "match-card" });
    const identity = element("div", { className: "match-person" });
    const metadata = person.metadata && Object.keys(person.metadata).length ? JSON.stringify(person.metadata) : "";
    identity.append(
      element("strong", { text: person.name || person.id || match.person_id || "Unknown" }),
      element("small", { text: [person.external_id || person.id, t("Face {id}", { id: match.matched_face_id ?? "—" }), metadata].filter(Boolean).join(" · "), title: metadata }),
    );
    card.append(element("span", { className: "match-rank", text: `#${index + 1}` }), identity, element("span", { className: "match-score", text: formatCosine(match.similarity) }));
    target.append(card);
  });
}

async function submitSearch(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const button = $("button[type=submit]", form);
  const file = form.elements.image.files[0];
  $("#search-results").replaceChildren();
  $("#search-summary").textContent = "";
  await renderImageFaces(file, $("#search-canvas"));
  const payload = await guarded(button, () => client.search(form.elements.collection.value, file, {
    limit: form.elements.limit.value,
    threshold: form.elements.threshold.value,
  }), "Search failed", "Searching…");
  if (payload === undefined) {
    $("#search-summary").textContent = t("Search not performed");
    return;
  }
  const searchedFace = payload.searched_face;
  await renderImageFaces(file, $("#search-canvas"), [searchedFace].filter(Boolean), { labels: [t("Search face")] });
  const matches = listItems(payload, ["matches"]);
  renderMatches($("#search-results"), matches);
  const summary = t(
    matches.length === 1 ? "{count} match · threshold {threshold} · {duration}" : "{count} matches · threshold {threshold} · {duration}",
    { count: matches.length, threshold: formatCosine(payload.threshold), duration: formatDuration(payload.processing_ms) },
  );
  $("#search-summary").textContent = [summary, livenessText(searchedFace)].filter(Boolean).join(" · ");
}

async function loadSystem() {
  const livenessRequest = livenessManager.start();
  const button = $("#system-refresh");
  const result = await guarded(button, async () => {
    const [systemPayload, modelsPayload] = await Promise.all([
      client.system(),
      client.models(),
    ]);
    return {
      system: unwrap(systemPayload, "system"),
      models: [...listItems(modelsPayload, ["models"]), ...(modelsPayload?.addons ?? [])],
      license: modelsPayload?.license ?? readAny(systemPayload, "model.license") ?? null,
    };
  }, "Could not load system diagnostics", "Refreshing…");
  await livenessRequest;
  if (!result) return;
  state.system = result.system;
  state.models = result.models;
  state.modelLicense = result.license;
  updateCreateCollectionProfiles();
  renderSystem();
}

function renderLivenessManagement() {
  const snapshot = state.livenessManagement ?? {};
  const view = livenessManagementView(snapshot);
  setText("#system-liveness-runtime", t(view.runtimeLabel));
  $("#system-liveness-details").replaceChildren(
    statusRow("Model file", view.installedLabel),
    statusRow("After restart", view.afterRestartLabel),
    ...(view.error || view.unavailableReason ? [
      statusRow("Model path", snapshot.status?.model_path ?? "—"),
      statusRow("Startup config", snapshot.status?.config_file ?? "built-in defaults"),
    ] : []),
  );
  const notice = $("#system-liveness-notice");
  notice.textContent = t(view.notice);
  notice.hidden = !view.notice;
  const error = $("#system-liveness-error");
  error.textContent = view.error ? `${view.error.code || "request_failed"}: ${livenessMessage(view.error.code, view.error.message, t)}` : "";
  error.hidden = !view.error;
  const refresh = $("#system-liveness-refresh");
  refresh.hidden = !snapshot.error;
  refresh.disabled = Boolean(snapshot.loading);
  const action = $("#system-liveness-enable");
  action.textContent = t(view.actionLabel);
  action.hidden = !view.showAction;
  action.disabled = view.actionDisabled;
  const manual = $("#system-liveness-manual");
  manual.hidden = !view.unavailableReason;
  setText("#system-liveness-unavailable", view.unavailableReason
    ? livenessMessage(view.unavailableCode, view.unavailableReason, t) : "");
}

function renderSystem() {
  const system = state.system ?? {};
  const osName = firstValue(readAny(system, "os.name"), readAny(system, "platform.os"), system.os, "—");
  const architecture = firstValue(readAny(system, "os.architecture"), readAny(system, "platform.architecture"), system.architecture, "—");
  const provider = systemProvider(system);
  const gpuEntries = listItems(readAny(system, "runtime.gpus", "gpu", "gpus", "hardware.gpus"), ["devices"]);
  const gpu = gpuEntries[0] ?? (typeof system.gpu === "object" ? system.gpu : {});
  setText("#system-server", firstValue(system.server_version, system.version, "—"));
  setText("#system-platform", `${osName} · ${architecture}`);
  setText("#system-provider", provider);
  setText("#system-ort", `ORT ${readAny(system, "runtime.onnx_runtime_version", "onnx_runtime_version", "onnx_runtime.version") ?? "—"}`);
  setText("#system-gpu", firstValue(gpu.name, gpu.model, system.gpu_model, "CPU"));
  setText("#system-driver", `Driver ${firstValue(gpu.driver_version, system.nvidia_driver, system.driver_version, "—")}`);
  setText("#system-auth", t(client.hasApiKey ? "Configured" : "Not configured"));
  const detectorInputSizes = readAny(system, "runtime.detector_input_sizes", "safe_config.detector_input_sizes");
  const detectorInputLabel = Array.isArray(detectorInputSizes)
    ? detectorInputSizes.map((size) => Array.isArray(size) ? size.join(" × ") : String(size)).join(" + ")
    : "—";

  $("#system-runtime").replaceChildren(
    statusRow("OS", osName),
    statusRow("Architecture", architecture),
    statusRow("CPU", firstValue(readAny(system, "cpu.model"), system.cpu_model, system.cpu, "—")),
    statusRow("GPU", firstValue(gpu.name, gpu.model, system.gpu_model, "Not reported")),
    statusRow("Compute capability", firstValue(gpu.compute_capability, system.compute_capability, "—")),
    statusRow("NVIDIA driver", firstValue(gpu.driver_version, system.nvidia_driver, system.driver_version, "—")),
    statusRow("CUDA runtime", firstValue(readAny(system, "runtime.cuda_runtime_version"), system.cuda_runtime_version, readAny(system, "cuda.runtime_version"), "—")),
    statusRow("cuDNN", firstValue(readAny(system, "runtime.cudnn_version"), system.cudnn_version, readAny(system, "cudnn.version"), "—")),
    statusRow("ONNX Runtime", firstValue(readAny(system, "runtime.onnx_runtime_version"), system.onnx_runtime_version, readAny(system, "onnx_runtime.version"), "—")),
    statusRow("Execution provider", provider),
    statusRow("Inference concurrency", firstValue(readAny(system, "runtime.inference_concurrency.max_concurrency"), system.safe_config?.inference_max_concurrency, "—")),
    statusRow("Detector input sizes", detectorInputLabel),
    statusRow("Detection threshold", firstValue(system.safe_config?.detection?.threshold, "—")),
    statusRow("Detector NMS threshold", firstValue(system.safe_config?.detection?.nms_threshold, "—")),
    statusRow("Single-face selection", firstValue(system.safe_config?.detection?.single_face_selection, "—")),
    statusRow("Liveness", system.safe_config?.addons?.includes("liveness") ? "enabled" : "disabled"),
    ...(system.safe_config?.addons?.includes("liveness") ? [
      statusRow("Liveness mode", ({ normal: "normal · require liveness", observe: "observe · continue recognition" })[system.safe_config.liveness_mode]
        ?? system.safe_config.liveness_mode),
      statusRow("Liveness threshold", system.safe_config.liveness_threshold),
      statusRow("Compare liveness scope", ({ both: "Both faces", source: "Source face", target: "Target face" })[system.safe_config.liveness_compare_scope]
        ?? system.safe_config.liveness_compare_scope),
      statusRow("Liveness on registration", system.safe_config.liveness_on_registration ? "enabled" : "disabled"),
    ] : []),
    statusRow("Startup config", firstValue(system.safe_config?.config_file, "built-in defaults")),
  );
  const databaseStatus = firstValue(readAny(system, "database.quick_check"), readAny(system, "database.status"), system.database_status, "unknown");
  const dataStatus = system.data && typeof system.data === "object" ? (system.data.exists && system.data.writable ? "ready, writable" : "unavailable") : firstValue(readAny(system, "paths.data.status"), system.data_status, "unknown");
  const modelsStatus = system.models && typeof system.models === "object" ? (system.models.exists && system.models.readable ? "ready, readable" : "unavailable") : firstValue(readAny(system, "models_path.status"), readAny(system, "paths.models.status"), system.models_status, "unknown");
  const storageRows = [
    statusRow("Database", databaseStatus, /ok|healthy|ready/i.test(String(databaseStatus)) ? "good" : ""),
    statusRow("Database path", firstValue(readAny(system, "database.path"), system.database_path, "—")),
    statusRow("/data", dataStatus, /ok|ready|writable/i.test(String(dataStatus)) ? "good" : ""),
    statusRow("/data path", firstValue(readAny(system, "data.path"), readAny(system, "paths.data.path"), "/data")),
    statusRow("/models", modelsStatus, /ok|ready|read.?only/i.test(String(modelsStatus)) ? "good" : ""),
    statusRow("/models path", firstValue(readAny(system, "models.path"), readAny(system, "models_path.path"), readAny(system, "paths.models.path"), "/models")),
    statusRow("Authentication", system.api_key?.authentication_enabled === false ? "development mode (disabled)" : system.api_key?.authentication_enabled === true ? "enabled" : "unknown"),
    statusRow("Maximum image size", system.safe_config?.max_image_bytes ? `${(Number(system.safe_config.max_image_bytes) / 1024 / 1024).toFixed(1)} MiB` : "—"),
    statusRow("Maximum pixels", firstValue(system.safe_config?.max_image_pixels, "—")),
    statusRow("Request timeout", system.safe_config?.request_timeout_seconds ? t("{count} seconds", { count: system.safe_config.request_timeout_seconds }) : "—"),
    statusRow("New-collection crop default", system.safe_config?.save_face_crops ? "enabled" : "disabled"),
  ];
  if (system.api_key?.authentication_enabled !== false) {
    storageRows.splice(
      7,
      0,
      statusRow("Server API key", system.api_key?.configured === true ? "configured" : system.api_key?.configured === false ? "not configured" : "unknown"),
    );
  }
  $("#system-storage").replaceChildren(...storageRows);

  const modelGrid = $("#system-models");
  modelGrid.replaceChildren();
  for (const model of systemModels(system)) {
    const card = element("article", { className: "model-card" });
    card.append(element("h4", { text: firstValue(model.model_id, model.id, "Unnamed model") }));
    const details = document.createElement("dl");
    details.append(
      statusRow("Task", firstValue(model.task, "—")),
      statusRow("Digest", firstValue(model.sha256, model.model_digest, "—")),
      statusRow("Input", Array.isArray(model.input_size) ? model.input_size.join(" × ") : "—"),
      statusRow("Embedding", firstValue(model.embedding_dimension, "—")),
    );
    card.append(details);
    modelGrid.append(card);
  }
  if (!modelGrid.children.length) modelGrid.append(element("p", { className: "muted", text: "No loaded models were reported." }));

  const license = state.modelLicense ?? readAny(system, "model.license");
  const licenseNotice = $("#system-license-notice");
  const licenseHeading = $("#system-license-heading");
  const licenseSummary = $("#system-license-summary");
  const commercialRow = $("#system-license-commercial-row");
  if (license && typeof license === "object") {
    const values = [
      license.license_id,
      license.grant,
      license.valid_from,
      license.valid_until,
      license.customer,
      license.reference,
    ]
      .filter((value) => value !== undefined && value !== null && value !== "");
    if (licenseNotice) licenseNotice.hidden = false;
    if (licenseHeading) licenseHeading.textContent = firstValue(license.model_id, t("Model"));
    if (licenseSummary && values.length) licenseSummary.textContent = values.join(" · ");
    if (commercialRow) commercialRow.hidden = license.commercial_use_permitted === true;
  } else {
    if (licenseNotice) licenseNotice.hidden = true;
  }
}

function selectedMonitor() {
  return state.monitors.find((monitor) => monitor.id === state.selectedMonitorId) ?? null;
}

function videoIdentity(result) {
  const person = result?.person ?? {};
  return {
    id: String(person.id ?? ""),
    name: String(person.name ?? ""),
    similarity: Number(result?.similarity),
  };
}

function monitorStatusLabel(status) {
  return {
    running: "Running",
    starting: "Starting",
    reconnecting: "Reconnecting",
    degraded: "Degraded",
    stopped: "Stopped",
    error: "Error",
  }[status] || status || "Unavailable";
}

function monitorEventLabel(type) {
  return {
    person_enter: "Known person entered",
    person_exit: "Known person left",
    unknown_enter: "Unknown person entered",
    unknown_exit: "Unknown person left",
    monitor_error: "Monitor error",
    monitor_recovered: "Monitor recovered",
  }[type] || type || "Event";
}

function renderMonitorList() {
  const target = $("#monitor-list");
  target.replaceChildren();
  $("#monitor-empty").hidden = state.monitors.length > 0;
  for (const monitor of state.monitors) {
    const button = element("button", {
      className: `monitor-list-item ${monitor.id === state.selectedMonitorId ? "selected" : ""}`,
    });
    button.type = "button";
    const identity = element("span");
    identity.append(
      element("strong", { text: monitor.name || monitor.id }),
      element("small", { text: `${monitor.id} · ${monitor.collection_id}` }),
    );
    const runtimeStatus = monitor.runtime?.status || (monitor.enabled ? "starting" : "stopped");
    const status = element("span", {
      className: `monitor-list-state ${runtimeStatus}`,
      text: monitorStatusLabel(runtimeStatus),
    });
    button.append(identity, status);
    button.addEventListener("click", () => selectMonitor(monitor.id));
    target.append(button);
  }
}

function renderSelectedMonitor() {
  const monitor = selectedMonitor();
  $("#selected-monitor-name").textContent = monitor?.name || t("Select a Monitor");
  $("#selected-monitor-id").textContent = monitor?.id || "—";
  const toggle = $("#monitor-toggle");
  const previewToggle = $("#monitor-preview-toggle");
  const edit = $("#monitor-edit");
  const remove = $("#monitor-delete");
  for (const control of [toggle, previewToggle, edit, remove]) control.disabled = !monitor;
  toggle.textContent = t(monitor?.enabled ? "Stop" : "Start");
  previewToggle.textContent = t(monitor?.preview_enabled ? "Disable preview" : "Enable preview");
  $("#monitor-config").hidden = !monitor;
  $("#monitor-source").textContent = monitor?.source?.url || "—";
  $("#monitor-collection").textContent = monitor?.collection_id || "—";
  $("#monitor-threshold").textContent = monitor
    ? monitor.match_threshold === null || monitor.match_threshold === undefined
      ? t("Collection default")
      : formatCosine(monitor.match_threshold)
    : "—";
  $("#monitor-event-capacity").textContent = monitor
    ? String(monitor.event_buffer_size ?? "—")
    : "—";
  if (!monitor) {
    $("#video-idle-title").textContent = t("No Monitor selected");
    $("#video-idle-message").textContent = t("Select a Monitor to inspect its live recognition state.");
    $("#video-status").textContent = t("Select a Monitor to inspect it.");
    $("#video-metrics").hidden = true;
    renderVideoMatches([]);
    renderMonitorEvents();
  } else if (!monitor.preview_enabled) {
    $("#video-idle-title").textContent = t("Preview is disabled");
    $("#video-idle-message").textContent = t("Recognition and events continue without video preview.");
  } else if (!monitor.enabled) {
    $("#video-idle-title").textContent = t("Monitor is stopped.");
    $("#video-idle-message").textContent = t("Start the Monitor to receive frames.");
  } else {
    $("#video-idle-title").textContent = t("Waiting for preview");
    $("#video-idle-message").textContent = t("The preview will appear after the RTSP source connects.");
  }
}

async function loadMonitors({ preserveSelection = true } = {}) {
  const payload = await guarded(
    null,
    () => client.listMonitors({ limit: 100 }),
    "Could not load Monitors",
  );
  if (payload === undefined) return;
  const previous = preserveSelection ? state.selectedMonitorId : "";
  state.monitors = listItems(payload, ["monitors"]);
  state.selectedMonitorId = state.monitors.some((item) => item.id === previous)
    ? previous
    : state.monitors[0]?.id || "";
  renderMonitorList();
  renderSelectedMonitor();
  if (state.selectedMonitorId) {
    state.monitorEventCursor = "";
    state.monitorEvents = [];
    stopMonitorPreview();
    void pollMonitor(state.selectedMonitorId);
  }
}

function selectMonitor(monitorId) {
  if (monitorId === state.selectedMonitorId) return;
  if (state.editingMonitorId) closeMonitorForm();
  state.selectedMonitorId = monitorId;
  state.monitorEventCursor = "";
  state.monitorEvents = [];
  state.videoStatus = null;
  if (state.monitorPollTimer !== null) window.clearTimeout(state.monitorPollTimer);
  state.monitorPollTimer = null;
  stopMonitorPreview();
  renderMonitorList();
  renderSelectedMonitor();
  renderMonitorEvents();
  void pollMonitor(monitorId);
}

function renderVideoMatches(recognitions = []) {
  const target = $("#video-matches");
  target.replaceChildren();
  recognitions.forEach((result, index) => {
    const recognized = result?.status === "matched";
    const row = element("div", { className: `video-match ${recognized ? "recognized" : "not-enrolled"}` });
    const number = element("span", { className: "face-index", text: String(index + 1) });
    const summary = element("div", { className: "video-match-person" });
    if (recognized) {
      const identity = videoIdentity(result);
      summary.append(
        element("strong", { text: identity.name || identity.id }),
        element("small", { text: t("ID {id}", { id: identity.id }) }),
      );
      row.append(number, summary, element("strong", { text: formatCosine(identity.similarity) }));
    } else {
      const score = formatScore(result?.face?.detection_score);
      summary.append(
        element("strong", { text: result?.status === "liveness_blocked" ? t("Liveness blocked recognition") : t("Detected, not enrolled") }),
        element("small", { text: t("Detection score {score}", { score }) }),
      );
      row.append(number, summary, element("strong", { text: "—" }));
    }
    if (result?.face?.liveness) summary.append(element("small", { text: livenessText(result.face) }));
    target.append(row);
  });
}

function drawMonitorOverlay(stream) {
  const canvas = $("#rtsp-overlay");
  const stage = canvas.parentElement;
  const width = stage.clientWidth;
  const height = stage.clientHeight;
  if (!width || !height) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  context.clearRect(0, 0, width, height);
  const sourceWidth = Number(stream?.source?.width);
  const sourceHeight = Number(stream?.source?.height);
  if (!sourceWidth || !sourceHeight || !stream?.connected) return;
  const scale = Math.min(width / sourceWidth, height / sourceHeight);
  const renderedWidth = sourceWidth * scale;
  const renderedHeight = sourceHeight * scale;
  const offsetX = (width - renderedWidth) / 2;
  const offsetY = (height - renderedHeight) / 2;
  context.font = "700 13px system-ui, sans-serif";
  context.lineWidth = 2;
  for (const result of stream.faces ?? []) {
    const box = result?.face?.bbox?.normalized;
    if (!box) continue;
    const x = offsetX + Number(box.left) * renderedWidth;
    const y = offsetY + Number(box.top) * renderedHeight;
    const boxWidth = Number(box.width) * renderedWidth;
    const boxHeight = Number(box.height) * renderedHeight;
    const recognized = result.status === "matched";
    const blocked = result.status === "liveness_blocked";
    const color = blocked ? "#f87171" : recognized ? "#57d368" : "#ff9e20";
    const identity = videoIdentity(result);
    const baseLabel = recognized
      ? `${identity.name || identity.id} ${formatCosine(identity.similarity)}`
      : blocked ? t("Liveness blocked recognition") : t("Not enrolled");
    const label = [baseLabel, livenessText(result.face, { compact: true })].filter(Boolean).join(" · ");
    context.strokeStyle = color;
    context.strokeRect(x, y, boxWidth, boxHeight);
    const labelWidth = context.measureText(label).width + 12;
    const labelY = Math.max(0, y - 23);
    context.fillStyle = color;
    context.fillRect(x, labelY, labelWidth, 23);
    context.fillStyle = "#09090b";
    context.fillText(label, x + 6, labelY + 16);
  }
}

function renderVideoStatus(stream) {
  state.videoStatus = stream;
  $("#video-metrics").hidden = false;
  const connected = stream?.connected === true;
  const statusLabel = connected
    ? "Connected"
    : ["starting", "connecting"].includes(stream?.status)
      ? "Connecting"
      : stream?.status === "reconnecting"
        ? "Reconnecting"
        : stream?.status || "Unavailable";
  $("#video-connection").textContent = t(statusLabel);
  const effective = Number(stream?.inference?.actual_fps);
  const requested = Number(stream?.inference?.configured_fps);
  $("#video-effective-fps").textContent = Number.isFinite(effective) && Number.isFinite(requested)
    ? t("{effective} / {requested} per second", { effective: effective.toFixed(2), requested: requested.toFixed(2) })
    : "—";
  $("#video-latency").textContent = formatDuration(stream?.inference?.processing_ms);
  $("#video-skipped").textContent = String(stream?.inference?.dropped_frames ?? 0);
  renderVideoMatches(stream?.faces ?? []);
  drawMonitorOverlay(stream);

  const monitor = selectedMonitor();
  if (!monitor?.enabled) {
    $("#video-status").textContent = t("Monitor is stopped.");
  } else if (stream?.last_error && !connected) {
    $("#video-status").textContent = t("RTSP stream unavailable. Reconnecting…");
  } else if (!connected) {
    $("#video-status").textContent = t("Connecting to RTSP camera…");
  } else {
    const timing = stream?.inference?.capacity_limited
      ? t("Inference-limited; stale frames are being skipped")
      : t("Running at the configured cadence");
    $("#video-status").textContent = t("{recognized} recognized · {unknown} not enrolled · {timing}", {
      recognized: Number(stream?.matched_faces ?? 0),
      unknown: Number(stream?.unknown_faces ?? 0),
      timing,
    });
    if (stream?.liveness_blocked_faces) {
      $("#video-status").textContent += ` · ${t("Liveness blocked {count}", { count: stream.liveness_blocked_faces })}`;
    }
  }
  if (monitor?.preview_enabled && connected) startMonitorPreview(monitor.id);
  else stopMonitorPreview();
}

function renderMonitorEvents() {
  const target = $("#monitor-events");
  target.replaceChildren();
  $("#monitor-event-count").textContent = String(state.monitorEvents.length);
  for (const event of [...state.monitorEvents].reverse()) {
    const row = element("div", { className: "monitor-event" });
    const eventLabel = monitorEventLabel(event.type);
    const type = element("span", { className: "badge", text: eventLabel });
    const summary = element("span");
    const identity = event.person?.name
      || event.person?.id
      || (event.type === "monitor_error" || event.type === "monitor_recovered"
        ? t(eventLabel)
        : t("Unknown person"));
    summary.append(
      element("strong", { text: identity }),
      element("small", {
        text: event.similarity === null || event.similarity === undefined
          ? event.track_id || event.error?.message || "—"
          : t("Similarity {score}", { score: formatCosine(event.similarity) }),
      }),
    );
    const occurred = event.occurred_at ? new Date(event.occurred_at) : null;
    const timestamp = element("time", {
      text: occurred && !Number.isNaN(occurred.getTime()) ? occurred.toLocaleTimeString() : "—",
    });
    row.append(type, summary, timestamp);
    target.append(row);
  }
}

async function pollMonitorEvents(monitorId) {
  const query = { limit: 100 };
  if (state.monitorEventCursor) query.cursor = state.monitorEventCursor;
  const payload = await client.monitorEvents(monitorId, query);
  if (monitorId !== state.selectedMonitorId) return;
  if (payload.stream_reset || payload.truncated) state.monitorEvents = [];
  state.monitorEvents.push(...(payload.events ?? []));
  state.monitorEvents = state.monitorEvents.slice(-1000);
  state.monitorEventCursor = payload.next_cursor || "";
  renderMonitorEvents();
}

async function pollMonitor(monitorId) {
  if (!monitorId || monitorId !== state.selectedMonitorId || state.route !== "video") return;
  try {
    const [statusPayload, monitorPayload] = await Promise.all([
      client.monitorState(monitorId),
      client.getMonitor(monitorId),
      pollMonitorEvents(monitorId),
    ]);
    if (monitorId !== state.selectedMonitorId) return;
    const refreshed = unwrap(monitorPayload, "monitor");
    state.monitors = state.monitors.map((item) => item.id === monitorId ? refreshed : item);
    renderMonitorList();
    renderSelectedMonitor();
    renderVideoStatus(unwrap(statusPayload, "state"));
  } catch (error) {
    if (monitorId !== state.selectedMonitorId) return;
    $("#video-status").textContent = describeError(error);
    if (error instanceof ApiError && error.status === 404) {
      await loadMonitors({ preserveSelection: false });
      return;
    }
  }
  if (monitorId === state.selectedMonitorId && state.route === "video") {
    state.monitorPollTimer = window.setTimeout(() => void pollMonitor(monitorId), 1000);
  }
}

function concatBytes(left, right) {
  const combined = new Uint8Array(left.length + right.length);
  combined.set(left);
  combined.set(right, left.length);
  return combined;
}

function headerEndIndex(buffer) {
  for (let index = 0; index <= buffer.length - 4; index += 1) {
    if (buffer[index] === 13 && buffer[index + 1] === 10 && buffer[index + 2] === 13 && buffer[index + 3] === 10) return index;
  }
  return -1;
}

async function consumeMjpeg(response, signal) {
  const reader = response.body?.getReader();
  if (!reader) throw new Error("The browser cannot read this preview stream.");
  let buffer = new Uint8Array();
  while (!signal.aborted) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer = concatBytes(buffer, value);
    while (buffer.length) {
      const headerEnd = headerEndIndex(buffer);
      if (headerEnd < 0) break;
      const header = new TextDecoder("ascii").decode(buffer.slice(0, headerEnd));
      const lengthMatch = header.match(/Content-Length:\s*(\d+)/i);
      if (!lengthMatch) {
        buffer = buffer.slice(headerEnd + 4);
        continue;
      }
      const contentLength = Number(lengthMatch[1]);
      const contentStart = headerEnd + 4;
      if (buffer.length < contentStart + contentLength) break;
      const jpeg = buffer.slice(contentStart, contentStart + contentLength);
      buffer = buffer.slice(Math.min(buffer.length, contentStart + contentLength + 2));
      const url = URL.createObjectURL(new Blob([jpeg], { type: "image/jpeg" }));
      const preview = $("#rtsp-preview");
      const previous = state.previewObjectUrl;
      state.previewObjectUrl = url;
      preview.src = url;
      preview.hidden = false;
      $("#rtsp-overlay").hidden = false;
      $("#video-idle").hidden = true;
      if (previous) URL.revokeObjectURL(previous);
    }
  }
}

function startMonitorPreview(monitorId) {
  if (state.previewAbortController || monitorId !== state.selectedMonitorId) return;
  const controller = new AbortController();
  state.previewAbortController = controller;
  void (async () => {
    try {
      const response = await client.monitorPreview(monitorId, { signal: controller.signal });
      await consumeMjpeg(response, controller.signal);
    } catch (error) {
      if (!controller.signal.aborted && monitorId === state.selectedMonitorId) {
        $("#video-status").textContent = describeError(error);
      }
    } finally {
      if (state.previewAbortController === controller) state.previewAbortController = null;
    }
  })();
}

function stopMonitorPreview() {
  state.previewAbortController?.abort();
  state.previewAbortController = null;
  const preview = $("#rtsp-preview");
  preview.removeAttribute("src");
  preview.hidden = true;
  $("#rtsp-overlay").hidden = true;
  const context = $("#rtsp-overlay").getContext("2d");
  context.clearRect(0, 0, $("#rtsp-overlay").width, $("#rtsp-overlay").height);
  if (state.previewObjectUrl) URL.revokeObjectURL(state.previewObjectUrl);
  state.previewObjectUrl = "";
  $("#video-idle").hidden = false;
}

function stopMonitorView() {
  if (state.monitorPollTimer !== null) window.clearTimeout(state.monitorPollTimer);
  state.monitorPollTimer = null;
  stopMonitorPreview();
}

function resetMonitorFormDefaults() {
  const form = $("#monitor-create-form");
  form.reset();
  form.elements.inference_fps.value = "2";
  form.elements.event_buffer_size.value = "1000";
  form.elements.confirm_frames.value = "3";
  form.elements.absence_timeout_seconds.value = "3";
  form.elements.cooldown_seconds.value = "10";
  form.elements.enabled.checked = true;
  form.elements.emit_unknown.checked = true;
  form.elements.preview_enabled.checked = false;
}

function renderMonitorFormMode() {
  const editing = Boolean(state.editingMonitorId);
  const form = $("#monitor-create-form");
  $("#monitor-form-eyebrow").textContent = t(editing ? "EDIT MONITOR" : "NEW MONITOR");
  $("#monitor-form-title").textContent = t(editing ? "Edit Monitor" : "Create Monitor");
  $("#monitor-form-submit").textContent = t(editing ? "Save changes" : "Create Monitor");
  $("#monitor-enabled-label").textContent = t(editing ? "Enabled" : "Start after creation");
  $("#monitor-enabled-help").textContent = t(
    editing ? "Run this Monitor on the server." : "Run independently of this browser.",
  );
  form.elements.id.readOnly = editing;
  form.elements.url.required = !editing;
  form.elements.url.placeholder = editing
    ? t("Leave blank to keep the current RTSP URL")
    : "rtsp://camera.example:8554/stream";
}

function openCreateMonitorForm() {
  state.editingMonitorId = "";
  resetMonitorFormDefaults();
  renderMonitorFormMode();
  const form = $("#monitor-create-form");
  form.hidden = false;
  form.elements.id.focus();
}

function openEditMonitorForm() {
  const monitor = selectedMonitor();
  if (!monitor) return;
  state.editingMonitorId = monitor.id;
  resetMonitorFormDefaults();
  const form = $("#monitor-create-form");
  form.elements.id.value = monitor.id;
  form.elements.name.value = monitor.name ?? "";
  form.elements.description.value = monitor.description ?? "";
  form.elements.url.value = "";
  form.elements.collection_id.value = monitor.collection_id;
  form.elements.inference_fps.value = String(monitor.inference_fps ?? 2);
  form.elements.match_threshold.value = monitor.match_threshold ?? "";
  form.elements.event_buffer_size.value = String(monitor.event_buffer_size ?? 1000);
  form.elements.confirm_frames.value = String(monitor.event_policy?.confirm_frames ?? 3);
  form.elements.absence_timeout_seconds.value = String(monitor.event_policy?.absence_timeout_seconds ?? 3);
  form.elements.cooldown_seconds.value = String(monitor.event_policy?.cooldown_seconds ?? 10);
  form.elements.enabled.checked = monitor.enabled === true;
  form.elements.emit_unknown.checked = monitor.event_policy?.emit_unknown !== false;
  form.elements.preview_enabled.checked = monitor.preview_enabled === true;
  renderMonitorFormMode();
  form.hidden = false;
  form.elements.name.focus();
}

function closeMonitorForm() {
  state.editingMonitorId = "";
  $("#monitor-create-form").hidden = true;
}

async function saveMonitor(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const editingMonitorId = state.editingMonitorId;
  const common = {
    name: form.elements.name.value.trim(),
    description: form.elements.description.value.trim(),
    enabled: form.elements.enabled.checked,
    collection_id: form.elements.collection_id.value,
    inference_fps: Number(form.elements.inference_fps.value),
    match_threshold: form.elements.match_threshold.value === ""
      ? null
      : Number(form.elements.match_threshold.value),
    event_buffer_size: Number(form.elements.event_buffer_size.value),
    event_policy: {
      confirm_frames: Number(form.elements.confirm_frames.value),
      absence_timeout_seconds: Number(form.elements.absence_timeout_seconds.value),
      cooldown_seconds: Number(form.elements.cooldown_seconds.value),
      emit_unknown: form.elements.emit_unknown.checked,
    },
    preview_enabled: form.elements.preview_enabled.checked,
  };
  const rtspUrl = form.elements.url.value.trim();
  if (editingMonitorId && rtspUrl) common.source = { type: "rtsp", url: rtspUrl };
  const payload = await guarded(
    $("button[type=submit]", form),
    () => editingMonitorId
      ? client.updateMonitor(editingMonitorId, common)
      : client.createMonitor({
          id: form.elements.id.value.trim(),
          name: common.name,
          description: common.description,
          enabled: common.enabled,
          url: rtspUrl,
          collectionId: common.collection_id,
          inferenceFps: common.inference_fps,
          matchThreshold: common.match_threshold,
          eventBufferSize: common.event_buffer_size,
          confirmFrames: common.event_policy.confirm_frames,
          absenceTimeoutSeconds: common.event_policy.absence_timeout_seconds,
          cooldownSeconds: common.event_policy.cooldown_seconds,
          emitUnknown: common.event_policy.emit_unknown,
          previewEnabled: common.preview_enabled,
        }),
    editingMonitorId ? "Monitor could not be updated" : "Monitor could not be created",
    editingMonitorId ? "Saving…" : "Creating…",
  );
  if (payload === undefined) return;
  const monitor = unwrap(payload, "monitor");
  toast(editingMonitorId ? "Monitor updated" : "Monitor created", monitor.id);
  closeMonitorForm();
  if (editingMonitorId) {
    state.monitors = state.monitors.map((item) => item.id === monitor.id ? monitor : item);
    if (!monitor.preview_enabled || !monitor.enabled) stopMonitorPreview();
    renderMonitorList();
    renderSelectedMonitor();
    void pollMonitor(monitor.id);
    return;
  }
  await loadMonitors({ preserveSelection: false });
  selectMonitor(monitor.id);
}

async function toggleMonitorEnabled() {
  const monitor = selectedMonitor();
  if (!monitor) return;
  const payload = await guarded(
    $("#monitor-toggle"),
    () => client.updateMonitor(monitor.id, { enabled: !monitor.enabled }),
    "Monitor could not be updated",
    monitor.enabled ? "Stopping…" : "Starting…",
  );
  if (payload === undefined) return;
  const updated = unwrap(payload, "monitor");
  state.monitors = state.monitors.map((item) => item.id === updated.id ? updated : item);
  if (!updated.enabled) stopMonitorPreview();
  renderMonitorList();
  renderSelectedMonitor();
  void pollMonitor(updated.id);
}

async function toggleMonitorPreview() {
  const monitor = selectedMonitor();
  if (!monitor) return;
  const payload = await guarded(
    $("#monitor-preview-toggle"),
    () => client.updateMonitor(monitor.id, { preview_enabled: !monitor.preview_enabled }),
    "Monitor preview could not be updated",
    "Saving…",
  );
  if (payload === undefined) return;
  const updated = unwrap(payload, "monitor");
  state.monitors = state.monitors.map((item) => item.id === updated.id ? updated : item);
  if (!updated.preview_enabled) stopMonitorPreview();
  renderMonitorList();
  renderSelectedMonitor();
}

async function deleteSelectedMonitor() {
  const monitor = selectedMonitor();
  if (!monitor || !window.confirm(t("Delete Monitor {id}?", { id: monitor.id }))) return;
  const result = await guarded(
    $("#monitor-delete"),
    () => client.deleteMonitor(monitor.id),
    "Monitor could not be deleted",
    "Deleting…",
  );
  if (result === undefined) return;
  toast("Monitor deleted", monitor.id);
  stopMonitorView();
  state.selectedMonitorId = "";
  state.monitorEvents = [];
  state.monitorEventCursor = "";
  await loadMonitors({ preserveSelection: false });
}

function bindEvents() {
  $$('[data-route]').forEach((control) => control.addEventListener("click", (event) => {
    if (control.tagName === "A") return;
    event.preventDefault();
    navigate(control.dataset.route);
  }));
  $("#menu-toggle").addEventListener("click", () => {
    const open = document.body.classList.toggle("menu-open");
    $("#menu-toggle").setAttribute("aria-expanded", String(open));
  });
  $("#refresh-page").addEventListener("click", () => void refreshRoute());
  $("#clear-errors").addEventListener("click", () => { state.errors = []; state.serverErrors = []; renderRecentErrors(); });

  const keyDialog = $("#api-key-dialog");
  $("#open-api-key").addEventListener("click", () => keyDialog.showModal());
  $("#api-key-form").addEventListener("submit", (event) => {
    event.preventDefault();
    client.setApiKey(event.currentTarget.elements.api_key.value);
    event.currentTarget.elements.api_key.value = "";
    keyDialog.close();
    updateAuthState();
    toast("API key configured", "It will be forgotten when this tab closes.");
    void refreshRoute();
  });
  $("#clear-api-key").addEventListener("click", () => {
    client.clearApiKey();
    $("#api-key-form").elements.api_key.value = "";
    keyDialog.close();
    updateAuthState();
    toast("API key cleared");
  });

  $("#show-create-collection").addEventListener("click", () => {
    updateCreateCollectionProfiles();
    const form = $("#collection-create-form");
    form.elements.save_face_crops.checked = newCollectionCropDefault();
    const detection = systemDetectionDefaults();
    form.elements.detector_input_sizes.value = formatDetectorInputSizes(detection.input_sizes);
    form.elements.detector_threshold.value = String(detection.threshold);
    form.elements.detector_nms_threshold.value = String(detection.nms_threshold);
    form.elements.single_face_selection.value = detection.single_face_selection;
    form.hidden = false;
    $("input", form).focus();
  });
  $("#collection-create-form .close-form").addEventListener("click", () => { $("#collection-create-form").hidden = true; });
  $("#collection-create-form").addEventListener("submit", (event) => void createCollection(event));
  $("#collection-filter").addEventListener("input", renderCollections);
  $("#collection-edit-form").addEventListener("submit", (event) => void updateCollection(event));

  $("#people-collection").addEventListener("change", (event) => { state.selectedCollection = event.target.value; state.selectedPerson = null; configureEmbeddingFields($("#person-create-form")); void loadPeople(); });
  $("#people-filter").addEventListener("input", renderPeople);
  $("#show-create-person").addEventListener("click", () => {
    if (!state.selectedCollection) { handleError(new Error("Select a collection first."), "Cannot register person"); return; }
    configureEmbeddingFields($("#person-create-form"));
    $("#person-create-form").hidden = false;
  });
  $("#person-create-form .close-form").addEventListener("click", () => { $("#person-create-form").hidden = true; });
  $("#person-create-form").elements.embedding_mode.addEventListener("change", (event) => configureEmbeddingFields(event.currentTarget.form));
  $("#person-create-form").addEventListener("submit", (event) => void createPerson(event));

  $("#detect-form").addEventListener("submit", (event) => void submitDetect(event));
  $("#compare-form").addEventListener("submit", (event) => void submitCompare(event));
  $("#search-form").addEventListener("submit", (event) => void submitSearch(event));
  $("#compare-threshold").addEventListener("input", (event) => { $("#compare-threshold-output").value = Number(event.target.value).toFixed(2); });

  $("#system-refresh").addEventListener("click", () => void loadSystem());
  $("#system-liveness-enable").addEventListener("click", () => void livenessManager.enable());
  $("#system-liveness-refresh").addEventListener("click", () => void livenessManager.refresh());
  $("#documentation-filter").addEventListener("input", filterDocumentationToc);
  window.addEventListener("insightface:localechange", () => {
    updateAuthState();
    renderRecentErrors();
    if (state.route === "collections") renderCollections();
    if (state.route === "people") renderPeople();
    if (state.route === "system") renderLivenessManagement();
    if (state.route === "video") {
      renderMonitorList();
      renderSelectedMonitor();
      renderMonitorEvents();
      if (state.videoStatus) renderVideoStatus(state.videoStatus);
      if (!$("#monitor-create-form").hidden) renderMonitorFormMode();
    }
    setText("#page-title", t(pageTitles[state.route]));
    document.title = `${t(pageTitles[state.route])} · InsightFace Server`;
    if (state.route === "help") {
      const anchor = state.helpPendingAnchor;
      state.helpPendingAnchor = null;
      void loadDocumentation(anchor);
    }
  });
  $("#show-create-monitor").addEventListener("click", openCreateMonitorForm);
  $("#monitor-create-form .close-form").addEventListener("click", closeMonitorForm);
  $("#monitor-create-form").addEventListener("submit", (event) => void saveMonitor(event));
  $("#refresh-monitors").addEventListener("click", () => void loadMonitors());
  $("#monitor-toggle").addEventListener("click", () => void toggleMonitorEnabled());
  $("#monitor-preview-toggle").addEventListener("click", () => void toggleMonitorPreview());
  $("#monitor-edit").addEventListener("click", openEditMonitorForm);
  $("#monitor-delete").addEventListener("click", () => void deleteSelectedMonitor());
  $("#rtsp-preview").addEventListener("load", () => { $("#video-idle").hidden = true; });
  $("#rtsp-preview").addEventListener("error", () => {
    if (state.selectedMonitorId) $("#video-status").textContent = t("Waiting for the first RTSP frame…");
  });
  window.addEventListener("beforeunload", stopMonitorView);
  window.addEventListener("beforeunload", releaseAllCropObjectUrls);

  $$('input[type="file"]').forEach(setupFileInput);
  setupImagePreview($("#detect-form input[name=image]"), $("#detect-canvas"), () => {
    $("#detect-results").replaceChildren();
    $("#detect-summary").textContent = t("Image ready. Press Detect faces to process it.");
  });
  setupImagePreview($("#compare-form input[name=source]"), $("#compare-source-canvas"), resetCompareVerdict);
  setupImagePreview($("#compare-form input[name=target]"), $("#compare-target-canvas"), resetCompareVerdict);
  setupImagePreview($("#search-form input[name=image]"), $("#search-canvas"), () => {
    $("#search-results").replaceChildren();
    $("#search-summary").textContent = t("Image ready. Press Search people to process it.");
  });
}

function resetCompareVerdict() {
  const verdict = $("#compare-verdict");
  verdict.className = "compare-verdict";
  $(".verdict-mark", verdict).textContent = "—";
  $("strong", verdict).textContent = t("Ready to compare");
  $("small", verdict).textContent = t("Similarity is not a probability.");
}

function updateAuthState() {
  const stateNode = $("#auth-state");
  stateNode.classList.toggle("configured", client.hasApiKey);
  stateNode.innerHTML = "";
  stateNode.append(element("span", { text: client.hasApiKey ? "●" : "○" }), document.createTextNode(` ${t(client.hasApiKey ? "API key configured" : "No API key")}`));
  setText("#system-auth", t(client.hasApiKey ? "Configured" : "Not configured"));
}

async function createCollection(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const button = $("button[type=submit]", form);
  let metadata;
  try { metadata = parseMetadata(form.elements.metadata.value); } catch (error) { handleError(error, "Invalid collection metadata"); return; }
  let detectorInputSizes;
  try { detectorInputSizes = parseDetectorInputSizes(form.elements.detector_input_sizes.value); } catch (error) { handleError(error, "Invalid detection profile"); return; }
  const payload = await guarded(button, () => client.createCollection({
    id: form.elements.id.value.trim(),
    name: form.elements.name.value.trim(),
    description: form.elements.description.value.trim(),
    threshold: Number(form.elements.threshold.value),
    metadata,
    save_face_crops: form.elements.save_face_crops.checked,
    detection: {
      input_sizes: detectorInputSizes,
      threshold: Number(form.elements.detector_threshold.value),
      nms_threshold: Number(form.elements.detector_nms_threshold.value),
      single_face_selection: form.elements.single_face_selection.value,
    },
    search: {
      profile: form.elements.search_profile.value,
      capacity_rows: Number(form.elements.capacity_rows.value),
      max_faces_per_person: Number(form.elements.max_faces_per_person.value),
      ...(form.elements.load_policy.value ? { load_policy: form.elements.load_policy.value } : {}),
    },
  }), "Collection could not be created", "Creating…");
  if (payload === undefined) return;
  const collection = unwrap(payload, "collection");
  toast("Collection created", collection.id ?? form.elements.id.value);
  form.reset();
  form.elements.threshold.value = "0.4";
  form.elements.search_profile.value = "fp32_v1";
  form.elements.capacity_rows.value = "100000";
  form.elements.max_faces_per_person.value = "20";
  form.elements.load_policy.value = "";
  form.elements.save_face_crops.checked = newCollectionCropDefault();
  const detection = systemDetectionDefaults();
  form.elements.detector_input_sizes.value = formatDetectorInputSizes(detection.input_sizes);
  form.elements.detector_threshold.value = String(detection.threshold);
  form.elements.detector_nms_threshold.value = String(detection.nms_threshold);
  form.elements.single_face_selection.value = detection.single_face_selection;
  form.elements.metadata.value = "{}";
  updateCreateCollectionProfiles();
  form.hidden = true;
  await loadCollections();
}

async function updateCollection(event) {
  event.preventDefault();
  const form = event.currentTarget;
  let metadata;
  try { metadata = parseMetadata(form.elements.metadata.value); } catch (error) { handleError(error, "Invalid collection metadata"); return; }
  let detectorInputSizes;
  try { detectorInputSizes = parseDetectorInputSizes(form.elements.detector_input_sizes.value); } catch (error) { handleError(error, "Invalid detection profile"); return; }
  const payload = await guarded($("button[type=submit]", form), () => client.updateCollection(form.elements.id.value, {
    name: form.elements.name.value.trim(),
    description: form.elements.description.value.trim(),
    threshold: Number(form.elements.threshold.value),
    metadata,
    save_face_crops: form.elements.save_face_crops.checked,
    detection: {
      input_sizes: detectorInputSizes,
      threshold: Number(form.elements.detector_threshold.value),
      nms_threshold: Number(form.elements.detector_nms_threshold.value),
      single_face_selection: form.elements.single_face_selection.value,
    },
    search: {
      capacity_rows: Number(form.elements.capacity_rows.value),
      max_faces_per_person: Number(form.elements.max_faces_per_person.value),
      load_policy: form.elements.load_policy.value,
    },
  }), "Collection could not be updated", "Saving…");
  if (payload === undefined) return;
  $("#collection-edit-dialog").close();
  toast("Collection updated", form.elements.id.value);
  await loadCollections();
}

async function createPerson(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const images = [...form.elements.images.files];
  let metadata;
  try { metadata = parseMetadata(form.elements.metadata.value); } catch (error) { handleError(error, "Invalid person metadata"); return; }
  let embeddingInput;
  try { embeddingInput = enrollmentEmbeddingInput(form, images); } catch (error) { handleError(error, "Invalid external embeddings"); return; }
  const button = $("button[type=submit]", form);
  setBusy(button, true, "Registering…");
  let payload;
  try {
    payload = await client.createPerson(state.selectedCollection, {
      id: form.elements.id.value.trim(),
      name: form.elements.name.value.trim(),
      externalId: form.elements.external_id.value.trim(),
      metadata,
      images,
      reviewMode: form.elements.review_mode.value,
      ...embeddingInput,
    });
  } catch (error) {
    const rejected = error instanceof ApiError ? error.details?.rejected_images : null;
    if (Array.isArray(rejected)) renderRejections($("#person-rejections"), rejected, images);
    handleError(error, "Person could not be registered");
    return;
  } finally {
    setBusy(button, false);
  }
  const faces = listItems(payload, ["faces", "face_samples"]);
  const rejected = payload.rejected_images ?? payload.rejected ?? [];
  renderRejections($("#person-rejections"), rejected, images);
  toast("Registration complete", t("{accepted} accepted, {rejected} rejected.", { accepted: faces.length, rejected: rejected.length }));
  form.elements.images.value = "";
  form.elements.external_embeddings.value = "";
  $(".file-summary", form).textContent = t("No files selected");
  await loadPeople();
  const person = unwrap(payload, "person");
  const listed = state.people.find((item) => item.id === person?.id);
  if (listed) await selectPerson(listed);
}

async function initialize() {
  initializeI18n();
  bindEvents();
  translateTree(document);
  updateAuthState();
  renderRecentErrors();
  try { await refreshHealth(); } catch { /* Health UI already shows unavailable. */ }
  const requestedRoute = window.location.hash.slice(1);
  const initialRoute = requestedRoute === "camera" ? "video" : requestedRoute;
  navigate(pageTitles[initialRoute] ? initialRoute : "dashboard");
}

void initialize();

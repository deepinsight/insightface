import { initializeI18n, t, translateTree } from "./i18n.mjs?v=0.3.1-r1";

const operationsRoot = document.querySelector("#operations");
const navigationRoot = document.querySelector("#operation-nav");
const filterInput = document.querySelector("#operation-filter");
const schemaRoot = document.querySelector("#schema-grid");
const methods = ["get", "post", "put", "patch", "delete"];
let specification;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = t(text);
  return node;
}

function safeId(method, path) {
  return `${method}-${path}`.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

function compactSchema(schema) {
  if (!schema) return {};
  const clone = structuredClone(schema);
  return clone;
}

function schemaLabel(schema) {
  if (!schema) return "unknown";
  if (schema.$ref) return schema.$ref.split("/").at(-1);
  if (schema.type === "array") return `array<${schemaLabel(schema.items)}>`;
  return schema.type || schema.format || "object";
}

function addParameterRows(container, parameters) {
  if (!parameters.length) return;
  const section = element("section", "operation-section");
  section.append(element("h3", "", "Parameters"));
  for (const parameter of parameters) {
    const row = element("div", "parameter");
    const name = element("div", "parameter-name", parameter.name);
    if (parameter.required) name.append(element("span", "required", "required"));
    const metadata = element("div", "parameter-meta");
    metadata.textContent = `${parameter.in} · ${schemaLabel(parameter.schema)}`;
    if (parameter.description) metadata.append(document.createElement("br"), document.createTextNode(t(parameter.description)));
    row.append(name, metadata);
    section.append(row);
  }
  container.append(section);
}

function addRequestBody(container, requestBody) {
  if (!requestBody) return;
  const section = element("section", "operation-section");
  const heading = element("h3", "", "Request body");
  if (requestBody.required) heading.append(element("span", "required", "required"));
  section.append(heading);
  for (const [contentType, media] of Object.entries(requestBody.content || {})) {
    section.append(element("p", "parameter-meta", contentType));
    const pre = element("pre", "code-block", JSON.stringify(compactSchema(media.schema), null, 2));
    section.append(pre);
  }
  container.append(section);
}

function addResponses(container, responses) {
  const section = element("section", "operation-section");
  section.append(element("h3", "", "Responses"));
  for (const [status, response] of Object.entries(responses || {})) {
    const row = element("div", "parameter");
    row.append(element("div", "parameter-name", status), element("div", "parameter-meta", response.description || "Response"));
    section.append(row);
  }
  container.append(section);
}

function renderOperation(method, path, operation, inheritedParameters) {
  const id = safeId(method, path);
  const details = element("details", "operation");
  details.id = id;
  details.dataset.search = `${method} ${path} ${operation.summary || ""} ${(operation.tags || []).join(" ")}`.toLowerCase();
  const summary = document.createElement("summary");
  summary.append(element("span", `method ${method}`, method.toUpperCase()), element("code", "operation-path", path), element("span", "operation-summary", operation.summary || operation.operationId || ""));
  details.append(summary);
  const body = element("div", "operation-body");
  if (operation.description) body.append(element("p", "operation-description", operation.description));
  addParameterRows(body, [...(inheritedParameters || []), ...(operation.parameters || [])]);
  addRequestBody(body, operation.requestBody);
  addResponses(body, operation.responses);
  details.append(body);
  operationsRoot.append(details);

  const link = document.createElement("a");
  link.href = `#${id}`;
  link.dataset.search = details.dataset.search;
  link.append(element("b", "", method.toUpperCase()), element("span", "", operation.summary || path));
  link.addEventListener("click", () => { details.open = true; });
  navigationRoot.append(link);
}

function renderSchemas(schemas) {
  schemaRoot.replaceChildren();
  const entries = Object.entries(schemas || {}).sort(([left], [right]) => left.localeCompare(right));
  document.querySelector("#schema-count").textContent = t("{count} schemas", { count: entries.length });
  for (const [name, schema] of entries) {
    const details = element("details", "schema-card");
    const summary = document.createElement("summary");
    summary.textContent = name;
    details.append(summary, element("pre", "", JSON.stringify(schema, null, 2)));
    schemaRoot.append(details);
  }
}

function renderDocument(specification) {
  operationsRoot.replaceChildren();
  navigationRoot.replaceChildren();
  document.querySelector("#api-description").textContent = specification.info?.description
    ? t(specification.info.description)
    : `${specification.info?.title || "InsightFace Server"} ${specification.info?.version || ""}`;
  const server = specification.servers?.[0]?.url;
  document.querySelector("#base-url").textContent = server || "/v1";
  for (const [path, pathItem] of Object.entries(specification.paths || {})) {
    for (const method of methods) {
      if (pathItem[method]) renderOperation(method, path, pathItem[method], pathItem.parameters);
    }
  }
  if (!operationsRoot.children.length) operationsRoot.append(element("div", "error", "The OpenAPI document contains no operations."));
  renderSchemas(specification.components?.schemas);
}

function applyFilter() {
  const query = filterInput.value.trim().toLowerCase();
  document.querySelectorAll(".operation").forEach((node) => { node.hidden = Boolean(query) && !node.dataset.search.includes(query); });
  navigationRoot.querySelectorAll("a").forEach((node) => { node.hidden = Boolean(query) && !node.dataset.search.includes(query); });
}

filterInput.addEventListener("input", applyFilter);
window.addEventListener("insightface:localechange", () => {
  translateTree(document);
  document.title = `${t("OpenAPI Schema")} · InsightFace Server`;
  if (specification) renderDocument(specification);
  applyFilter();
});

initializeI18n();
translateTree(document);
document.title = `${t("OpenAPI Schema")} · InsightFace Server`;

try {
  const response = await fetch("/openapi.json", { headers: { Accept: "application/json" } });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  specification = await response.json();
  renderDocument(specification);
} catch (error) {
  operationsRoot.replaceChildren(element("div", "error", t("Could not load OpenAPI document: {message}", { message: error.message })));
}

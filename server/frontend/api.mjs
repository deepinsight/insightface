export class ApiError extends Error {
  constructor({ code = "request_failed", message = "Request failed.", details = {}, status = 0, requestId = "" } = {}) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.details = details;
    this.status = status;
    this.requestId = requestId;
  }
}

export class ApiClient {
  #apiKey = "";

  constructor(baseUrl = "", { fetchFn = globalThis.fetch?.bind(globalThis) } = {}) {
    if (!fetchFn) throw new TypeError("A fetch implementation is required");
    this.baseUrl = String(baseUrl).replace(/\/$/, "");
    this.fetchFn = fetchFn;
  }

  setApiKey(value) {
    this.#apiKey = String(value ?? "").trim();
  }

  clearApiKey() {
    this.#apiKey = "";
  }

  get hasApiKey() {
    return Boolean(this.#apiKey);
  }

  async request(path, { method = "GET", body, query, signal, keepalive = false } = {}) {
    const url = new URL(`${this.baseUrl}${path}`, globalThis.location?.origin ?? "http://localhost");
    for (const [key, value] of Object.entries(query ?? {})) {
      if (value !== undefined && value !== null && value !== "") url.searchParams.set(key, String(value));
    }
    const headers = new Headers({ Accept: "application/json" });
    if (this.#apiKey) headers.set("Authorization", `Bearer ${this.#apiKey}`);

    let requestBody = body;
    if (body && !(body instanceof FormData)) {
      headers.set("Content-Type", "application/json");
      requestBody = JSON.stringify(body);
    }

    let response;
    try {
      response = await this.fetchFn(url, { method, headers, body: requestBody, signal, keepalive });
    } catch (error) {
      if (error?.name === "AbortError") {
        throw new ApiError({ code: "request_cancelled", message: "The request was cancelled." });
      }
      throw new ApiError({ code: "network_error", message: "Could not reach the InsightFace Server." });
    }

    const requestId = response.headers.get("x-request-id") ?? "";
    if (response.status === 204) return null;
    const text = await response.text();
    let payload = null;
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch {
        if (!response.ok) {
          throw new ApiError({ status: response.status, requestId, message: "The server returned an invalid error response." });
        }
        throw new ApiError({ status: response.status, requestId, code: "invalid_response", message: "The server returned invalid JSON." });
      }
    }

    if (!response.ok) {
      const error = payload?.error ?? payload ?? {};
      throw new ApiError({
        code: error.code ?? "request_failed",
        message: error.message ?? `Request failed with HTTP ${response.status}.`,
        details: error.details ?? {},
        status: response.status,
        requestId: payload?.request_id ?? requestId,
      });
    }
    if (payload && typeof payload === "object" && !payload.request_id && requestId) payload.request_id = requestId;
    return payload;
  }

  async requestBlob(path, { signal } = {}) {
    const url = new URL(`${this.baseUrl}${path}`, globalThis.location?.origin ?? "http://localhost");
    const headers = new Headers({ Accept: "image/jpeg" });
    if (this.#apiKey) headers.set("Authorization", `Bearer ${this.#apiKey}`);

    let response;
    try {
      response = await this.fetchFn(url, { method: "GET", headers, signal });
    } catch (error) {
      if (error?.name === "AbortError") {
        throw new ApiError({ code: "request_cancelled", message: "The request was cancelled." });
      }
      throw new ApiError({ code: "network_error", message: "Could not reach the InsightFace Server." });
    }

    const requestId = response.headers.get("x-request-id") ?? "";
    if (!response.ok) {
      let payload = null;
      try {
        const text = await response.text();
        payload = text ? JSON.parse(text) : null;
      } catch {
        payload = null;
      }
      const error = payload?.error ?? payload ?? {};
      throw new ApiError({
        code: error.code ?? "request_failed",
        message: error.message ?? `Request failed with HTTP ${response.status}.`,
        details: error.details ?? {},
        status: response.status,
        requestId: payload?.request_id ?? requestId,
      });
    }

    const mediaType = (response.headers.get("content-type") ?? "").split(";", 1)[0].trim().toLowerCase();
    const blob = await response.blob();
    if (mediaType !== "image/jpeg" || !blob.size) {
      throw new ApiError({
        code: "invalid_response",
        message: "The server returned an invalid face crop.",
        status: response.status,
        requestId,
      });
    }
    return blob;
  }

  async requestStream(path, { signal } = {}) {
    const url = new URL(`${this.baseUrl}${path}`, globalThis.location?.origin ?? "http://localhost");
    const headers = new Headers({ Accept: "multipart/x-mixed-replace" });
    if (this.#apiKey) headers.set("Authorization", `Bearer ${this.#apiKey}`);
    let response;
    try {
      response = await this.fetchFn(url, { method: "GET", headers, signal });
    } catch (error) {
      if (error?.name === "AbortError") throw error;
      throw new ApiError({ code: "network_error", message: "Could not reach the InsightFace Server." });
    }
    if (!response.ok) {
      const requestId = response.headers.get("x-request-id") ?? "";
      let payload = null;
      try {
        const text = await response.text();
        payload = text ? JSON.parse(text) : null;
      } catch {
        payload = null;
      }
      const error = payload?.error ?? payload ?? {};
      throw new ApiError({
        code: error.code ?? "request_failed",
        message: error.message ?? `Request failed with HTTP ${response.status}.`,
        details: error.details ?? {},
        status: response.status,
        requestId: payload?.request_id ?? requestId,
      });
    }
    return response;
  }

  health() { return this.request("/v1/health"); }
  system() { return this.request("/v1/system"); }
  models() { return this.request("/v1/models"); }
  liveness() { return this.request("/v1/addons/liveness"); }
  enableLiveness() { return this.request("/v1/addons/liveness/enable", { method: "POST", body: {} }); }

  detect(image, { maxFaces, collectionId } = {}) {
    const form = new FormData();
    form.append("image", image);
    if (maxFaces !== undefined && maxFaces !== "") form.append("max_faces", String(maxFaces));
    if (collectionId) form.append("collection_id", collectionId);
    return this.request("/v1/detect", { method: "POST", body: form });
  }

  compare(source, target, { threshold, collectionId } = {}) {
    const form = new FormData();
    form.append("source", source);
    form.append("target", target);
    if (threshold !== undefined && threshold !== "") form.append("threshold", String(threshold));
    if (collectionId) form.append("collection_id", collectionId);
    return this.request("/v1/compare", { method: "POST", body: form });
  }

  embeddings(image, { collectionId } = {}) {
    const form = new FormData();
    form.append("image", image);
    if (collectionId) form.append("collection_id", collectionId);
    return this.request("/v1/embeddings", { method: "POST", body: form });
  }

  createCollection(input) { return this.request("/v1/collections", { method: "POST", body: input }); }
  listCollections(query) { return this.request("/v1/collections", { query }); }
  getCollection(id) { return this.request(`/v1/collections/${encodeURIComponent(id)}`); }
  updateCollection(id, input) { return this.request(`/v1/collections/${encodeURIComponent(id)}`, { method: "PATCH", body: input }); }
  deleteCollection(id, force = false) {
    return this.request(`/v1/collections/${encodeURIComponent(id)}`, { method: "DELETE", query: { force } });
  }

  createPerson(collectionId, {
    id,
    name,
    externalId,
    metadata,
    images,
    reviewMode = "off",
    embeddingMode = "server",
    externalEmbeddings,
    embeddingContractId,
  }) {
    const form = new FormData();
    if (id) form.append("id", id);
    if (name) form.append("name", name);
    if (externalId) form.append("external_id", externalId);
    form.append("metadata", JSON.stringify(metadata ?? {}));
    form.append("review_mode", reviewMode);
    form.append("embedding_mode", embeddingMode);
    if (embeddingMode === "external_trusted") {
      form.append("external_embeddings", JSON.stringify(externalEmbeddings));
      form.append("embedding_contract_id", embeddingContractId ?? "");
    }
    for (const image of images ?? []) form.append("images", image);
    return this.request(`/v1/collections/${encodeURIComponent(collectionId)}/persons`, { method: "POST", body: form });
  }

  listPeople(collectionId, query) {
    return this.request(`/v1/collections/${encodeURIComponent(collectionId)}/persons`, { query });
  }
  getPerson(collectionId, personId) {
    return this.request(`/v1/collections/${encodeURIComponent(collectionId)}/persons/${encodeURIComponent(personId)}`);
  }
  updatePerson(collectionId, personId, input) {
    return this.request(`/v1/collections/${encodeURIComponent(collectionId)}/persons/${encodeURIComponent(personId)}`, { method: "PATCH", body: input });
  }
  deletePerson(collectionId, personId) {
    return this.request(`/v1/collections/${encodeURIComponent(collectionId)}/persons/${encodeURIComponent(personId)}`, { method: "DELETE" });
  }
  addFaces(collectionId, personId, images, {
    reviewMode = "off",
    embeddingMode = "server",
    externalEmbeddings,
    embeddingContractId,
  } = {}) {
    const form = new FormData();
    form.append("review_mode", reviewMode);
    form.append("embedding_mode", embeddingMode);
    if (embeddingMode === "external_trusted") {
      form.append("external_embeddings", JSON.stringify(externalEmbeddings));
      form.append("embedding_contract_id", embeddingContractId ?? "");
    }
    for (const image of images ?? []) form.append("images", image);
    return this.request(`/v1/collections/${encodeURIComponent(collectionId)}/persons/${encodeURIComponent(personId)}/faces`, { method: "POST", body: form });
  }
  listFaces(collectionId, personId, query) {
    return this.request(`/v1/collections/${encodeURIComponent(collectionId)}/persons/${encodeURIComponent(personId)}/faces`, { query });
  }
  deleteFace(collectionId, personId, faceId) {
    return this.request(`/v1/collections/${encodeURIComponent(collectionId)}/persons/${encodeURIComponent(personId)}/faces/${encodeURIComponent(faceId)}`, { method: "DELETE" });
  }
  getFaceCrop(collectionId, personId, faceId) {
    return this.requestBlob(`/v1/collections/${encodeURIComponent(collectionId)}/persons/${encodeURIComponent(personId)}/faces/${encodeURIComponent(faceId)}/image`);
  }

  search(collectionId, image, { limit = 5, threshold } = {}) {
    const form = new FormData();
    form.append("image", image);
    form.append("limit", String(limit));
    if (threshold !== undefined && threshold !== "") form.append("threshold", String(threshold));
    return this.request(`/v1/collections/${encodeURIComponent(collectionId)}/search`, { method: "POST", body: form });
  }

  createMonitor({
    id,
    name,
    description = "",
    enabled = true,
    url,
    collectionId,
    inferenceFps = 2,
    matchThreshold,
    eventBufferSize = 1000,
    confirmFrames = 3,
    absenceTimeoutSeconds = 3,
    cooldownSeconds = 10,
    emitUnknown = true,
    previewEnabled = false,
  } = {}) {
    const body = {
      id,
      name,
      description,
      enabled,
      source: { type: "rtsp", url },
      collection_id: collectionId,
      inference_fps: Number(inferenceFps),
      match_threshold: matchThreshold === undefined || matchThreshold === null || matchThreshold === ""
        ? null
        : Number(matchThreshold),
      event_buffer_size: Number(eventBufferSize),
      event_policy: {
        confirm_frames: Number(confirmFrames),
        absence_timeout_seconds: Number(absenceTimeoutSeconds),
        cooldown_seconds: Number(cooldownSeconds),
        emit_unknown: Boolean(emitUnknown),
      },
      preview_enabled: Boolean(previewEnabled),
    };
    return this.request("/v1/monitors", { method: "POST", body });
  }

  listMonitors(query) {
    return this.request("/v1/monitors", { query });
  }

  getMonitor(monitorId) {
    return this.request(`/v1/monitors/${encodeURIComponent(monitorId)}`);
  }

  updateMonitor(monitorId, input) {
    return this.request(`/v1/monitors/${encodeURIComponent(monitorId)}`, {
      method: "PATCH",
      body: input,
    });
  }

  deleteMonitor(monitorId) {
    return this.request(`/v1/monitors/${encodeURIComponent(monitorId)}`, {
      method: "DELETE",
    });
  }

  monitorState(monitorId) {
    return this.request(`/v1/monitors/${encodeURIComponent(monitorId)}/state`);
  }

  monitorEvents(monitorId, query) {
    return this.request(`/v1/monitors/${encodeURIComponent(monitorId)}/events`, { query });
  }

  monitorPreview(monitorId, { signal } = {}) {
    return this.requestStream(
      `/v1/monitors/${encodeURIComponent(monitorId)}/preview.mjpeg`,
      { signal },
    );
  }
}

/**
 * Browser-independent helpers shared by the UI and its Node tests.
 * Keep this module free of DOM globals so it can be imported by `node --test`.
 */

export const DEFAULT_THRESHOLD = 0.4;
export const SEARCH_PROFILES = Object.freeze([
  "fp32_v1",
  "fp16_v1",
  "bf16_v1",
  "int8_x736_v1",
  "int8_x1000_v1",
]);

/**
 * Older servers did not publish auth_enabled in their health response and
 * required authentication by default. Keep their API-key controls visible.
 */
export function authenticationEnabledFromHealth(health) {
  return health?.auth_enabled !== false;
}

/**
 * Return the public profiles reported by GET /v1/system. `null` means the
 * server did not expose capability information (for compatibility with an
 * older server); an empty array means it explicitly supports none of them.
 */
export function searchProfilesFromSystem(system) {
  const profiles = system?.search?.profiles;
  if (!Array.isArray(profiles)) return null;
  const reported = new Set(profiles.filter((profile) => typeof profile === "string"));
  return SEARCH_PROFILES.filter((profile) => reported.has(profile));
}

/**
 * Apply runtime capabilities to the create-Collection profile control.
 * Existing Collection profile fields are deliberately not passed here: a
 * Collection is pinned to its profile and its edit form only displays it.
 */
export function applySearchProfileAvailability(select, supportedProfiles) {
  if (!select) return "";
  const options = [...(select.options ?? [])];
  const reported = Array.isArray(supportedProfiles) ? new Set(supportedProfiles) : null;

  for (const option of options) {
    option.disabled = reported !== null && !reported.has(option.value);
  }

  const enabled = options.filter((option) => !option.disabled);
  if (!enabled.some((option) => option.value === select.value)) {
    select.value = enabled[0]?.value ?? "";
  }
  select.disabled = enabled.length === 0;
  return select.value;
}

export function clamp(value, minimum, maximum) {
  const number = Number(value);
  if (!Number.isFinite(number)) return minimum;
  return Math.min(maximum, Math.max(minimum, number));
}

export function normalizeThreshold(value, fallback = DEFAULT_THRESHOLD) {
  if (value === "" || value === null || value === undefined) return fallback;
  return clamp(value, 0, 1);
}

export function formatScore(value, digits = 3) {
  const score = Number(value);
  if (!Number.isFinite(score)) return "—";
  return score.toFixed(digits);
}

export function formatCosine(value, digits = 4) {
  const cosine = Number(value);
  if (!Number.isFinite(cosine)) return "—";
  return cosine.toFixed(digits);
}

export function formatDuration(value) {
  const duration = Number(value);
  if (!Number.isFinite(duration)) return "—";
  return `${duration.toFixed(duration < 10 ? 1 : 0)} ms`;
}

export function parseMetadata(value) {
  const text = String(value ?? "").trim();
  if (!text) return {};
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error("Metadata must be valid JSON.");
  }
  if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") {
    throw new Error("Metadata must be a JSON object.");
  }
  return parsed;
}

export function parseExternalEmbeddings(value, expectedCount, expectedDimension) {
  const text = String(value ?? "").trim();
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error("External embeddings must be valid JSON.");
  }
  if (!Array.isArray(parsed)) {
    throw new Error("External embeddings must be a JSON array of vectors.");
  }
  if (parsed.length !== Number(expectedCount)) {
    throw new Error(`Provide exactly one external embedding for each image (${parsed.length} != ${expectedCount}).`);
  }
  return parsed.map((vector, index) => {
    if (!Array.isArray(vector) || vector.length === 0) {
      throw new Error(`External embedding ${index + 1} must be a non-empty array.`);
    }
    if (expectedDimension && vector.length !== Number(expectedDimension)) {
      throw new Error(`External embedding ${index + 1} must contain ${expectedDimension} values.`);
    }
    const values = vector.map((value) => {
      if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new Error(`External embedding ${index + 1} contains a non-finite or non-numeric value.`);
      }
      return value;
    });
    const norm = Math.sqrt(values.reduce((sum, value) => sum + value * value, 0));
    if (!Number.isFinite(norm) || Math.abs(norm - 1) > 2e-4) {
      throw new Error(`External embedding ${index + 1} must be L2-normalized within 0.0002 of unit norm.`);
    }
    return values;
  });
}

export function listItems(payload, preferredKeys = []) {
  if (Array.isArray(payload)) return payload;
  if (!payload || typeof payload !== "object") return [];
  for (const key of [...preferredKeys, "items", "results"]) {
    if (Array.isArray(payload[key])) return payload[key];
  }
  return [];
}

export function facePixels(face, imageWidth, imageHeight) {
  const bbox = face?.bbox ?? face?.bounding_box ?? {};
  const pixels = bbox.pixels ?? bbox.pixel ?? {};
  if ([pixels.x, pixels.y, pixels.width, pixels.height].every(Number.isFinite)) {
    return {
      x: Number(pixels.x),
      y: Number(pixels.y),
      width: Number(pixels.width),
      height: Number(pixels.height),
    };
  }

  const normalized = bbox.normalized ?? bbox;
  const left = Number(normalized.left ?? normalized.x);
  const top = Number(normalized.top ?? normalized.y);
  const width = Number(normalized.width);
  const height = Number(normalized.height);
  if (![left, top, width, height].every(Number.isFinite)) return null;
  return {
    x: left * imageWidth,
    y: top * imageHeight,
    width: width * imageWidth,
    height: height * imageHeight,
  };
}

export function faceLandmarks(face) {
  const points = face?.landmarks;
  if (!Array.isArray(points)) return [];
  return points
    .map((point) => {
      if (Array.isArray(point) && point.length >= 2) {
        return { x: Number(point[0]), y: Number(point[1]) };
      }
      if (point && typeof point === "object") {
        return { x: Number(point.x), y: Number(point.y) };
      }
      return null;
    })
    .filter((point) => point && Number.isFinite(point.x) && Number.isFinite(point.y));
}

export function bestPersonMatches(samples, threshold, limit) {
  const floor = normalizeThreshold(threshold, 0);
  const count = Math.max(1, Math.floor(Number(limit) || 5));
  const byPerson = new Map();
  for (const sample of samples ?? []) {
    const personId = String(sample?.person?.id ?? sample?.person_id ?? "");
    const similarity = Number(sample?.similarity);
    if (!personId || !Number.isFinite(similarity) || similarity < floor) continue;
    const previous = byPerson.get(personId);
    if (!previous || similarity > Number(previous.similarity)) byPerson.set(personId, sample);
  }
  return [...byPerson.values()]
    .sort((left, right) => Number(right.similarity) - Number(left.similarity))
    .slice(0, count);
}

export function filterPeople(people, query) {
  const needle = String(query ?? "").trim().toLocaleLowerCase();
  if (!needle) return [...(people ?? [])];
  return (people ?? []).filter((person) =>
    [person.id, person.name, person.external_id]
      .filter(Boolean)
      .some((value) => String(value).toLocaleLowerCase().includes(needle)),
  );
}

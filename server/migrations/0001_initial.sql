CREATE TABLE collections (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    default_threshold REAL NOT NULL CHECK(default_threshold BETWEEN 0 AND 1),
    model_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    model_digest TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    preprocessing_version TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE persons (
    collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    id TEXT NOT NULL,
    name TEXT,
    external_id TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(collection_id, id),
    UNIQUE(collection_id, external_id)
);

CREATE TABLE face_samples (
    id TEXT PRIMARY KEY,
    collection_id TEXT NOT NULL,
    person_id TEXT NOT NULL,
    embedding BLOB NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    bounding_box_json TEXT NOT NULL,
    landmarks_json TEXT,
    detection_score REAL NOT NULL CHECK(detection_score BETWEEN 0 AND 1),
    quality_json TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    model_digest TEXT NOT NULL,
    preprocessing_version TEXT NOT NULL,
    crop_path TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(collection_id, person_id)
        REFERENCES persons(collection_id, id) ON DELETE CASCADE
);

CREATE INDEX face_samples_collection_person
    ON face_samples(collection_id, person_id, id);
CREATE INDEX persons_collection_external
    ON persons(collection_id, external_id);

CREATE TABLE api_keys (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    salt BLOB NOT NULL,
    digest BLOB NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

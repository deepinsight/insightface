CREATE TABLE monitors (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1
        CHECK(enabled IN (0, 1)),
    source_type TEXT NOT NULL DEFAULT 'rtsp'
        CHECK(source_type = 'rtsp'),
    source_url_ciphertext TEXT NOT NULL,
    collection_id TEXT NOT NULL
        REFERENCES collections(id) ON DELETE RESTRICT,
    inference_fps REAL NOT NULL DEFAULT 2.0
        CHECK(inference_fps BETWEEN 0.1 AND 30.0),
    match_threshold REAL
        CHECK(match_threshold IS NULL OR match_threshold BETWEEN 0.0 AND 1.0),
    event_buffer_size INTEGER NOT NULL DEFAULT 1000
        CHECK(event_buffer_size BETWEEN 10 AND 10000),
    confirm_frames INTEGER NOT NULL DEFAULT 3
        CHECK(confirm_frames BETWEEN 1 AND 100),
    absence_timeout_seconds REAL NOT NULL DEFAULT 3.0
        CHECK(absence_timeout_seconds BETWEEN 0.1 AND 300.0),
    cooldown_seconds REAL NOT NULL DEFAULT 10.0
        CHECK(cooldown_seconds BETWEEN 0.0 AND 3600.0),
    emit_unknown INTEGER NOT NULL DEFAULT 1
        CHECK(emit_unknown IN (0, 1)),
    preview_enabled INTEGER NOT NULL DEFAULT 0
        CHECK(preview_enabled IN (0, 1)),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX monitors_collection_id ON monitors(collection_id, id);
CREATE INDEX monitors_enabled ON monitors(enabled, id);

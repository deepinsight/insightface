ALTER TABLE collections ADD COLUMN search_profile TEXT NOT NULL DEFAULT 'fp32_v1'
    CHECK(search_profile IN ('fp32_v1', 'fp16_v1', 'bf16_v1', 'int8_x1000_v1'));
ALTER TABLE collections ADD COLUMN capacity_rows INTEGER NOT NULL DEFAULT 100000
    CHECK(capacity_rows BETWEEN 1 AND 9223372036854775807);
ALTER TABLE collections ADD COLUMN max_faces_per_person INTEGER NOT NULL DEFAULT 20
    CHECK(max_faces_per_person BETWEEN 1 AND 1000000);
ALTER TABLE collections ADD COLUMN load_policy TEXT NOT NULL DEFAULT 'lazy'
    CHECK(load_policy IN ('lazy', 'eager'));
ALTER TABLE collections ADD COLUMN search_revision INTEGER NOT NULL DEFAULT 0
    CHECK(search_revision >= 0);

-- Native search uses unsigned 64-bit identifiers. SQLite INTEGER is signed, so
-- generated identifiers intentionally stay in the positive signed 63-bit range.
-- AUTOINCREMENT makes an identifier stable and prevents its reuse after delete.
CREATE TABLE search_person_ids (
    numeric_id INTEGER PRIMARY KEY AUTOINCREMENT
        CHECK(numeric_id BETWEEN 1 AND 9223372036854775807),
    collection_id TEXT NOT NULL,
    person_id TEXT NOT NULL,
    UNIQUE(collection_id, person_id),
    UNIQUE(collection_id, numeric_id),
    FOREIGN KEY(collection_id, person_id)
        REFERENCES persons(collection_id, id) ON DELETE CASCADE
);

CREATE TABLE search_vector_ids (
    numeric_id INTEGER PRIMARY KEY AUTOINCREMENT
        CHECK(numeric_id BETWEEN 1 AND 9223372036854775807),
    collection_id TEXT NOT NULL,
    person_id TEXT NOT NULL,
    person_numeric_id INTEGER NOT NULL,
    face_id TEXT NOT NULL UNIQUE,
    UNIQUE(collection_id, numeric_id),
    FOREIGN KEY(collection_id, person_id)
        REFERENCES persons(collection_id, id) ON DELETE CASCADE,
    FOREIGN KEY(collection_id, person_numeric_id)
        REFERENCES search_person_ids(collection_id, numeric_id) ON DELETE CASCADE,
    FOREIGN KEY(face_id)
        REFERENCES face_samples(id) ON DELETE CASCADE
);

CREATE INDEX search_vector_ids_collection_person
    ON search_vector_ids(collection_id, person_numeric_id, numeric_id);

-- Backfill stable IDs for databases created before native search support.
INSERT INTO search_person_ids(collection_id, person_id)
SELECT collection_id, id FROM persons ORDER BY collection_id, id;

INSERT INTO search_vector_ids(
    collection_id, person_id, person_numeric_id, face_id
)
SELECT f.collection_id, f.person_id, p.numeric_id, f.id
FROM face_samples f
JOIN search_person_ids p
  ON p.collection_id=f.collection_id AND p.person_id=f.person_id
ORDER BY f.collection_id, f.person_id, f.id;

-- This pending-only durable outbox is deliberately not linked by foreign keys.
-- Delete events must survive deletion of their Collection, Person, FaceSample
-- and ID mappings until the in-memory index acknowledges their global seq.
CREATE TABLE search_changes (
    seq INTEGER PRIMARY KEY AUTOINCREMENT
        CHECK(seq BETWEEN 1 AND 9223372036854775807),
    collection_id TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK(revision >= 1),
    operation TEXT NOT NULL CHECK(operation IN ('add', 'delete', 'clear')),
    vector_id INTEGER,
    person_numeric_id INTEGER,
    face_id TEXT,
    created_at TEXT NOT NULL,
    CHECK(
        (operation IN ('add', 'delete')
          AND vector_id IS NOT NULL
          AND person_numeric_id IS NOT NULL
          AND face_id IS NOT NULL)
        OR
        (operation='clear'
          AND vector_id IS NULL
          AND person_numeric_id IS NULL
          AND face_id IS NULL)
    )
);

CREATE INDEX search_changes_collection_seq
    ON search_changes(collection_id, seq);
CREATE INDEX search_changes_collection_revision
    ON search_changes(collection_id, revision, seq);

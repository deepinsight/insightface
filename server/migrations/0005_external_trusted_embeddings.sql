ALTER TABLE face_samples
    ADD COLUMN embedding_source TEXT NOT NULL DEFAULT 'server'
    CHECK(embedding_source IN ('server', 'external_trusted'));

ALTER TABLE face_samples ADD COLUMN embedding_contract_id TEXT;

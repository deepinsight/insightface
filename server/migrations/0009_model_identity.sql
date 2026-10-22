-- Preserve existing externally supplied embedding contracts before removing
-- the obsolete model version. New Collections use a version-free V2 contract.
ALTER TABLE collections ADD COLUMN embedding_contract_id TEXT NOT NULL DEFAULT '';

UPDATE collections SET embedding_contract_id = legacy_embedding_contract_id(
    model_id, model_version, model_digest, embedding_dimension, preprocessing_version
);

ALTER TABLE collections DROP COLUMN model_version;
ALTER TABLE face_samples DROP COLUMN model_version;

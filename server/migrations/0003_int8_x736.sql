-- Keep the profile string as the durable quantization contract. SQLite cannot
-- widen an existing CHECK constraint in place, so replace only this column;
-- all Collection, Person, FaceSample, and stable search-ID rows stay intact.
ALTER TABLE collections RENAME COLUMN search_profile TO search_profile_legacy;

ALTER TABLE collections ADD COLUMN search_profile TEXT NOT NULL DEFAULT 'fp32_v1'
    CHECK(search_profile IN (
        'fp32_v1',
        'fp16_v1',
        'bf16_v1',
        'int8_x1000_v1',
        'int8_x736_v1'
    ));

UPDATE collections SET search_profile = search_profile_legacy;

ALTER TABLE collections DROP COLUMN search_profile_legacy;

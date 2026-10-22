ALTER TABLE collections
    ADD COLUMN save_face_crops INTEGER NOT NULL DEFAULT 0
    CHECK(save_face_crops IN (0, 1));

ALTER TABLE face_samples ADD COLUMN crop_image BLOB;
ALTER TABLE face_samples ADD COLUMN crop_media_type TEXT;

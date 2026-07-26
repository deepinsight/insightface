ALTER TABLE collections
    ADD COLUMN detector_input_sizes_json TEXT NOT NULL
    DEFAULT '[[96,96],[512,512]]';

ALTER TABLE collections
    ADD COLUMN detector_threshold REAL NOT NULL DEFAULT 0.50
    CHECK(detector_threshold BETWEEN 0 AND 1);

ALTER TABLE collections
    ADD COLUMN detector_nms_threshold REAL NOT NULL DEFAULT 0.40
    CHECK(detector_nms_threshold BETWEEN 0 AND 1);

ALTER TABLE collections
    ADD COLUMN single_face_selection TEXT NOT NULL DEFAULT 'largest'
    CHECK(single_face_selection IN ('largest', 'center_largest'));

ALTER TABLE collections
    ADD COLUMN detection_revision INTEGER NOT NULL DEFAULT 1
    CHECK(detection_revision >= 1);

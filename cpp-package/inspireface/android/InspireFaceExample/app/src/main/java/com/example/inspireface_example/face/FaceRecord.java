package com.example.inspireface_example.face;

/** Local metadata paired with one FeatureHub identity. */
public final class FaceRecord {
    public final long id;
    public final String name;
    public final String cropPath;
    public final long updatedAt;

    public FaceRecord(long id, String name, String cropPath, long updatedAt) {
        this.id = id;
        this.name = name;
        this.cropPath = cropPath;
        this.updatedAt = updatedAt;
    }
}

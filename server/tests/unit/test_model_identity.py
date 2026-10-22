from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from insightface_server.storage import Database, Repository


def test_model_version_migration_preserves_contracts_and_registered_data(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[2] / "migrations"
    old_migrations = tmp_path / "old-migrations"
    old_migrations.mkdir()
    for migration in source.glob("*.sql"):
        if migration.name < "0009":
            shutil.copy(migration, old_migrations)
    path = tmp_path / "legacy.db"
    old = Database(path, old_migrations)
    old.initialize()
    contract = "ifsemb-v1-sha256:0473aa8e9422b084c939259ce447572a82ff9d23dc3a341bd22cf4806b4494b5"
    vector = np.asarray([0.5, -0.5, 0.5, -0.5], dtype=np.float32)
    liveness = {"status": "ok", "is_live": True, "live_score": 0.98}
    with old.write() as connection:
        connection.execute(
            """INSERT INTO collections(
                id,name,default_threshold,model_id,model_version,model_digest,
                embedding_dimension,preprocessing_version,created_at,updated_at
            ) VALUES('existing','Existing',0.4,'recognition','1',?,4,'1','before','before')""",
            ("a" * 64,),
        )
        connection.execute(
            """INSERT INTO persons(collection_id,id,name,created_at,updated_at)
            VALUES('existing','alice','Alice','before','before')"""
        )
        connection.execute(
            """INSERT INTO face_samples(
                id,collection_id,person_id,embedding,embedding_dimension,
                bounding_box_json,detection_score,quality_json,model_id,model_version,
                model_digest,preprocessing_version,created_at,embedding_source,
                embedding_contract_id,liveness_json
            ) VALUES('sample','existing','alice',?,4,'{}',0.99,'{}','recognition',
                '1',?,'1','before','external_trusted',?,?)""",
            (vector.tobytes(), "a" * 64, contract, json.dumps(liveness)),
        )
        connection.execute(
            "INSERT INTO search_person_ids VALUES(71,'existing','alice')"
        )
        connection.execute(
            "INSERT INTO search_vector_ids VALUES(99,'existing','alice',71,'sample')"
        )

    database = Database(path, source)
    database.initialize()
    database.initialize()
    repository = Repository(database)
    collection = repository.get_collection("existing")
    face = repository.get_face("existing", "alice", "sample")
    assert collection is not None and face is not None
    assert collection["embedding_contract_id"] == contract
    assert face["embedding_contract_id"] == contract
    assert face["embedding_source"] == "external_trusted"
    assert face["liveness"] == liveness
    np.testing.assert_array_equal(face["embedding"], vector)
    assert "model_version" not in collection and "model_version" not in face
    indexed = next(repository.iter_index_faces("existing", batch_size=10)).faces[0]
    assert (indexed.vector_id, indexed.person_numeric_id, indexed.face_id) == (99, 71, "sample")
    assert database.status()["migration_count"] == 9
    with database.read() as connection:
        assert not connection.execute("PRAGMA foreign_key_check").fetchall()
        for table in ("collections", "face_samples"):
            assert "model_version" not in {
                row[1] for row in connection.execute(f"PRAGMA table_info({table})")
            }
    new = repository.create_collection({
        "id": "new", "name": "New", "default_threshold": 0.4,
        "model_id": "recognition", "model_digest": "a" * 64,
        "embedding_dimension": 4, "preprocessing_version": "1",
    })
    assert new["embedding_contract_id"].startswith("ifsemb-v2-sha256:")
    assert repository.get_collection("existing")["embedding_contract_id"] == contract

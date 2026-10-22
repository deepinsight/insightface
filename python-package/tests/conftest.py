"""Opt-in synthetic model packages for person SDK and GUI contract tests."""
import hashlib
import json

import pytest


def _synthetic_manifest(name='synthetic_person', body_size=640):
    """Minimal metadata, independent of release assets and runtime node names."""
    tasks = {
        'person_detection': dict(input_size=[body_size, body_size]),
        'person_reid': dict(input_size=[256, 128]),
        'detection': dict(preprocessing=dict(mean=0, std=1)),
        'recognition': dict(input_size=[112, 112], preprocessing=dict(mean=0, std=1)),
    }
    for task, descriptor in tasks.items():
        filename = task + '.onnx'
        descriptor.update(file=filename, sha256=hashlib.sha256(filename.encode()).hexdigest())
    return dict(manifest_version=2, model_id=name, license='MODEL.LICENSE', tasks=tasks)


def _write_synthetic_package(root, document):
    folder = root / 'models' / document['model_id']
    folder.mkdir(parents=True)
    for descriptor in document['tasks'].values():
        # Deliberately not ONNX data: these tests must never run real inference.
        (folder / descriptor['file']).write_bytes(descriptor['file'].encode())
    (folder / 'manifest.json').write_text(json.dumps(document))
    return folder


@pytest.fixture
def synthetic_person_manifest():
    return _synthetic_manifest


@pytest.fixture
def synthetic_person_package():
    return _write_synthetic_package

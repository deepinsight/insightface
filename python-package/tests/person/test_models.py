"""Deterministic model contracts; real model smoke coverage is separate."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from insightface.model_zoo import person_detection, person_reid, person_package
from insightface.model_zoo.person_package import (
    _check_nodes, _check_warmup, create_session, load_person_package,
    model_fingerprint,
)
from insightface.model_zoo.package_manifest import load_model_package


def runtime_metadata(task):
    """Fake model metadata deliberately lives outside the package manifest."""
    def node(name, shape):
        return dict(name=name, shape=shape)
    if task == 'person_detection':
        inputs = [node('image', [1, 3, None, None]), node('scale_factor', [1, 2])]
        outputs = [node('nms_pre_boxes', [1, None, 4]), node('nms_pre_scores', [1, 1, None])]
    elif task == 'person_reid':
        inputs, outputs = [node('data', [1, 3, 256, 128])], [node('reid_embedding', [1, 256])]
    elif task == 'detection':
        inputs = [node('face_image', [1, 3, None, None])]
        outputs = [node(f'{kind}_{stride}', [None, width])
                   for kind, width in (('score', 1), ('box', 4), ('landmarks', 10))
                   for stride in (8, 16, 32)]
    else:
        inputs, outputs = [node('face_crop', [1, 3, 112, 112])], [node('embedding', [1, 512])]
    return dict(task=task, inputs=inputs, outputs=outputs)


def metadata_session(metadata):
    def nodes(kind):
        return [SimpleNamespace(name=node['name'], shape=node['shape'], type=node.get('type', 'tensor(float)'))
                for node in metadata[kind]]
    return SimpleNamespace(get_inputs=lambda: nodes('inputs'), get_outputs=lambda: nodes('outputs'))


@pytest.fixture
def manifest(synthetic_person_manifest):
    return synthetic_person_manifest()


@pytest.fixture
def package(tmp_path, manifest, synthetic_person_package):
    folder = synthetic_person_package(tmp_path, manifest)
    return tmp_path, folder, manifest


def fake_detector(monkeypatch, descriptor, boxes, scores, input_dtype='float32'):
    model = SimpleNamespace(run=lambda names, feed: [boxes, scores])
    def create(descriptor, *args):
        descriptor['input_dtype'] = input_dtype
        return model, {}
    monkeypatch.setattr(person_detection, 'create_session', create)
    return person_detection.PersonDetection(descriptor, ["CPUExecutionProvider"])


def fake_reid(monkeypatch, descriptor, input_dtype='float32'):
    model = SimpleNamespace(run=lambda *args: [np.ones((1, 256), np.float32)])
    def create(descriptor, *args):
        descriptor['input_dtype'] = input_dtype
        return model, {}
    monkeypatch.setattr(person_reid, 'create_session', create)
    return person_reid.PersonReID(descriptor, ['CPUExecutionProvider'])


def test_minimal_model_package_normalizes_defaults_without_model_sessions(package, monkeypatch):
    import onnxruntime
    monkeypatch.setattr(onnxruntime, 'InferenceSession', lambda *a, **kw: pytest.fail('session'))
    root, folder, _ = package
    parsed = load_person_package(folder.name, root)
    assert parsed["manifest_version"] == 2
    assert 'person_schema_version' not in parsed
    assert Path(parsed["tasks"]["person_reid"]["path"]) == folder / "person_reid.onnx"
    assert parsed['tasks']['recognition']['preprocessing'] == dict(mean=0., std=1.)
    assert parsed['tasks']['recognition']['input_size'] == [112, 112]
    assert parsed['tasks']['recognition']['embedding_dimension'] == 512
    assert parsed['tasks']['person_reid']['input_size'] == [256, 128]
    assert parsed['tasks']['person_detection']['input_size'] == [640, 640]
    assert parsed['tasks']['detection']['input_size'] == [640, 640]
    for task in ('person_detection', 'person_reid'):
        assert parsed['tasks'][task]['preprocessing'] == dict(mean=[0., 0., 0.], std=[1., 1., 1.], scale=1.)
    assert all('adapter' not in task for task in parsed['tasks'].values())
    assert all('inputs' not in task and 'outputs' not in task for task in parsed['tasks'].values())


@pytest.mark.parametrize("mutate", [
    lambda doc: doc.update(manifest_version=True),
    lambda doc: doc["tasks"]["person_detection"].pop('file'),
    lambda doc: doc["tasks"]["person_reid"].update(file="../outside.onnx"),
    lambda doc: doc["tasks"]["person_reid"].update(sha256="not-a-hash"),
    lambda doc: doc["tasks"]["person_reid"].update(preprocessing=dict(color='BGR')),
    lambda doc: doc["tasks"]["recognition"].update(embedding_dimension=True),
    lambda doc: doc["tasks"]["recognition"].update(embedding_dimension=0),
    lambda doc: doc.update(display_name=''),
    lambda doc: doc['tasks']['detection'].update(preprocessing_version=''),
    lambda doc: doc["tasks"]["person_detection"].update(input_size=[0, 640]),
    lambda doc: doc['tasks'].pop('recognition'),
    lambda doc: doc['tasks'].update(verification={}),
])
def test_manifest_rejects_incompatible_or_unsafe_contract(package, mutate):
    root, folder, document = package
    mutate(document)
    (folder / "manifest.json").write_text(json.dumps(document))
    with pytest.raises((ValueError, TypeError)):
        load_person_package(folder.name, root)


@pytest.mark.parametrize('task', ['person_detection', 'person_reid', 'detection', 'recognition'])
def test_omitted_input_size_uses_task_default_without_rewriting_manifest(package, task):
    root, folder, document = package
    document['tasks'][task].pop('input_size', None)
    path = folder / 'manifest.json'
    path.write_text(json.dumps(document))
    original = path.read_bytes()
    expected = {'person_detection': [640, 640], 'person_reid': [256, 128],
                'detection': [640, 640], 'recognition': [112, 112]}
    assert load_person_package(folder.name, root)['tasks'][task]['input_size'] == expected[task]
    assert path.read_bytes() == original


@pytest.mark.parametrize('task', ['detection', 'recognition'])
def test_face_preprocessing_defaults_match_generic_v2(package, task):
    root, folder, document = package
    document['tasks'][task].pop('preprocessing')
    (folder / 'manifest.json').write_text(json.dumps(document))
    generic = load_model_package(folder).task(task).as_config()
    assert load_person_package(folder.name, root)['tasks'][task]['preprocessing'] == generic['preprocessing']


def test_unknown_root_tasks_and_task_metadata_are_ignored_like_v2(package):
    root, folder, document = package
    original = load_person_package(folder.name, root)
    document.update(person_schema_version=99, future_root={'anything': True})
    document['tasks']['future_task'] = 'opaque future value'
    document['tasks']['verification'] = {'file': 'unused_verifier.onnx'}
    for descriptor in list(document['tasks'].values())[:4]:
        descriptor.update(adapter='ignored', original_name='ignored',
                          inputs='ignored', outputs='ignored', future_metadata=[1, 2, 3])
    (folder / 'manifest.json').write_text(json.dumps(document))
    assert load_person_package(folder.name, root) == original


@pytest.mark.parametrize('value', [[320, 320], [31, 640], None, 'ignored', {'unknown': True}])
def test_detection_manifest_input_size_is_ignored_like_generic_v2(package, value):
    root, folder, document = package
    document['tasks']['detection']['input_size'] = value
    (folder / 'manifest.json').write_text(json.dumps(document))
    assert 'input_size' not in load_model_package(folder).task('detection').as_config()
    assert load_person_package(folder.name, root)['tasks']['detection']['input_size'] == [640, 640]


def test_generic_v2_identity_and_face_metadata_are_reused(package):
    root, folder, document = package
    document['model_id'] = 'different_model_id'
    for task in ('detection', 'recognition'):
        document['tasks'][task] = {'file': document['tasks'][task]['file']}
    (folder / 'manifest.json').write_text(json.dumps(document))
    generic = load_model_package(folder)
    parsed = load_person_package(folder.name, root)
    assert parsed['model_id'] == generic.model_id == 'different_model_id'
    assert parsed['display_name'] == generic.display_name == 'different_model_id'
    for task in ('detection', 'recognition'):
        assert all(parsed['tasks'][task][key] == value
                   for key, value in generic.task(task).as_config().items())


@pytest.mark.parametrize('task', ['person_detection', 'person_reid', 'detection', 'recognition'])
def test_sha256_is_optional_but_actual_model_identity_is_retained(package, task):
    root, folder, document = package
    pinned = load_person_package(folder.name, root)['tasks'][task]
    document['tasks'][task].pop('sha256')
    path = folder / 'manifest.json'
    path.write_text(json.dumps(document))
    original = path.read_bytes()
    unpinned = load_person_package(folder.name, root)['tasks'][task]
    assert unpinned['sha256'] == hashlib.sha256(Path(unpinned['path']).read_bytes()).hexdigest()
    assert model_fingerprint(unpinned) == model_fingerprint(pinned)
    assert path.read_bytes() == original


@pytest.mark.parametrize('task', ['person_reid', 'recognition'])
def test_unpinned_weight_change_cannot_reuse_features_from_previous_model(package, task):
    from insightface.app import PersonAnalysis
    from insightface.app.person import Face, Person

    root, folder, document = package
    document['tasks'][task].pop('sha256')
    (folder / 'manifest.json').write_text(json.dumps(document))
    before = load_person_package(folder.name, root)['tasks'][task]
    Path(before['path']).write_bytes(b'other model weights')
    after = load_person_package(folder.name, root)['tasks'][task]
    previous_id, current_id = model_fingerprint(before), model_fingerprint(after)
    assert current_id != previous_id
    app = PersonAnalysis(root=root)
    if task == 'recognition':
        observation = Person(face=Face(np.array([0, 0, 10, 10]), 1., embedding=np.ones(512, np.float32)))
        observation._face_model_id, app.face_model_id = previous_id, current_id
    else:
        observation = Person(reid_feature=np.ones(256, np.float32))
        observation._reid_model_id, app.reid_model_id = previous_id, current_id
    with pytest.raises(ValueError, match='feature model mismatch'):
        app._validate_observation(observation)


def test_manifest_rejects_weight_corruption_before_session(package, monkeypatch):
    root, folder, _ = package
    (folder / "person_detection.onnx").write_bytes(b"corrupted")
    import onnxruntime
    monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *a, **kw: pytest.fail("session must not be created"))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_person_package(folder.name, root)


def test_fingerprint_tracks_feature_semantics_not_location_or_notes(package):
    root, folder, _ = package
    tasks = load_person_package(folder.name, root)['tasks']
    for task in ('person_reid', 'recognition'):
        descriptor = tasks[task]
        original = model_fingerprint(descriptor)
        relocated = dict(descriptor, path='/some/other/location', file='renamed.onnx',
                         original_name='origin', model_version='notes', display_name='Person')
        assert model_fingerprint(relocated) == original
        assert model_fingerprint(dict(descriptor, sha256='0' * 64)) != original
    face = tasks['recognition']
    assert model_fingerprint(dict(face, preprocessing=dict(mean=1, std=2))) != model_fingerprint(face)
    assert model_fingerprint(dict(face, preprocessing_version='different-processing')) != model_fingerprint(face)


def test_raccoon_v2_face_fields_and_display_name_are_supported(package):
    root, folder, document = package
    document['display_name'] = 'Synthetic Cheetah'
    document['tasks']['recognition'].update(
        preprocessing=dict(mean=127.5, std=127.5),
        preprocessing_version='insightface-arcface-1', embedding_dimension=512)
    document['tasks']['detection']['preprocessing_version'] = 'insightface-scrfd-1'
    (folder / 'manifest.json').write_text(json.dumps(document))
    parsed = load_person_package(folder.name, root)
    assert parsed['display_name'] == 'Synthetic Cheetah'
    for key, value in document['tasks']['recognition'].items():
        assert parsed['tasks']['recognition'][key] == value


def test_omitted_and_explicit_v2_face_versions_have_same_fingerprint(package):
    root, folder, document = package
    implicit = load_person_package(folder.name, root)['tasks']
    for task in ('detection', 'recognition'):
        document['tasks'][task]['preprocessing_version'] = implicit[task]['preprocessing_version']
    document['tasks']['recognition']['embedding_dimension'] = 512
    (folder / 'manifest.json').write_text(json.dumps(document))
    explicit = load_person_package(folder.name, root)['tasks']
    for task in ('detection', 'recognition'):
        assert model_fingerprint(implicit[task]) == model_fingerprint(explicit[task])


def test_omitted_body_normalization_and_explicit_neutral_rules_have_same_semantics(package):
    root, folder, document = package
    implicit = load_person_package(folder.name, root)['tasks']
    for task, descriptor in document['tasks'].items():
        descriptor['input_size'] = implicit[task]['input_size']
        descriptor['preprocessing'] = implicit[task]['preprocessing']
    (folder / 'manifest.json').write_text(json.dumps(document))
    explicit = load_person_package(folder.name, root)['tasks']
    for task in implicit:
        assert explicit[task]['input_size'] == implicit[task]['input_size']
        assert explicit[task]['preprocessing'] == implicit[task]['preprocessing']
        assert model_fingerprint(explicit[task]) == model_fingerprint(implicit[task])


@pytest.mark.parametrize('task', ['person_detection', 'person_reid'])
def test_scalar_and_repeated_rgb_rules_have_the_same_fingerprint(package, task):
    root, folder, document = package
    document['tasks'][task]['preprocessing'] = dict(mean=10, std=2, scale=.5)
    (folder / 'manifest.json').write_text(json.dumps(document))
    scalar = load_person_package(folder.name, root)['tasks'][task]
    document['tasks'][task]['preprocessing'] = dict(mean=[10., 10., 10.], std=[2., 2., 2.], scale=.5)
    (folder / 'manifest.json').write_text(json.dumps(document))
    vector = load_person_package(folder.name, root)['tasks'][task]
    assert scalar['preprocessing'] == vector['preprocessing']
    assert model_fingerprint(scalar) == model_fingerprint(vector)


@pytest.mark.parametrize('task', ['person_detection', 'person_reid'])
@pytest.mark.parametrize('preprocessing', [
    {'mean': True}, {'mean': 'zero'}, {'mean': [0, 0]}, {'mean': [0, None, 0]},
    {'mean': float('nan')}, {'mean': float('inf')},
    {'std': 0}, {'std': -1}, {'std': True}, {'std': [1, 1, 0]},
    {'scale': 0}, {'scale': -1}, {'scale': [1, 1, 1]}, {'scale': float('inf')},
    {'padding': .2},
])
def test_invalid_body_preprocessing_rejected_without_session(package, task, preprocessing, monkeypatch):
    import onnxruntime
    root, folder, document = package
    document['tasks'][task]['preprocessing'] = preprocessing
    (folder / 'manifest.json').write_text(json.dumps(document))
    monkeypatch.setattr(onnxruntime, 'InferenceSession', lambda *args, **kwargs: pytest.fail('session'))
    with pytest.raises((ValueError, TypeError)):
        load_person_package(folder.name, root)


@pytest.mark.parametrize('task', ['person_detection', 'person_reid'])
@pytest.mark.parametrize('preprocessing', [
    {'mean': 1e100}, {'std': 1e-40}, {'std': 1e100}, {'scale': 1e-100},
    {'scale': 1e38}, {'mean': 3e38, 'std': .5},
    {'scale': 1e36, 'mean': -3e38, 'std': 1},
])
def test_body_fp32_execution_limits_do_not_narrow_manifest_schema(package, task, preprocessing, monkeypatch):
    import onnxruntime
    root, folder, document = package
    document['tasks'][task]['preprocessing'] = preprocessing
    (folder / 'manifest.json').write_text(json.dumps(document))
    descriptor = load_person_package(folder.name, root)['tasks'][task]
    monkeypatch.setattr(onnxruntime, 'InferenceSession', lambda *args, **kwargs: pytest.fail('session'))
    with pytest.raises(ValueError, match='float32'):
        create_session(descriptor, ['CPUExecutionProvider'])


@pytest.mark.parametrize('task', ['detection', 'recognition'])
@pytest.mark.parametrize('preprocessing', [
    {'mean': 0}, {'mean': [0, 0, 0], 'std': 1}, {'mean': 0, 'std': 0},
    {'mean': 0, 'std': 1, 'scale': 1}, 'unknown-normalization',
])
def test_face_rules_keep_the_scalar_v2_contract(package, task, preprocessing):
    root, folder, document = package
    document['tasks'][task]['preprocessing'] = preprocessing
    (folder / 'manifest.json').write_text(json.dumps(document))
    with pytest.raises((ValueError, TypeError)):
        load_person_package(folder.name, root)


@pytest.mark.parametrize('task', ['detection', 'recognition'])
@pytest.mark.parametrize('preprocessing', [
    {'mean': 1e100, 'std': 1}, {'mean': 0, 'std': 1e40}, {'mean': 0, 'std': 1e-40},
    {'mean': 3e38, 'std': .5}, {'mean': -3e38, 'std': .5}, {'mean': 0, 'std': 1e-38},
])
def test_finite_face_rules_match_v2_without_extra_fp32_rejection(package, task, preprocessing, monkeypatch):
    import onnx
    import onnxruntime
    from insightface.model_zoo import onnxruntime_utils

    root, folder, document = package
    document['tasks'][task]['preprocessing'] = preprocessing
    (folder / 'manifest.json').write_text(json.dumps(document))
    descriptor = load_person_package(folder.name, root)['tasks'][task]
    assert descriptor['preprocessing'] == load_model_package(folder).task(task).as_config()['preprocessing']
    monkeypatch.setattr(onnx, 'load', lambda *args, **kwargs: SimpleNamespace(graph=SimpleNamespace(node=[])))
    monkeypatch.setattr(onnxruntime, 'get_available_providers', lambda: ['CPUExecutionProvider'])
    monkeypatch.setattr(onnxruntime_utils, 'preload_cuda_libraries', lambda providers: None)
    monkeypatch.setattr(onnxruntime, 'SessionOptions', SimpleNamespace)
    def session(*args, **kwargs):
        raise RuntimeError('session construction reached')
    monkeypatch.setattr(onnxruntime, 'InferenceSession', session)
    with pytest.raises(RuntimeError, match='session construction reached'):
        create_session(descriptor, ['CPUExecutionProvider'])


@pytest.mark.parametrize('size', [[112, 128], [0, 0]])
def test_face_recognition_size_uses_generic_v2_schema(package, size):
    root, folder, document = package
    document['tasks']['recognition']['input_size'] = size
    (folder / 'manifest.json').write_text(json.dumps(document))
    with pytest.raises((ValueError, TypeError)):
        load_person_package(folder.name, root)


@pytest.mark.parametrize('task', ['person_detection', 'recognition'])
def test_adapter_size_limits_are_checked_at_prepare_not_manifest_load(package, task, monkeypatch):
    import onnxruntime
    root, folder, document = package
    document['tasks'][task]['input_size'] = [160, 160]
    (folder / 'manifest.json').write_text(json.dumps(document))
    descriptor = load_person_package(folder.name, root)['tasks'][task]
    assert descriptor['input_size'] == [160, 160]
    monkeypatch.setattr(onnxruntime, 'InferenceSession', lambda *args, **kwargs: pytest.fail('session'))
    with pytest.raises(ValueError, match='input size|input_size'):
        create_session(descriptor, ['CPUExecutionProvider'])


@pytest.mark.parametrize('task,size', [('person_reid', [128, 64]), ('recognition', [224, 224])])
def test_feature_input_size_must_be_supported_by_actual_model(package, task, size):
    root, folder, document = package
    document['tasks'][task]['input_size'] = size
    (folder / 'manifest.json').write_text(json.dumps(document))
    descriptor = load_person_package(folder.name, root)['tasks'][task]
    assert descriptor['input_size'] == size
    with pytest.raises(ValueError, match='node shape'):
        _check_nodes(metadata_session(runtime_metadata(task)), descriptor)
    metadata = runtime_metadata(task)
    metadata['inputs'][0]['shape'][2:] = size
    assert _check_nodes(metadata_session(metadata), descriptor) == [[1, 3, *size]]


def test_reid_configured_size_controls_resize_and_partial_rules_use_defaults(monkeypatch, manifest):
    descriptor = manifest['tasks']['person_reid']
    descriptor.update(input_size=[128, 64], preprocessing=dict(scale=2))
    model = fake_reid(monkeypatch, descriptor)
    assert model._preprocess(np.zeros((10, 20, 3), np.uint8)).shape == (1, 3, 128, 64)
    assert model.descriptor['preprocessing']['scale'] == 2
    assert model.descriptor['preprocessing']['mean'] == [0., 0., 0.]
    assert model.descriptor['preprocessing']['std'] == [1., 1., 1.]


def test_detector_nms_and_original_pixel_coordinates(monkeypatch, manifest):
    boxes = np.array([[[1, 2, 21, 40], [2, 3, 20, 39], [70, 5, 99, 45], [4, 4, 4, 8]]], np.float32)
    scores = np.array([[[.9, .8, .85, .99]]], np.float32)
    detector = fake_detector(monkeypatch, manifest["tasks"]["person_detection"], boxes, scores)
    output = detector.detect(np.zeros((50, 100, 3), np.uint8))
    np.testing.assert_allclose(output, [[1, 2, 21, 40, .9], [70, 5, 99, 45, .85]])
    assert output.dtype == np.float32


def test_detector_empty_and_nonfinite_are_distinct(monkeypatch, manifest):
    descriptor = manifest["tasks"]["person_detection"]
    model = fake_detector(monkeypatch, descriptor, np.empty((1, 0, 4), np.float32), np.empty((1, 1, 0), np.float32))
    assert model.detect(np.zeros((10, 10, 3), np.uint8)).shape == (0, 5)
    model = fake_detector(monkeypatch, descriptor, np.array([[[1, 1, 8, 8]]], np.float32), np.array([[[np.nan]]], np.float32))
    with pytest.raises(RuntimeError, match="nonfinite"):
        model.detect(np.zeros((10, 10, 3), np.uint8))


def test_detector_preprocessing_matches_supplied_demo(monkeypatch, manifest):
    manifest['tasks']['person_detection']['preprocessing'] = dict(
        scale=1 / 255, mean=[.485, .456, .406], std=[.229, .224, .225])
    model = fake_detector(monkeypatch, manifest["tasks"]["person_detection"], None, None)
    bgr = np.random.default_rng(23).integers(0, 256, (37, 53, 3), np.uint8)
    feed = model._preprocess(bgr)
    reference = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), None, fx=640/53, fy=640/37, interpolation=cv2.INTER_CUBIC).astype(np.float32)
    reference *= 1/255
    reference -= np.array([.485,.456,.406])[None,None,:]
    reference /= np.array([.229,.224,.225])[None,None,:]
    np.testing.assert_array_equal(feed["image"], reference.transpose(2,0,1)[None])
    np.testing.assert_allclose(feed["scale_factor"], [[640/37, 640/53]])


@pytest.mark.parametrize('task', ['person_detection', 'person_reid'])
def test_configured_rgb_mean_std_and_scale_change_actual_model_input(monkeypatch, manifest, task):
    descriptor = manifest['tasks'][task]
    descriptor['preprocessing'] = dict(scale=.5, mean=[1, 2, 3], std=[2, 4, 8])
    model = (fake_detector(monkeypatch, descriptor, None, None) if task == 'person_detection'
             else fake_reid(monkeypatch, descriptor))
    image = np.full((12, 20, 3), [12, 34, 56], np.uint8)
    actual = model._preprocess(image)
    if task == 'person_detection':
        actual = actual['image']
    expected = (np.array([56, 34, 12], np.float32) * .5 - [1, 2, 3]) / [2, 4, 8]
    np.testing.assert_array_equal(actual[0, :, 0, 0], expected)
    neutral = {key: descriptor[key] for key in ('sha256', 'input_size')}
    original = (fake_detector(monkeypatch, neutral, None, None) if task == 'person_detection'
                else fake_reid(monkeypatch, neutral))
    assert model.model_id != original.model_id


@pytest.mark.parametrize('task', ['person_detection', 'person_reid'])
def test_omitted_body_preprocessing_preserves_rgb_pixel_values(monkeypatch, manifest, task):
    descriptor = manifest['tasks'][task]
    assert 'preprocessing' not in descriptor
    model = (fake_detector(monkeypatch, descriptor, None, None) if task == 'person_detection'
             else fake_reid(monkeypatch, descriptor))
    actual = model._preprocess(np.full((12, 20, 3), [12, 34, 56], np.uint8))
    if task == 'person_detection':
        actual = actual['image']
    np.testing.assert_array_equal(actual[0, :, 0, 0], [56, 34, 12])


@pytest.mark.parametrize('task', ['person_detection', 'person_reid'])
@pytest.mark.parametrize('input_dtype', ['float32', 'uint8'])
def test_embedded_body_models_receive_raw_rgb_and_preserve_marker(monkeypatch, manifest, task, input_dtype):
    descriptor = manifest['tasks'][task]
    descriptor['preprocessing'] = 'embedded'
    model = (fake_detector(monkeypatch, descriptor, None, None, input_dtype) if task == 'person_detection'
             else fake_reid(monkeypatch, descriptor, input_dtype))
    # Re-resolving an already normalized descriptor must not re-enable external normalization.
    assert person_package.resolve_person_descriptor(task, model.descriptor)['preprocessing'] == 'embedded'
    image = np.full((12, 20, 3), [12, 34, 56], np.uint8)
    actual = model._preprocess(image)
    if task == 'person_detection':
        assert actual['scale_factor'].dtype == np.float32
        actual = actual['image']
    assert actual.dtype == np.dtype(input_dtype)
    np.testing.assert_array_equal(actual[0, :, 0, 0], [56, 34, 12])


@pytest.mark.parametrize('task', ['detection', 'recognition'])
@pytest.mark.parametrize('input_dtype', ['float32', 'uint8'])
def test_embedded_face_wrappers_send_raw_rgb(task, input_dtype):
    from insightface.model_zoo.arcface_onnx import ArcFaceONNX
    from insightface.model_zoo.scrfd import SCRFD
    from insightface.model_zoo.model_zoo import _configure_image_preprocessing

    blobs = []
    def run(names, feed):
        blobs.append(feed['image'])
        return [np.ones((1, 512), np.float32)]
    node_type = 'tensor(uint8)' if input_dtype == 'uint8' else 'tensor(float)'
    session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(type=node_type)], run=run)
    model = SCRFD.__new__(SCRFD) if task == 'detection' else ArcFaceONNX.__new__(ArcFaceONNX)
    _configure_image_preprocessing(model, session, task, 'embedded', 0., 1.)
    image = np.full((12, 20, 3), [12, 34, 56], np.uint8)
    if task == 'detection':
        blob = model._prepare_input_blob(image, (32, 32))
    else:
        model.input_size, model.input_name, model.output_names = (112, 112), 'image', ['embedding']
        model.session = session
        model.get_feat(image)
        blob = blobs[0]
    assert blob.dtype == np.dtype(input_dtype)
    np.testing.assert_array_equal(blob[0, :, 0, 0], [56, 34, 12])


@pytest.mark.parametrize('name,default_size,override_size,manifest_face_size,embedded_dtype', [
    ('synthetic_small', 320, 640, 640, None),
    ('synthetic_large', 640, 320, 640, None),
    ('synthetic_legacy_face_size', 320, 640, 320, None),
    ('synthetic_embedded_float', 320, 640, 640, 'float32'),
    ('synthetic_embedded_uint8', 320, 640, 640, 'uint8'),
])
def test_detection_size_overrides_precede_warmup_and_are_independent_across_models_and_instances(
        tmp_path, monkeypatch, synthetic_person_manifest, synthetic_person_package,
        name, default_size, override_size, manifest_face_size, embedded_dtype):
    """Use the real package loader/session setup with fake ONNX runtime calls."""
    import onnx
    import onnxruntime
    from insightface.model_zoo import arcface_onnx, onnxruntime_utils, scrfd

    document = synthetic_person_manifest(name, default_size)
    document['tasks']['detection']['input_size'] = [manifest_face_size, manifest_face_size]
    if manifest_face_size != 640:
        document['tasks']['detection']['preprocessing'] = dict(mean=100, std=2)
        document['tasks']['recognition']['preprocessing'] = dict(mean=64, std=64)
    if embedded_dtype:
        for descriptor in document['tasks'].values():
            descriptor['preprocessing'] = 'embedded'
    folder = synthetic_person_package(tmp_path, document)
    manifest_path = folder / 'manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    original = copy.deepcopy(document)
    by_path = {str(folder / descriptor['file']): runtime_metadata(task)
               for task, descriptor in document['tasks'].items()}
    if embedded_dtype == 'uint8':
        for metadata in by_path.values():
            metadata['inputs'][0]['type'] = 'tensor(uint8)'
    rewritten = {}
    rewrite_calls = []
    sessions = []

    class FakeSession:
        def __init__(self, source, sess_options, providers):
            self.model_source = source
            self.descriptor = rewritten[source] if isinstance(source, bytes) else by_path[str(source)]
            self.options = sess_options
            self.providers = providers
            self.warmup_feeds = []
            sessions.append(self)

        def disable_fallback(self):
            pass

        def get_providers(self):
            return self.providers

        def nodes(self, kind):
            return [SimpleNamespace(name=node['name'], shape=node['shape'], type=node.get('type', 'tensor(float)'))
                    for node in self.descriptor[kind]]

        def get_inputs(self):
            return self.nodes('inputs')

        def get_outputs(self):
            return self.nodes('outputs')

        def run(self, names, feed):
            assert names == [node['name'] for node in self.descriptor['outputs']]
            for node in self.descriptor['inputs']:
                expected_dtype = np.uint8 if node.get('type') == 'tensor(uint8)' else np.float32
                assert feed[node['name']].dtype == expected_dtype
            self.warmup_feeds.append({name: tuple(value.shape) for name, value in feed.items()})
            task = self.descriptor['task']
            if task == 'person_detection':
                shapes = [(1, 2, 4), (1, 1, 2)]
            elif task == 'person_reid':
                shapes = [(1, 256)]
            elif task == 'recognition':
                shapes = [(1, 512)]
            else:
                height, width = next(iter(feed.values())).shape[2:]
                shapes = [(height // stride * (width // stride) * 2, dimension)
                          for dimension in (1, 4, 10) for stride in (8, 16, 32)]
            return [np.zeros(shape, np.float32) for shape in shapes]

        def end_profiling(self):
            path = Path(self.options.profile_file_prefix + '.json')
            path.write_text('[]')
            return str(path)

    class FakeFaceDetector:
        static_input_size = None

        def __init__(self, path, *, session, static_shape_sessions):
            assert static_shape_sessions is False
            self.session = session

        def prepare(self, *, ctx_id, input_size, det_thresh):
            assert ctx_id == 0 and det_thresh == .5
            self.input_size = input_size

    class FakeFaceRecognizer:
        def __init__(self, path, *, session):
            self.session = session

    def static_scrfd_model(path, input_size, input_name):
        descriptor = copy.deepcopy(by_path[str(path)])
        assert descriptor['task'] == 'detection'
        assert descriptor['inputs'][0]['name'] == input_name
        rewrite_calls.append((str(path), input_size, input_name))
        width, height = input_size
        descriptor['inputs'][0]['shape'] = [1, 3, height, width]
        source = f'static SCRFD {width}x{height}'.encode()
        rewritten[source] = descriptor
        return source

    def load_graph(path, **kwargs):
        inputs = [SimpleNamespace(name=node['name']) for node in by_path[str(path)]['inputs']]
        return SimpleNamespace(graph=SimpleNamespace(node=[], input=inputs))

    monkeypatch.setattr(onnx, 'load', load_graph)
    monkeypatch.setattr(onnxruntime, 'SessionOptions', SimpleNamespace)
    monkeypatch.setattr(onnxruntime, 'InferenceSession', FakeSession)
    monkeypatch.setattr(onnxruntime, 'get_available_providers', lambda: ['CPUExecutionProvider'])
    monkeypatch.setattr(onnxruntime_utils, 'preload_cuda_libraries', lambda providers: None)
    monkeypatch.setattr(scrfd, '_static_scrfd_model', static_scrfd_model)
    monkeypatch.setattr(scrfd, 'SCRFD', FakeFaceDetector)
    monkeypatch.setattr(arcface_onnx, 'ArcFaceONNX', FakeFaceRecognizer)

    default = person_package.prepare_models(name, tmp_path)
    body_changed = person_package.prepare_models(name, tmp_path, body_det_size=override_size)
    face_changed = person_package.prepare_models(name, tmp_path, face_det_size=160)
    combined = person_package.prepare_models(name, tmp_path, body_det_size=override_size, face_det_size=320)
    default_again = person_package.prepare_models(name, tmp_path, body_det_size=0, face_det_size=0)
    image = np.zeros((37, 53, 3), np.uint8)
    cases = ((default, default_size, 640), (body_changed, override_size, 640),
             (face_changed, default_size, 160), (combined, override_size, 320),
             (default_again, default_size, 640))
    for models, size, face_size in cases:
        detector = models['detector']
        assert detector.descriptor['input_size'] == [size, size]
        assert detector.session.warmup_feeds == [{'image': (1, 3, size, size), 'scale_factor': (1, 2)}]
        actual_feed = detector._preprocess(image)
        assert actual_feed['image'].shape == (1, 3, size, size)
        assert actual_feed['image'].dtype == (np.uint8 if embedded_dtype == 'uint8' else np.float32)
        np.testing.assert_allclose(actual_feed['scale_factor'], [[size / 37, size / 53]])
        assert models['reid'].descriptor['input_size'] == [256, 128]
        assert models['reid'].session.warmup_feeds == [{'data': (1, 3, 256, 128)}]
        assert models['face_detector'].input_size == (face_size, face_size)
        assert models['face_detector'].session.warmup_feeds == [{'face_image': (1, 3, face_size, face_size)}]
        face_session = models['face_detector'].session
        if models is face_changed or models is combined or face_size != 640:
            assert isinstance(face_session.model_source, bytes)
            assert face_session.get_inputs()[0].shape == [1, 3, face_size, face_size]
        else:
            assert face_session.model_source == str(folder / document['tasks']['detection']['file'])
        assert models['face_recognizer'].session.warmup_feeds == [{'face_crop': (1, 3, 112, 112)}]
        for task, name in (('detection', 'face_detector'), ('recognition', 'face_recognizer')):
            expected = document['tasks'][task]['preprocessing']
            assert models[name].input_mean == (0. if expected == 'embedded' else expected['mean'])
            assert models[name].input_std == (1. if expected == 'embedded' else expected['std'])
            assert models[name].preprocessing == expected
            assert models[name].input_dtype == (np.uint8 if embedded_dtype == 'uint8' else np.float32)
        for task, adapter in (('person_detection', 'detector'), ('person_reid', 'reid'),
                              ('recognition', 'face_recognizer')):
            assert models[adapter].session.model_source == str(folder / document['tasks'][task]['file'])
        assert models['face_model_id'] == default['face_model_id']
        assert models['reid_model_id'] == default['reid_model_id']
    assert len(sessions) == 20
    assert rewrite_calls == [
        (str(folder / document['tasks']['detection']['file']), (face_size, face_size), 'face_image')
        for models, _, face_size in cases
        if models is face_changed or models is combined or face_size != 640
    ]
    assert len({id(models['detector'].descriptor) for models, _, _ in cases}) == len(cases)
    assert len({id(models['face_detector']) for models, _, _ in cases}) == len(cases)
    assert document == original
    assert manifest_path.read_bytes() == manifest_bytes


def test_reid_preprocessing_preserves_original_multipliers(monkeypatch, manifest):
    image = np.zeros((12, 20, 3), np.uint8)
    image[:] = [12, 34, 56]
    mean = np.array([123.67500305175781, 116.27999877929688, 103.52999877929688], np.float32)
    scale = np.array([0.017124753445386887, 0.017507003620266914, 0.01742919348180294], np.float32)
    manifest['tasks']['person_reid']['preprocessing'] = dict(
        mean=mean.tolist(), std=[1.0 / float(value) for value in scale])
    actual = fake_reid(monkeypatch, manifest['tasks']['person_reid'])._preprocess(image)
    expected = (np.array([56,34,12], np.float32) - mean) * scale
    assert actual.shape == (1,3,256,128)
    np.testing.assert_array_equal(actual[0,:,0,0], expected)


def test_reid_output_normalized_and_invalid_output_fails(monkeypatch, manifest):
    for output, error in [(np.arange(256,dtype=np.float32)[None], False), (np.zeros((1,256),np.float32), True), (np.full((1,256),np.nan,np.float32), True)]:
        fake = SimpleNamespace(run=lambda *args: [output])
        monkeypatch.setattr(person_reid,"create_session",lambda *args: (fake, {}))
        model = person_reid.PersonReID(manifest["tasks"]["person_reid"], ["CPUExecutionProvider"])
        if error:
            with pytest.raises(RuntimeError): model.embed(np.zeros((10,10,3),np.uint8))
        else:
            feature = model.embed(np.zeros((10,10,3),np.uint8))
            assert feature.shape == (256,)
            assert np.linalg.norm(feature) == pytest.approx(1,abs=1e-6)


@pytest.mark.parametrize("image", [None, np.zeros((0,5,3),np.uint8), np.zeros((5,5),np.uint8),np.zeros((5,5,4),np.uint8),np.zeros((5,5,3),np.float32)])
def test_invalid_images_rejected_before_inference(image, monkeypatch, manifest):
    model = fake_reid(monkeypatch, manifest['tasks']['person_reid'])
    with pytest.raises(ValueError): model._preprocess(image)


def test_explicit_unavailable_provider_never_falls_back(package, monkeypatch):
    import onnxruntime
    root, folder, _ = package
    descriptor = load_person_package(folder.name, root)["tasks"]["person_reid"]
    monkeypatch.setattr(onnxruntime,"get_available_providers",lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(onnxruntime,"InferenceSession",lambda *a, **kw: pytest.fail("must fail before session"))
    with pytest.raises(RuntimeError,match="unavailable"):
        create_session(descriptor,["CUDAExecutionProvider"])


@pytest.mark.parametrize('task,mutate', [
    ('person_reid', lambda data: data['outputs'][0].update(name='wrong_output')),
    ('person_reid', lambda data: data['outputs'][0].update(shape=[1, 128])),
    ('person_reid', lambda data: data['outputs'][0].update(shape=[1, 'features'])),
    ('person_reid', lambda data: data['inputs'][0].update(type='tensor(uint8)')),
    ('person_detection', lambda data: data['inputs'][0].update(name='guessed')),
    ('person_detection', lambda data: data['inputs'][0].update(shape=[1, 1, None, None])),
    ('person_detection', lambda data: data['outputs'][1].update(shape=[1, 80, None])),
    ('recognition', lambda data: data['outputs'][0].update(shape=[1, 256])),
    ('recognition', lambda data: data['inputs'][0].update(shape=[1, 3, 224, 224])),
    ('recognition', lambda data: data['inputs'][0].update(shape=[1, 3, None, None])),
    ('recognition', lambda data: data['outputs'][0].update(shape=[1, None])),
    ('detection', lambda data: data['outputs'].__delitem__(slice(6, 9))),
    ('detection', lambda data: data['outputs'][6].update(shape=[None, 4])),
    ('detection', lambda data: data['outputs'][0].update(type='tensor(double)')),
])
def test_runtime_node_contracts_reject_wrong_models(package, task, mutate):
    root, folder, _ = package
    descriptor = load_person_package(folder.name, root)['tasks'][task]
    data = runtime_metadata(task)
    mutate(data)
    with pytest.raises(ValueError):
        _check_nodes(metadata_session(data), descriptor)


@pytest.mark.parametrize('task', ['person_detection', 'person_reid', 'detection', 'recognition'])
def test_uint8_is_only_allowed_for_embedded_image_input(package, task):
    root, folder, document = package
    descriptor = load_person_package(folder.name, root)['tasks'][task]
    metadata = runtime_metadata(task)
    metadata['inputs'][0]['type'] = 'tensor(uint8)'
    with pytest.raises(ValueError, match='dtype'):
        _check_nodes(metadata_session(metadata), descriptor)
    document['tasks'][task]['preprocessing'] = 'embedded'
    (folder / 'manifest.json').write_text(json.dumps(document))
    descriptor = load_person_package(folder.name, root)['tasks'][task]
    _check_nodes(metadata_session(metadata), descriptor)
    assert descriptor['input_dtype'] == 'uint8'
    assert descriptor['preprocessing'] == 'embedded'
    metadata['outputs'][0]['type'] = 'tensor(uint8)'
    with pytest.raises(ValueError, match='dtype'):
        _check_nodes(metadata_session(metadata), descriptor)
    if task == 'person_detection':
        metadata['outputs'][0]['type'] = 'tensor(float)'
        metadata['inputs'][1]['type'] = 'tensor(uint8)'
        with pytest.raises(ValueError, match='dtype'):
            _check_nodes(metadata_session(metadata), descriptor)


def test_recognition_dimension_is_declared_and_checked_against_actual_model(package):
    root, folder, document = package
    document['tasks']['recognition']['embedding_dimension'] = 128
    (folder / 'manifest.json').write_text(json.dumps(document))
    descriptor = load_person_package(folder.name, root)['tasks']['recognition']
    assert descriptor['embedding_dimension'] == 128
    metadata = runtime_metadata('recognition')
    with pytest.raises(ValueError, match='node shape'):
        _check_nodes(metadata_session(metadata), descriptor)
    metadata['outputs'][0]['shape'] = [1, 128]
    _check_nodes(metadata_session(metadata), descriptor)
    _check_warmup(descriptor, [np.zeros((1, 128), np.float32)])
    with pytest.raises(ValueError, match='warmup output shape'):
        _check_warmup(descriptor, [np.zeros((1, 512), np.float32)])


@pytest.mark.parametrize('task', ['person_detection', 'person_reid', 'detection', 'recognition'])
def test_runtime_nodes_are_read_from_model_and_feature_semantics_stay_stable(package, task):
    root, folder, _ = package
    descriptor = load_person_package(folder.name, root)['tasks'][task]
    fingerprint = model_fingerprint(descriptor)
    metadata = runtime_metadata(task)
    if task in ('detection', 'recognition'):
        for index, node in enumerate(metadata['inputs'] + metadata['outputs']):
            node['name'] = f'actual_graph_node_{index}'
    _check_nodes(metadata_session(metadata), descriptor)
    assert descriptor['inputs'] == metadata['inputs']
    assert descriptor['outputs'] == metadata['outputs']
    assert model_fingerprint(descriptor) == fingerprint


@pytest.mark.parametrize('task,dimension', [('person_reid', 256), ('recognition', 512)])
def test_feature_models_allow_dynamic_batch_but_warmup_uses_one(package, task, dimension):
    root, folder, _ = package
    descriptor = load_person_package(folder.name, root)['tasks'][task]
    metadata = runtime_metadata(task)
    for node in metadata['inputs'] + metadata['outputs']:
        node['shape'][0] = 'batch'
    input_shapes = _check_nodes(metadata_session(metadata), descriptor)
    assert input_shapes[0][0] == 1
    _check_warmup(descriptor, [np.zeros((1, dimension), np.float32)])
    with pytest.raises(ValueError, match='warmup output shape'):
        _check_warmup(descriptor, [np.zeros((2, dimension), np.float32)])


@pytest.mark.parametrize('task,shapes', [
    ('person_detection', [(1, 3, 4), (1, 1, 2)]),
    ('person_reid', [(1, 128)]),
    ('recognition', [(2, 512)]),
    ('detection', [(1, 1)] * 3 + [(1, 4)] * 3 + [(1, 10)] * 3),
])
def test_warmup_rejects_wrong_dynamic_output_dimensions(package, task, shapes):
    root, folder, _ = package
    descriptor = load_person_package(folder.name, root)['tasks'][task]
    _check_nodes(metadata_session(runtime_metadata(task)), descriptor)
    values = [np.zeros(shape, np.float32) for shape in shapes]
    with pytest.raises(ValueError, match='warmup output shape'):
        _check_warmup(descriptor, values)


@pytest.mark.parametrize('declared_hash', [False, True])
def test_artifact_rechecked_after_metadata_load_before_session(package, monkeypatch, declared_hash):
    import onnxruntime
    root, folder, document = package
    if not declared_hash:
        document['tasks']['person_reid'].pop('sha256')
        (folder / 'manifest.json').write_text(json.dumps(document))
    descriptor = load_person_package(folder.name, root)['tasks']['person_reid']
    Path(descriptor['path']).write_bytes(b'changed after metadata check')
    monkeypatch.setattr(onnxruntime, 'InferenceSession', lambda *args, **kwargs: pytest.fail('session'))
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        create_session(descriptor, ['CPUExecutionProvider'])


def test_in_graph_nms_rejected_before_session(package, monkeypatch):
    import onnx
    import onnxruntime
    from insightface.model_zoo import onnxruntime_utils
    root, folder, _ = package
    descriptor = load_person_package(folder.name, root)['tasks']['person_detection']
    graph = SimpleNamespace(graph=SimpleNamespace(node=[SimpleNamespace(op_type='NonMaxSuppression')]))
    monkeypatch.setattr(onnx, 'load', lambda *args, **kwargs: graph)
    monkeypatch.setattr(onnxruntime, 'get_available_providers', lambda: ['CPUExecutionProvider'])
    monkeypatch.setattr(onnxruntime_utils, 'preload_cuda_libraries', lambda providers: None)
    monkeypatch.setattr(onnxruntime, 'InferenceSession', lambda *args, **kwargs: pytest.fail('session'))
    with pytest.raises(ValueError, match='NonMaxSuppression'):
        create_session(descriptor, ['CPUExecutionProvider'])

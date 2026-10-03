"""Public value shapes, configuration and strict input contracts."""
import json
from dataclasses import asdict, FrozenInstanceError

import numpy as np
import pytest

from insightface.app import PersonAnalysis
from insightface.app.person import Face, Person, MatchResult, PersonConfig, UpdateResult
from insightface.app.person.types import normalize_feature, validate_image


def test_public_serialization_contains_only_the_minimal_observation_fields():
    person = Person(np.array([0, 0, 10, 20]), .9, np.array([1., 0.]),
                    Face(np.array([1, 1, 5, 5]), .95, embedding=np.array([0., 1.])))
    person._face_model_id = 'private-model'
    payload = person.to_dict()
    assert set(payload) == {'body_bbox', 'det_score', 'reid_feature', 'face'}
    assert set(payload['face']) == {'bbox', 'det_score', 'kps', 'embedding'}
    unmatched = MatchResult(person).to_dict()
    assert set(unmatched) == {'observation', 'person_id', 'matched_by', 'similarity'}
    assert unmatched['person_id'] is unmatched['matched_by'] is unmatched['similarity'] is None
    json.dumps(unmatched)
    assert UpdateResult(1, 2, 3).to_dict() == dict(added=1, replaced=2, skipped=3)


@pytest.mark.parametrize('option', [dict(face_min_size=0), dict(max_body_samples=True),
    dict(face_similarity_threshold=float('nan')), dict(reid_margin=-.1),
    dict(reference_capacity=1.5), dict(duplicate_similarity_threshold=1.1)])
def test_configuration_rejects_invalid_values(option):
    with pytest.raises(ValueError):
        PersonConfig(**option)


def test_configuration_has_no_stateful_project_options():
    config = PersonConfig()
    assert config.max_body_samples == 4
    assert config.body_det_size == 0
    assert config.face_det_size == 0
    for name in ('face_confirmations', 'anonymous_identity_enabled', 'history_cache_max_bytes',
                 'face_interval_ms', 'track_confirmations', 'max_face_samples'):
        assert not hasattr(config, name)


@pytest.mark.parametrize('size', [0, 64, 320, 640, 1280])
def test_body_detector_size_accepts_model_default_or_positive_multiples_of_64(size):
    config = PersonConfig(body_det_size=size)
    assert config.body_det_size == size
    with pytest.raises(FrozenInstanceError):
        config.body_det_size = 640


@pytest.mark.parametrize('size', [-64, -1, 1, 319, 321, True, False, 0., 320.,
                                '320', None, float('nan'), float('inf')])
def test_body_detector_size_rejects_invalid_values(size):
    with pytest.raises(ValueError, match='body_det_size'):
        PersonConfig(body_det_size=size)


@pytest.mark.parametrize('size', [0, 32, 160, 320, 640, 1280])
def test_face_detector_size_accepts_model_default_or_positive_multiples_of_32(size):
    config = PersonConfig(face_det_size=size)
    assert config.face_det_size == size
    with pytest.raises(FrozenInstanceError):
        config.face_det_size = 640


@pytest.mark.parametrize('size', [-32, -1, 1, 33, 641, True, False, 0., 320.,
                                '320', [320, 320], None, float('nan'), float('inf')])
def test_face_detector_size_rejects_invalid_values(size):
    with pytest.raises(ValueError, match='face_det_size'):
        PersonConfig(face_det_size=size)


@pytest.mark.parametrize('value', [None, np.zeros((0, 2, 3), np.uint8),
    np.zeros((10, 10), np.uint8), np.zeros((10, 10, 3), np.float32)])
def test_invalid_bgr_inputs_fail_clearly(value):
    with pytest.raises(ValueError, match='BGR uint8'):
        validate_image(value)


@pytest.mark.parametrize('value', [[], [0., 0.], [float('nan'), 1.], [[1., 0.]],
                                   ['1', '0'], [True, False], [1 + 1j, 0]])
def test_invalid_vectors_are_rejected(value):
    with pytest.raises(ValueError):
        normalize_feature(value)


def test_valid_features_are_independent_normalized_fp32():
    raw = np.array([3., 4.], np.float64)
    result = normalize_feature(raw)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [.6, .8])
    np.testing.assert_array_equal(raw, [3., 4.])


@pytest.mark.parametrize('body_det_size,face_det_size', [(0, 0), (320, 160), (640, 320)])
def test_simple_json_configuration_is_lazy_and_file_relative(tmp_path, body_det_size, face_det_size):
    path = tmp_path / 'options.json'
    config = PersonConfig(max_body_samples=2, body_det_size=body_det_size, face_det_size=face_det_size)
    path.write_text(json.dumps({'name': 'cheetah_s', 'root': 'models',
                               'config': asdict(config)}))
    app = PersonAnalysis.from_config(path)
    assert not app._prepared and app.root == str(tmp_path / 'models')
    assert app.config.max_body_samples == 2
    assert app.config == config and app.config.body_det_size == body_det_size
    assert app.config.face_det_size == face_det_size
    restored = PersonAnalysis.from_config(json.loads(path.read_text()))
    assert restored.config == config
    assert PersonAnalysis.from_config({'name': 'cheetah_l'}, name='cheetah_s').name == 'cheetah_s'
    with pytest.raises(ValueError, match='unknown'):
        PersonAnalysis.from_config({'database': True})
    with pytest.raises(TypeError):
        PersonAnalysis.from_config({'config': {'face_confirmations': 2}})


@pytest.mark.parametrize('person_id', [None, '', ' ', True, 0, -1, []])
def test_person_ids_are_explicit_labels_without_anonymous_namespace(person_id):
    with pytest.raises(ValueError):
        PersonAnalysis._person_id(person_id)

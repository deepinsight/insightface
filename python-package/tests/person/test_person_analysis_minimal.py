"""Deterministic interface and evidence tests; no model accuracy claims."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from insightface.app import PersonAnalysis
from insightface.app.person import PersonConfig


IMAGE = np.full((200, 260, 3), 80, np.uint8)


@pytest.fixture
def setup(monkeypatch):
    state = SimpleNamespace(body_boxes=[[0, 0, 100, 190, .95]],
        face_boxes=[[20, 20, 70, 75, .99]], face_feature=np.array([1., 0., 0.]),
        body_feature=np.array([1., 0., 0., 0.]), calls=[], preparations=0, prepare_options=[],
        face_inputs=[])

    class Detector:
        def detect(self, image):
            state.calls.append('body_detection')
            return np.asarray(state.body_boxes, np.float32).reshape(-1, 5)

    class FaceDetector:
        static_input_size = None

        def __init__(self, size=640):
            self.input_size = self.configured_input_size = (size, size)

        def detect(self, image, **kwargs):
            assert kwargs['input_size'] == self.configured_input_size
            state.face_inputs.append((kwargs['input_size'], image.shape))
            state.calls.append('face_detection')
            boxes = np.asarray(state.face_boxes, np.float32).reshape(-1, 5)
            points = np.tile(np.array([[30, 35], [55, 35], [43, 45], [34, 58], [54, 58]], np.float32),
                             (len(boxes), 1, 1))
            return boxes, points

    class FaceRecognizer:
        output_shape = [1, 3]

        def get(self, image, face):
            state.calls.append('face_embedding')
            return state.face_feature.copy()

    class BodyRecognizer:
        descriptor = {'outputs': [{'shape': [1, 4]}]}

        def embed(self, image):
            assert image.size > 0
            state.calls.append('body_embedding')
            return state.body_feature.copy()

    def factory(name, root, providers, threads=4, body_det_size=0, face_det_size=0):
        state.preparations += 1
        state.prepare_options.append(dict(name=name, threads=threads,
                                          body_det_size=body_det_size, face_det_size=face_det_size))
        return dict(detector=Detector(), reid=BodyRecognizer(), face_detector=FaceDetector(face_det_size or 640),
                    face_recognizer=FaceRecognizer(), face_model_id='face-' + name, reid_model_id='body-v1')

    monkeypatch.setattr('insightface.app.person_analysis.prepare_models', factory)
    return state, PersonAnalysis(name='cheetah_s')


def test_prepare_returns_none_and_reuses_models(setup):
    state, app = setup
    assert app.prepare() is None and app.prepare() is None
    assert state.preparations == 1
    assert state.prepare_options[0]['body_det_size'] == 0
    assert state.prepare_options[0]['face_det_size'] == 0
    assert len(app._face_references) == len(app._body_references) == 0


@pytest.mark.parametrize('body_det_size', [0, 320, 640])
def test_body_detector_size_is_forwarded_once_during_initial_preparation(setup, body_det_size):
    state, _ = setup
    app = PersonAnalysis(name='cheetah_s', config=PersonConfig(body_det_size=body_det_size))
    with pytest.raises(TypeError):
        app.prepare(body_det_size=body_det_size)
    assert state.preparations == 0
    app.prepare()
    detector = app.detector
    app.prepare()
    app.get(IMAGE)
    assert app.detector is detector and state.preparations == 1
    assert state.prepare_options == [dict(name='cheetah_s', threads=4,
                                         body_det_size=body_det_size, face_det_size=0)]
    app.close()


@pytest.mark.parametrize('face_det_size', [0, 160, 320, 640])
def test_face_size_is_initialized_once_and_used_for_registration_and_original_image_coordinates(
        setup, face_det_size):
    state, _ = setup
    state.body_boxes = []
    # Coordinates outside a 160-pixel detector input still belong to the
    # original 260x200 image and must not be clipped to the model input size.
    state.face_boxes = [[170, 110, 245, 185, .99]]
    app = PersonAnalysis(name='cheetah_s', config=PersonConfig(body_det_size=320,
                                                            face_det_size=face_det_size))
    with pytest.raises(TypeError):
        app.prepare(face_det_size=face_det_size)
    assert state.preparations == 0
    registered = app.register('alice', IMAGE)
    assert registered.accepted == 1 and not registered.rejected
    detector = app.face_detector
    # The app keeps its initialization size even if the adapter's convenience
    # attribute later changes; repeated prepare must not recreate/reset models.
    detector.input_size = (1280, 1280)
    app.prepare()
    persons = app.get(IMAGE)
    assert app.face_detector is detector and state.preparations == 1
    assert state.prepare_options == [dict(name='cheetah_s', threads=4,
                                         body_det_size=320, face_det_size=face_det_size)]
    size = face_det_size or 640
    assert state.face_inputs == [((size, size), IMAGE.shape)] * 2
    assert len(persons) == 1 and persons[0].body_bbox is None
    np.testing.assert_array_equal(persons[0].face.bbox, [170, 110, 245, 185])
    assert app.match(persons)[0].person_id == 'alice'
    app.close()


@pytest.mark.parametrize('static_size,compatible', [((320, 320), True), ((640, 640), False)])
def test_static_face_model_size_must_match_initially_selected_size(setup, static_size, compatible):
    state, _ = setup
    app = PersonAnalysis(name='cheetah_s', config=PersonConfig(face_det_size=320))
    app.prepare()
    app.face_detector.static_input_size = static_size
    if compatible:
        assert app.get(IMAGE)
        assert state.face_inputs == [((320, 320), IMAGE.shape)]
    else:
        with pytest.raises(ValueError):
            app.get(IMAGE)
        assert not state.face_inputs
    app.close()


def test_close_releases_a_partially_initialized_application(setup, monkeypatch):
    from insightface.app.person.matrix import ReferenceMatrix
    state, app = setup

    def allocate(dimension, model_id, **options):
        if model_id == 'body-v1':
            raise RuntimeError('cannot allocate body references')
        return ReferenceMatrix(dimension, model_id, **options)

    monkeypatch.setattr('insightface.app.person_analysis.ReferenceMatrix', allocate)
    with pytest.raises(RuntimeError, match='allocate body'):
        app.prepare()
    assert not app._prepared and app._face_references is not None and app._body_references is None
    app.close()
    app.close()
    assert app._face_references is app._body_references is None
    assert app.detector is app.reid is app.face_detector is app.face_recognizer is None


def test_get_and_match_do_not_register_or_update_any_references(setup):
    state, app = setup
    observations = app.get(IMAGE)
    assert len(observations) == 1 and observations[0].face is not None
    assert observations[0].body_bbox is not None
    result = app.match(observations)
    assert len(result) == len(observations)
    assert result[0].observation is observations[0]
    assert (result[0].person_id, result[0].matched_by, result[0].similarity) == (None, None, None)
    assert len(app._face_references) == len(app._body_references) == 0
    assert app.update(result).to_dict() == dict(added=0, replaced=0, skipped=1)


def test_register_face_then_explicit_update_enables_body_only_matching(setup):
    state, app = setup
    assert app.register('alice', IMAGE).accepted == 1
    observations = app.get(IMAGE)
    face_match = app.match(observations)
    assert face_match[0].matched_by == 'face' and face_match[0].person_id == 'alice'
    assert len(app._body_references) == 0
    assert app.update(face_match).added == 1
    assert app.update(face_match).skipped == 1
    state.face_boxes = []
    body_matches = app.match(app.get(IMAGE))
    assert body_matches[0].matched_by == 'body' and body_matches[0].person_id == 'alice'
    state.body_feature = np.array([.95, .31, 0., 0.])
    assert app.update(body_matches).skipped == 1
    assert len(app._body_references) == 1  # A clothing-only match cannot reinforce itself.


@pytest.mark.parametrize('single_call', [True, False])
def test_face_registration_retains_all_distinct_photos_without_a_sample_cap(setup, monkeypatch, single_call):
    state, _ = setup
    app = PersonAnalysis(name='cheetah_s', config=PersonConfig(reference_capacity=2, max_body_samples=1))
    app.prepare()
    features = np.eye(3, dtype=np.float32)
    pending = iter(features)
    with monkeypatch.context() as patch:
        patch.setattr(app.face_recognizer, 'get', lambda image, face: next(pending).copy())
        batches = [[IMAGE] * 3] if single_call else [[IMAGE]] * 3
        results = [app.register('alice', images) for images in batches]
    assert sum(result.accepted for result in results) == 3
    assert all(not result.rejected for result in results)
    assert app._face_references.capacity == 4
    np.testing.assert_array_equal(app._face_references.matrix[:3], features)

    # Each registered angle remains searchable; update only learns body samples.
    for feature in features:
        state.face_feature = feature
        matches = app.match(app.get(IMAGE))
        assert matches[0].person_id == 'alice' and matches[0].matched_by == 'face'
        assert matches[0].similarity == pytest.approx(1.)
        app.update(matches)
    np.testing.assert_array_equal(app._face_references.matrix[:3], features)
    assert app._face_references.sample_counts == {'alice': 3}

    state.face_feature = np.array([-1., 0., 0.])
    assert app.register('alice', IMAGE).accepted == 1
    duplicate = app.register('alice', IMAGE)
    assert duplicate.accepted == 0 and duplicate.rejected[0]['reason'] == 'duplicate'
    assert app._face_references.sample_counts == {'alice': 4}
    assert app._face_references.match(features[0])['similarity'] == pytest.approx(1.)

    # The body cap and oldest automatic replacement still apply independently.
    state.body_feature = np.array([0., 1., 0., 0.])
    assert app.update(app.match(app.get(IMAGE))).replaced == 1
    assert app._body_references.sample_counts == {'alice': 1}
    assert app.remove_person('alice') == 5
    assert len(app._face_references) == len(app._body_references) == 0


def test_match_entire_batch_before_update_prevents_same_batch_self_learning(setup):
    state, app = setup
    app.register('alice', IMAGE)
    joint = app.get(IMAGE)[0]
    state.face_boxes = []
    body = app.get(IMAGE)[0]
    results = app.match([joint, body])
    assert [result.matched_by for result in results] == ['face', None]
    assert len(app._body_references) == 0
    assert app.update(results).to_dict() == dict(added=1, replaced=0, skipped=1)
    assert app.match([body])[0].person_id == 'alice'


def test_face_only_works_without_body_and_never_invents_reid(setup):
    state, app = setup
    app.register('alice', IMAGE)
    state.body_boxes = []
    state.calls.clear()
    observations = app.get(IMAGE)
    assert len(observations) == 1 and observations[0].body_bbox is None
    assert observations[0].det_score is None and observations[0].face.det_score == pytest.approx(.99)
    assert observations[0].reid_feature is None
    matches = app.match(observations)
    assert matches[0].person_id == 'alice' and matches[0].matched_by == 'face'
    assert app.update(matches).skipped == 1 and 'body_embedding' not in state.calls


def test_ambiguous_face_body_ownership_stays_independent(setup):
    state, app = setup
    app.register('alice', IMAGE)
    state.body_boxes = [[0, 0, 100, 190, .95], [0, 0, 102, 190, .94]]
    observations = app.get(IMAGE)
    assert len(observations) == 3
    assert all(person.face is None for person in observations if person.body_bbox is not None)
    face_only = next(person for person in observations if person.body_bbox is None)
    assert face_only.face.embedding is not None
    matches = app.match(observations)
    assert sum(result.matched_by == 'face' for result in matches) == 1
    assert app.update(matches).added == 0


def test_low_quality_detections_keep_boxes_without_features(setup):
    state, app = setup
    state.face_boxes = [[20, 20, 30, 30, .99]]
    state.body_boxes = [[0, 0, 100, 190, .3]]
    state.calls.clear()
    observations = app.get(IMAGE)
    assert observations[0].face.embedding is None and observations[0].reid_feature is None
    assert 'body_embedding' not in state.calls and 'face_embedding' not in state.calls
    assert app.match(observations)[0].person_id is None


def test_get_reid_is_normalized_and_does_not_detect(setup):
    state, app = setup
    state.body_feature = np.array([2., 0., 0., 0.])
    feature = app.get_reid(IMAGE[:80, :40])
    assert feature.dtype == np.float32 and np.linalg.norm(feature) == pytest.approx(1.)
    assert state.calls == ['body_embedding']


def test_manual_body_reference_is_supported_without_a_face_reference(setup):
    state, app = setup
    registered = app.register('alice', [], body_images=IMAGE)
    assert registered.accepted == 1 and not registered.rejected
    assert len(app._face_references) == 0
    state.face_boxes = []
    assert app.match(app.get(IMAGE))[0].matched_by == 'body'


def test_face_evidence_wins_over_a_conflicting_body_match(setup):
    state, app = setup
    app.register('alice', IMAGE)
    app.register('bob', [], body_images=IMAGE)
    result = app.match(app.get(IMAGE))[0]
    assert result.person_id == 'alice' and result.matched_by == 'face'


def test_qualified_unmatched_face_does_not_suppress_body_match(setup):
    state, app = setup
    app.register('alice', IMAGE, body_images=IMAGE)
    state.face_feature = np.array([0., 1., 0.])
    result = app.match(app.get(IMAGE))[0]
    assert result.person_id == 'alice' and result.matched_by == 'body'
    assert app.update([result]).skipped == 1


def test_all_manual_references_are_protected_from_automatic_replacement(setup):
    state, _ = setup
    app = PersonAnalysis(name='cheetah_s', config=PersonConfig(max_body_samples=1))
    app.register('alice', IMAGE, body_images=IMAGE)
    state.body_feature = np.array([0., 1., 0., 0.])
    match = app.match(app.get(IMAGE))
    assert app.update(match).to_dict() == dict(added=0, replaced=0, skipped=1)
    np.testing.assert_array_equal(app._body_references.matrix[0], [1., 0., 0., 0.])


def test_automatic_body_samples_are_bounded_and_oldest_automatic_is_replaced(setup):
    state, _ = setup
    app = PersonAnalysis(name='cheetah_s', config=PersonConfig(max_body_samples=1))
    app.register('alice', IMAGE)
    assert app.update(app.match(app.get(IMAGE))).added == 1
    state.body_feature = np.array([0., 1., 0., 0.])
    assert app.update(app.match(app.get(IMAGE))).replaced == 1
    assert app._body_references.sample_counts == {'alice': 1}
    assert app._face_references.sample_counts == {'alice': 1}


def test_update_uses_the_matched_snapshot_not_a_later_mutation(setup):
    state, app = setup
    app.register('alice', IMAGE)
    observations = app.get(IMAGE)
    matches = app.match(observations)
    observations[0].reid_feature[:] = [0., 1., 0., 0.]
    assert app.update(matches).added == 1
    np.testing.assert_array_equal(app._body_references.matrix[0], [1., 0., 0., 0.])


def test_registration_changes_and_other_instances_reject_stale_update_batches(setup):
    state, app = setup
    app.register('alice', IMAGE)
    matches = app.match(app.get(IMAGE))
    other = PersonAnalysis(name='cheetah_s')
    with pytest.raises(ValueError, match='another'):
        other.update(matches)
    app.remove_person('alice')
    with pytest.raises(ValueError, match='changed'):
        app.update(matches)
    assert len(app._body_references) == 0


def test_update_validates_whole_batch_before_writing_any_reference(setup):
    state, app = setup
    app.register('alice', IMAGE)
    own = app.match(app.get(IMAGE))[0]
    other = PersonAnalysis(name='cheetah_s')
    foreign = other.match(other.get(IMAGE))[0]
    with pytest.raises(ValueError, match='another'):
        app.update([own, foreign])
    assert len(app._body_references) == 0 and not own._used
    assert app.update([own, own]).to_dict() == dict(added=1, replaced=0, skipped=1)


def test_model_features_cannot_cross_incompatible_face_models(setup):
    state, app = setup
    observations = app.get(IMAGE)
    other = PersonAnalysis(name='cheetah_l')
    with pytest.raises(ValueError, match='face feature model mismatch'):
        other.match(observations)
    observations[0].face = None
    assert other.match(observations)[0].person_id is None  # The shared ReID model is compatible.
    observations[0]._reid_model_id = 'another-body-model'
    with pytest.raises(ValueError, match='body feature model mismatch'):
        app.match(observations)


def test_unicode_reference_path_and_registration_rejections(setup, tmp_path):
    state, app = setup
    path = tmp_path / '张三参考照片.png'
    ok, encoded = cv2.imencode('.png', IMAGE)
    assert ok
    encoded.tofile(path)
    assert app.register('alice', path).accepted == 1
    state.face_boxes = []
    rejected = app.register('bob', IMAGE)
    assert rejected.accepted == 0 and rejected.rejected[0]['reason'] == 'no_face'
    assert app.register('bob', tmp_path / 'missing.png').accepted == 0


def test_get_deduplicates_faces_before_feature_extraction(setup):
    state, app = setup
    state.face_boxes.append([21, 20, 70, 75, .98])
    app.get(IMAGE)
    assert state.calls.count('face_embedding') == 1


def test_zero_or_wrong_shape_feature_never_becomes_a_face_reference(setup):
    state, app = setup
    state.face_feature = np.zeros(3)
    result = app.register('alice', IMAGE)
    assert result.accepted == 0 and result.rejected[0]['reason'] == 'invalid_embedding'
    assert len(app._face_references) == 0


def test_remove_clear_close_and_no_old_project_api(setup):
    state, app = setup
    app.register('alice', IMAGE, body_images=IMAGE)
    assert app.remove_person('alice') == 2
    app.register('alice', IMAGE)
    app.clear_references()
    assert len(app._face_references) == len(app._body_references) == 0
    for name in ('open_camera', 'open_video', 'run', 'create_session', 'query_events', 'database'):
        assert not hasattr(app, name)
    app.close()
    app.close()
    assert app._face_references is app._body_references is None
    with pytest.raises(RuntimeError, match='closed'):
        app.get(IMAGE)
    with pytest.raises(RuntimeError, match='closed'):
        app.prepare()

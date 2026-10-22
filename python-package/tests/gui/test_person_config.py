"""GUI person options use exactly the small SDK PersonConfig fields."""
import json
from pathlib import Path
import pytest
from insightface.gui.core.config import AppConfig, PersonConfigError, load_config, save_config


@pytest.mark.parametrize('analysis_max_fps', [0, 0.25, 5, 15.5])
def test_person_options_roundtrip_without_history_fields(tmp_path, analysis_max_fps):
    options = {'face_similarity_threshold': .5, 'reid_similarity_threshold': .8,
               'max_body_samples': 4, 'reference_capacity': 512}
    config = AppConfig(workspace_path=str(tmp_path), person_config=options,
                       person_analysis_max_fps=analysis_max_fps)
    path = save_config(config)
    document = json.loads(path.read_text())
    assert document['person_config'] == options
    assert document['person_analysis_max_fps'] == analysis_max_fps
    assert 'analysis_max_fps' not in document['person_config']
    assert 'history_face_max_samples' not in document and 'history_body_max_samples' not in document
    restored, loaded = load_config(path)
    assert loaded and restored.to_dict() == config.to_dict()


@pytest.mark.parametrize('options', [None, [], {'face_interval_ms': 500},
    {'history_face_max_samples': 2}, {'anonymous_identity_enabled': True}, {'max_face_samples': 2},
    {'max_body_samples': 0}, {'reid_similarity_threshold': 1.1}, {'cpu_threads': True}]
    + [{'body_det_size': value} for value in (-64, 319, True, False, 0., 320., '320')]
    + [{'face_det_size': value} for value in (-32, 33, 641, True, False, 0., 320., '320', [320, 320])])
def test_invalid_person_settings_are_reported_without_rewriting_file(tmp_path, options):
    path = tmp_path / 'config.json'
    original = json.dumps({'workspace_path': str(tmp_path), 'person_config': options})
    path.write_text(original)
    with pytest.raises(PersonConfigError) as error:
        load_config(path)
    assert str(path) in str(error.value) and path.read_text() == original


def test_mutable_settings_are_independent_and_validated_before_save(tmp_path):
    options = {'max_body_samples': 4}
    config = AppConfig(workspace_path=str(tmp_path), person_config=options)
    options['max_body_samples'] = 1
    assert config.person_config == {'max_body_samples': 4}
    path = save_config(config)
    original = path.read_bytes()
    config.person_config['max_body_samples'] = 0
    with pytest.raises(PersonConfigError):
        save_config(config)
    assert path.read_bytes() == original


@pytest.mark.parametrize('body_det_size,face_det_size', [(0, 0), (320, 160), (640, 320)])
def test_detector_sizes_roundtrip_independently_of_global_face_size(tmp_path, body_det_size, face_det_size):
    settings = {'body_det_size': body_det_size, 'face_det_size': face_det_size}
    config = AppConfig(workspace_path=str(tmp_path), det_size=[1280, 720],
                       person_config=settings)
    path = save_config(config)
    saved = json.loads(path.read_text())
    assert saved['person_config'] == settings
    restored, loaded = load_config(path)
    assert loaded and restored.person_config == settings
    assert restored.det_size == [1280, 720]
    original = path.read_bytes()
    for key, invalid in (('body_det_size', 321), ('face_det_size', 641)):
        config.person_config[key] = invalid
        with pytest.raises(PersonConfigError, match=key):
            save_config(config)
        assert path.read_bytes() == original
        config.person_config[key] = settings[key]


def test_missing_settings_use_defaults_without_creating_file(tmp_path):
    path = tmp_path / 'new' / 'config.json'
    config, loaded = load_config(path)
    assert not loaded and not path.exists() and config.person_config == {}
    assert config.person_analysis_max_fps == 0


def test_existing_settings_without_analysis_cap_default_to_auto(tmp_path):
    path = tmp_path / 'config.json'
    original = json.dumps({'workspace_path': str(tmp_path), 'person_config': {}})
    path.write_text(original)
    config, loaded = load_config(path)
    assert loaded and config.person_analysis_max_fps == 0
    assert path.read_text() == original


@pytest.mark.parametrize('analysis_max_fps', [-1, -.01, float('nan'), float('inf'),
                                         -float('inf'), True, False, '5', None, []])
def test_invalid_analysis_cap_is_rejected_without_rewriting_settings(tmp_path, analysis_max_fps):
    with pytest.raises(PersonConfigError, match='person_analysis_max_fps'):
        AppConfig(workspace_path=str(tmp_path), person_analysis_max_fps=analysis_max_fps)

    path = tmp_path / 'config.json'
    original = json.dumps({'workspace_path': str(tmp_path),
                           'person_analysis_max_fps': analysis_max_fps})
    path.write_text(original)
    with pytest.raises(PersonConfigError, match='person_analysis_max_fps') as error:
        load_config(path)
    assert str(path) in str(error.value) and path.read_text() == original

    config = AppConfig(workspace_path=str(tmp_path), person_analysis_max_fps=5)
    save_config(config, path)
    original = path.read_bytes()
    config.person_analysis_max_fps = analysis_max_fps
    with pytest.raises(PersonConfigError, match='person_analysis_max_fps'):
        save_config(config, path)
    assert path.read_bytes() == original


def test_failed_settings_replacement_preserves_file_and_removes_temporary_file(tmp_path, monkeypatch):
    config = AppConfig(workspace_path=str(tmp_path), person_analysis_max_fps=5)
    path = save_config(config)
    original = path.read_bytes()
    config.person_analysis_max_fps = 10

    def fail_replace(self, target):
        raise OSError('injected settings replacement failure')

    monkeypatch.setattr(Path, 'replace', fail_replace)
    with pytest.raises(OSError, match='injected settings replacement failure'):
        save_config(config)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob('config.json.*.tmp'))

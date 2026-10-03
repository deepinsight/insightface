"""The GUI owns capture; the SDK owns get/match/update. No models or devices."""
from pathlib import Path
import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from insightface.gui.core import person_analysis as module


def make_job(tmp_path, **changes):
    values = dict(model_name='cheetah_s', model_root=tmp_path / 'models', provider_choice='CPU',
                  source_kind='video', source=str(tmp_path / 'movie.mp4'),
                  references=(('张三', tmp_path / '张三.png'),))
    return module.PersonAnalysisJob(**{**values, **changes})


def start_runner(runner, **kwargs):
    outcome = SimpleNamespace(result=None, error=None)
    def run():
        try:
            outcome.result = runner.run(**kwargs)
        except BaseException as error:
            outcome.error = error
    outcome.thread = threading.Thread(target=run, daemon=True)
    outcome.thread.start()
    return outcome


def finish_runner(outcome):
    outcome.thread.join(3)
    assert not outcome.thread.is_alive(), 'runner did not stop after capture was unblocked'
    if outcome.error is not None:
        raise outcome.error
    return outcome.result


@pytest.fixture
def runtime(monkeypatch):
    state = dict(calls=[], frames=3, captures=[], apps=[], read_threads=set(), frames_seen=[],
                 started=threading.Event(), release=threading.Event())
    class Capture:
        def __init__(self):
            state['captures'].append(self)
            self.index = 0
            self.released = False
            self.reading = False
            self.image = np.zeros((30, 40, 3), np.uint8)
        def open(self, *args):
            state['open_args'] = args
            if state.get('open_error'):
                raise RuntimeError(state['open_error'])
            if state.get('reject_timeout_options') and len(args) == 3:
                return False
            return not state.get('open_failed', False)
        def isOpened(self):
            return True
        def set(self, *_):
            return True
        def read(self):
            state['read_threads'].add(threading.current_thread())
            self.reading = True
            try:
                if state.get('before_read'):
                    state['before_read'](self.index)
                if state.get('block'):
                    state['started'].set()
                    assert state['release'].wait(3), 'capture was not unblocked'
                if state.get('read_error'):
                    raise RuntimeError(state['read_error'])
                if self.index >= state['frames']:
                    return False, None
                self.image.fill(self.index)
                self.index += 1
                state['calls'].append('read')
                return True, self.image
            finally:
                self.reading = False
        def get(self, prop):
            if prop == module.cv2.CAP_PROP_FPS:
                return state.get('video_fps', 25.)
            if prop == module.cv2.CAP_PROP_POS_MSEC:
                if 'video_pts' in state:
                    return state['video_pts'][self.index - 1]
                return (self.index - 1) * 1000 / state.get('video_fps', 25.)
            return 0.
        def release(self):
            assert not self.reading
            self.released = True
            state['calls'].append('release')
            if state.get('after_release'):
                state['after_release']()
    class SDK:
        def __init__(self, **options):
            assert set(options) == {'name', 'root', 'providers', 'config'}
            self.options = options
            self.closed = False
            state['apps'].append(self)
        def prepare(self):
            state['calls'].append('prepare')
            if state.get('prepare_error'):
                raise RuntimeError(state['prepare_error'])
        def register(self, name, path):
            state['calls'].append('register')
            state['registration'] = name, path
            return state.get('registration_result', SimpleNamespace(accepted=1, rejected=[]))
        def get(self, frame):
            state['calls'].append('get')
            state['frames_seen'].append(int(frame[0, 0, 0]))
            if state.get('on_get'):
                state['on_get'](frame)
            if state.get('get_error'):
                raise RuntimeError(state['get_error'])
            face = SimpleNamespace(bbox=np.array([2, 3, 10, 12]), embedding=None)
            return [SimpleNamespace(body_bbox=None, face=face)]
        def match(self, persons):
            state['calls'].append('match')
            state['matches'] = [SimpleNamespace(observation=persons[0], person_id='张三',
                                               matched_by='face', similarity=.9)]
            if state.get('after_match'):
                state['after_match']()
            return state['matches']
        def update(self, matches):
            state['calls'].append('update')
            assert matches is state['matches']
            if state.get('update_error'):
                raise RuntimeError(state['update_error'])
            return SimpleNamespace(added=1, replaced=0, skipped=0)
        def close(self):
            assert all(capture.released and not capture.reading for capture in state['captures'])
            assert all(thread is threading.current_thread() or not thread.is_alive()
                       for thread in state['read_threads'])
            self.closed = True
            state['calls'].append('close')
    def ensure(*args, **kwargs):
        state['calls'].append('ensure')
        if state.get('ensure_error'):
            raise RuntimeError(state['ensure_error'])
    monkeypatch.setattr(module.cv2, 'VideoCapture', Capture)
    monkeypatch.setattr(module, 'PersonAnalysis', SDK)
    monkeypatch.setattr(module, 'ensure_person_model', ensure)
    monkeypatch.setattr(module, 'person_model_providers', lambda _: ['CPUExecutionProvider'])
    return state


@pytest.fixture
def live_clock(monkeypatch, runtime):
    """Drive the real reader's waits with deterministic capture publications.

    Existing event-based tests exercise the capture thread. Here only its
    publication/condition wakeups are simulated, so no frame-rate test sleeps.
    """
    origin = 100_000_000_000
    utc_origin = 1_750_000_000_000
    clock = SimpleNamespace(ticks=origin, frame_times=[], next_frame=0, waits=[],
                            starts=[], inference_durations=[], readers=[],
                            cancel_at=None, cancel_action=None)

    def monotonic():
        return clock.ticks / 1_000_000_000

    def publish_due(reader):
        while (clock.next_frame < len(clock.frame_times)
               and clock.frame_times[clock.next_frame] <= clock.ticks):
            capture_at = clock.frame_times[clock.next_frame]
            ok, frame = reader._capture.read()
            assert ok
            reader._latest = (frame.copy(), utc_origin + (capture_at - origin) // 1_000_000)
            clock.next_frame += 1

    def advance(seconds):
        clock.ticks += round(seconds * 1_000_000_000)
        for reader in clock.readers:
            publish_due(reader)

    class ScheduledCondition:
        def __init__(self, reader):
            self.reader = reader

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def wait(self, timeout):
            assert 0 < timeout <= .05, 'cancellation polling exceeded 50 ms'
            clock.waits.append(timeout)
            assert len(clock.waits) < 1000, 'reader never reached its next frame or cancellation'
            target = clock.ticks + max(1, math.ceil(timeout * 1_000_000_000))
            if clock.next_frame < len(clock.frame_times):
                target = min(target, clock.frame_times[clock.next_frame])
            if clock.cancel_at is not None:
                target = min(target, clock.cancel_at)
            clock.ticks = target
            publish_due(self.reader)
            if clock.cancel_at is not None and clock.ticks >= clock.cancel_at:
                clock.cancel_at = None
                clock.cancel_action()

    class ScheduledReader(module._LatestFrameReader):
        def __init__(self, capture):
            super().__init__(capture)
            self._condition = ScheduledCondition(self)
            clock.readers.append(self)

        def start(self):
            publish_due(self)

        def close(self):
            self._stopped = True
            self._latest = None
            self._capture.release()

    def configure(frame_times, inference_durations=()):
        clock.frame_times = [origin + round(seconds * 1_000_000_000) for seconds in frame_times]
        clock.inference_durations = inference_durations
        runtime['frames'] = len(clock.frame_times)

    def on_get(_):
        clock.starts.append(monotonic())
        index = len(clock.starts) - 1
        if index < len(clock.inference_durations):
            advance(clock.inference_durations[index])

    def cancel_after(seconds, action):
        clock.cancel_at = origin + round(seconds * 1_000_000_000)
        clock.cancel_action = action

    clock.configure = configure
    clock.cancel_after = cancel_after
    clock.utc_origin = utc_origin
    runtime['on_get'] = on_get
    monkeypatch.setattr(module, '_LatestFrameReader', ScheduledReader)
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=monotonic))
    return clock


def test_video_pipeline_registers_photo_and_processes_in_order(tmp_path, runtime):
    runner = module.PersonAnalysisRunner(make_job(tmp_path))
    result = runner.run()
    assert result['status'] == 'completed' and result['frames_processed'] == 3
    assert runtime['registration'] == ('张三', tmp_path / '张三.png')
    assert runtime['calls'] == ['ensure', 'prepare', 'register'] + ['read', 'get', 'match', 'update'] * 3 + ['release', 'close']
    assert runtime['frames_seen'] == [0, 1, 2]
    assert result['registered'] == 1 and result['body_references_added'] == 3
    assert not list(tmp_path.rglob('*.rocksdb'))
    with pytest.raises(RuntimeError, match='new PersonAnalysisRunner'):
        runner.run()


@pytest.mark.parametrize('body_det_size,face_det_size', [(0, 0), (320, 160), (640, 320)])
def test_runner_forwards_independent_detector_sizes_in_sdk_configuration(
        tmp_path, runtime, body_det_size, face_det_size):
    runner = module.PersonAnalysisRunner(make_job(
        tmp_path, person_config={'body_det_size': body_det_size, 'face_det_size': face_det_size}))
    assert runner.run()['status'] == 'completed'
    assert runtime['apps'][0].options['config'].body_det_size == body_det_size
    assert runtime['apps'][0].options['config'].face_det_size == face_det_size
    assert runtime['calls'].count('prepare') == 1


def test_disabled_update_does_not_modify_references_and_preview_is_bounded(tmp_path, runtime):
    runner = module.PersonAnalysisRunner(make_job(tmp_path, auto_update=False))
    result = runner.run()
    assert 'update' not in runtime['calls']
    assert result['body_references_added'] == 0 and result['previews_replaced'] == 2
    runtime['captures'][0].image.fill(255)
    runtime['matches'][0].observation.face.bbox[:] = 999
    image, preview = runner.take_preview()
    assert image[0, 0, 0] == 2 and preview.frame_index == 3 and preview.timestamp == 80
    assert preview.matches[0].observation.face.bbox.tolist() == [2, 3, 10, 12]
    assert runner.take_preview() is None


@pytest.mark.parametrize('kind,source', [('camera', 0), ('rtsp', 'rtsp://host.test/live')])
def test_live_source_uses_same_pipeline_and_stops_between_frames(tmp_path, runtime, kind, source):
    runner = module.PersonAnalysisRunner(make_job(tmp_path, source_kind=kind, source=source, references=()))
    runtime['after_match'] = runner.request_stop
    result = runner.run()
    assert result['status'] == 'stopped' and result['frames_processed'] == 1
    _, preview = runner.take_preview()
    assert preview.time_basis == 'utc' and preview.timestamp > 0
    assert runtime['captures'][0].released and runtime['apps'][0].closed
    if kind == 'rtsp':
        assert runtime['open_args'] == (source, module.cv2.CAP_FFMPEG,
            [module.cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, module.cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])


@pytest.mark.parametrize('kind,source', [('camera', 0), ('rtsp', 'rtsp://host.test/live')])
@pytest.mark.parametrize('wait_for_capture_release', [False, True],
                         ids=['natural_order', 'release_during_inference'])
def test_live_capture_replaces_pending_frames_during_slow_inference(
        tmp_path, runtime, monkeypatch, kind, source, wait_for_capture_release):
    first_inference = threading.Event()
    pending_frames_read = threading.Event()
    release_last_read = threading.Event()
    capture_released = threading.Event()
    runtime['frames'] = 4
    clock = {'ns': 1_750_000_000_000_000_000}
    captured_at = [clock['ns'] + index * 10_000_000 for index in range(4)]
    monkeypatch.setattr(module.time, 'time_ns', lambda: clock['ns'])
    runner = module.PersonAnalysisRunner(make_job(tmp_path, source_kind=kind, source=source))

    def before_read(index):
        if index == 1:
            assert first_inference.wait(3), 'first frame did not reach inference'
        if index == 4:
            # Reaching the next native read proves all earlier frames were published.
            pending_frames_read.set()
            assert release_last_read.wait(3), 'last read was not unblocked'
        else:
            clock['ns'] = captured_at[index]

    def on_get(frame):
        if len(runtime['frames_seen']) == 1:
            first_inference.set()
            try:
                assert pending_frames_read.wait(3), 'capture stopped draining during inference'
                # The fake backend reuses its ndarray for every read.
                assert np.all(frame == 0), 'capture mutated the frame being inferred'
                clock['ns'] = captured_at[-1] + 9_000_000_000
            finally:
                release_last_read.set()
            if wait_for_capture_release:
                # EOF may release capture while this copied frame is still being analyzed.
                assert capture_released.wait(3), 'capture did not release after EOF'

    def after_match():
        if len(runtime['frames_seen']) == 2:
            runner.request_stop()

    runtime.update(before_read=before_read, on_get=on_get, after_match=after_match,
                   after_release=capture_released.set)
    result = runner.run()
    assert result['status'] == 'stopped' and result['frames_processed'] == 2
    assert runtime['frames_seen'] == [0, 3]
    assert threading.current_thread() not in runtime['read_threads']
    image, preview = runner.take_preview()
    assert np.all(image == 3)
    assert preview.frame_index == 2 and preview.time_basis == 'utc'
    assert preview.timestamp == captured_at[3] // 1_000_000
    assert [call for call in runtime['calls'] if call in ('release', 'close')] == ['release', 'close']
    assert runtime['calls'][-1] == 'close'


@pytest.mark.parametrize('kind,source', [('camera', 0), ('rtsp', 'rtsp://host.test/live')])
def test_live_five_fps_input_under_fifteen_fps_cap_processes_each_new_frame_once(
        tmp_path, runtime, live_clock, kind, source):
    live_clock.configure([0, .2, .4, .6])
    runner = module.PersonAnalysisRunner(make_job(
        tmp_path, source_kind=kind, source=source, analysis_max_fps=15))
    # Continue past the final frame's deadline: a cap must never repeat it.
    live_clock.cancel_after(.75, runner.request_stop)
    result = runner.run()
    assert result['status'] == 'stopped' and result['frames_processed'] == 4
    assert runtime['frames_seen'] == [0, 1, 2, 3]
    assert live_clock.starts == pytest.approx([100, 100.2, 100.4, 100.6])
    assert len(live_clock.waits) > 0


@pytest.mark.parametrize('kind,source', [('camera', 0), ('rtsp', 'rtsp://host.test/live')])
def test_live_five_fps_cap_takes_newest_frame_after_wait_without_queueing(
        tmp_path, runtime, live_clock, kind, source):
    live_clock.configure([index * .05 for index in range(17)])
    runner = module.PersonAnalysisRunner(make_job(
        tmp_path, source_kind=kind, source=source, analysis_max_fps=5))
    runtime['after_match'] = lambda: runner.request_stop() if len(runtime['frames_seen']) == 4 else None
    result = runner.run()
    assert result['status'] == 'stopped' and result['frames_processed'] == 4
    assert runtime['frames_seen'] == [0, 4, 8, 12]
    assert live_clock.starts == pytest.approx([100, 100.2, 100.4, 100.6])
    assert all(later - earlier >= .2 - 1e-8
               for earlier, later in zip(live_clock.starts, live_clock.starts[1:]))
    image, preview = runner.take_preview()
    assert np.all(image == 12) and preview.time_basis == 'utc'
    assert preview.timestamp == live_clock.utc_origin + 600
    assert runtime['captures'][0].index == 13
    assert runtime['calls'][-2:] == ['release', 'close']


@pytest.mark.parametrize('kind,source', [('camera', 0), ('rtsp', 'rtsp://host.test/live')])
def test_live_inference_overrun_runs_immediately_then_resumes_cap_from_actual_start(
        tmp_path, runtime, live_clock, kind, source):
    live_clock.configure([index * .05 for index in range(17)], [.35, .01, .1])
    runner = module.PersonAnalysisRunner(make_job(
        tmp_path, source_kind=kind, source=source, analysis_max_fps=5))
    runtime['after_match'] = lambda: runner.request_stop() if len(runtime['frames_seen']) == 3 else None
    result = runner.run()
    assert result['status'] == 'stopped' and result['frames_processed'] == 3
    assert runtime['frames_seen'] == [0, 7, 11]
    # The 350 ms inference already consumed its 200 ms interval. The following
    # short inference must still wait until 550 ms, without catch-up bursts.
    assert live_clock.starts == pytest.approx([100, 100.35, 100.55])
    image, preview = runner.take_preview()
    assert np.all(image == 11)
    assert preview.timestamp == live_clock.utc_origin + 550


@pytest.mark.parametrize('kind,source', [('camera', 0), ('rtsp', 'rtsp://host.test/live')])
@pytest.mark.parametrize('cancel_via', ['request_stop', 'callback'])
def test_stop_interrupts_live_throttle_before_next_analysis(
        tmp_path, runtime, live_clock, kind, source, cancel_via):
    live_clock.configure([index * .025 for index in range(20)])
    runner = module.PersonAnalysisRunner(make_job(
        tmp_path, source_kind=kind, source=source, analysis_max_fps=.1))
    cancel = threading.Event()
    live_clock.cancel_after(.125, runner.request_stop if cancel_via == 'request_stop' else cancel.set)
    result = runner.run(is_cancelled=cancel.is_set)
    assert result['status'] == 'stopped' and result['frames_processed'] == 1
    assert runtime['frames_seen'] == [0] and runtime['captures'][0].index > 1
    assert live_clock.ticks / 1_000_000_000 <= 100.175
    assert max(live_clock.waits) <= .05
    assert runtime['captures'][0].released and runtime['apps'][0].closed


@pytest.fixture
def video_runtime(runtime, monkeypatch):
    def reject_live_reader(*_):
        pytest.fail('video must keep its sequential capture path')
    monkeypatch.setattr(module, '_LatestFrameReader', reject_live_reader)
    # Media-time sampling must complete with a stationary wall clock and no
    # sleep API: it selects decoded frames without pacing video playback.
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: 100.))
    return runtime


@pytest.mark.parametrize('native_fps,analysis_fps,frames,expected', [
    (30, 5, 31, [0, 6, 12, 18, 24, 30]),
    (5, 15, 6, [0, 1, 2, 3, 4, 5]),
    (30, 0, 31, list(range(31))),
])
def test_video_samples_media_time_without_sleeping_or_repeating_frames(
        tmp_path, video_runtime, native_fps, analysis_fps, frames, expected):
    runtime = video_runtime
    runtime.update(video_fps=native_fps, frames=frames)
    runner = module.PersonAnalysisRunner(make_job(tmp_path, analysis_max_fps=analysis_fps))
    result = runner.run()
    assert result['status'] == 'completed' and result['frames_processed'] == len(expected)
    assert runtime['frames_seen'] == expected
    assert runtime['captures'][0].index == frames
    assert runtime['read_threads'] == {threading.current_thread()}
    assert runtime['calls'].count('match') == runtime['calls'].count('update') == len(expected)
    image, preview = runner.take_preview()
    assert np.all(image == expected[-1])
    assert preview.time_basis == 'video' and preview.timestamp == 1000
    assert runtime['captures'][0].released and runtime['apps'][0].closed


def test_video_variable_frame_timestamps_take_precedence_over_nominal_fps(tmp_path, video_runtime):
    runtime = video_runtime
    pts = [0, 40, 210, 220, 260, 425, 600, 630, 900]
    runtime.update(frames=len(pts), video_fps=100, video_pts=pts)
    runner = module.PersonAnalysisRunner(make_job(tmp_path, analysis_max_fps=5))
    assert runner.run()['frames_processed'] == 5
    assert runtime['frames_seen'] == [0, 2, 5, 7, 8]
    _, preview = runner.take_preview()
    assert preview.timestamp == 900


def test_video_rate_boundary_tolerates_tiny_timestamp_rounding_error(tmp_path, video_runtime):
    runtime = video_runtime
    runtime.update(frames=3, video_fps=30, video_pts=[0, 199.9999995, 400])
    runner = module.PersonAnalysisRunner(make_job(tmp_path, analysis_max_fps=5))
    assert runner.run()['frames_processed'] == 3
    assert runtime['frames_seen'] == [0, 1, 2]


@pytest.mark.parametrize('pts_value', [float('nan'), float('inf'), -1, 0])
def test_video_missing_or_stuck_timestamps_fall_back_to_native_fps(tmp_path, video_runtime, pts_value):
    runtime = video_runtime
    runtime.update(frames=16, video_fps=25, video_pts=[pts_value] * 16)
    runner = module.PersonAnalysisRunner(make_job(tmp_path, analysis_max_fps=5))
    assert runner.run()['frames_processed'] == 4
    assert runtime['frames_seen'] == [0, 5, 10, 15]
    _, preview = runner.take_preview()
    assert preview.time_basis == 'video' and preview.timestamp == 600


@pytest.mark.parametrize('later_pts', [1040, 1020])
def test_video_stuck_or_backwards_timestamps_fallback_preserves_media_offset(
        tmp_path, video_runtime, later_pts):
    runtime = video_runtime
    runtime.update(frames=16, video_fps=25, video_pts=[1000, 1040] + [later_pts] * 14)
    runner = module.PersonAnalysisRunner(make_job(tmp_path, analysis_max_fps=5))
    assert runner.run()['frames_processed'] == 4
    assert runtime['frames_seen'] == [0, 5, 10, 15]
    _, preview = runner.take_preview()
    assert preview.timestamp == 1600


@pytest.mark.parametrize('native_fps', [0, -1, float('nan'), float('inf')])
def test_video_valid_timestamps_do_not_require_native_fps(tmp_path, video_runtime, native_fps):
    runtime = video_runtime
    runtime.update(frames=6, video_fps=native_fps, video_pts=[0, 50, 100, 200, 250, 400])
    runner = module.PersonAnalysisRunner(make_job(tmp_path, analysis_max_fps=5))
    assert runner.run()['frames_processed'] == 3
    assert runtime['frames_seen'] == [0, 3, 5]
    _, preview = runner.take_preview()
    assert preview.timestamp == 400


@pytest.mark.parametrize('pts,expected_frames', [
    ([float('nan')] * 3, []),
    ([0, 0, 0], [0]),
])
def test_video_positive_rate_fails_clearly_when_media_timing_is_unavailable(
        tmp_path, video_runtime, pts, expected_frames):
    runtime = video_runtime
    runtime.update(frames=3, video_fps=0, video_pts=pts)
    runner = module.PersonAnalysisRunner(make_job(tmp_path, analysis_max_fps=5))
    with pytest.raises(RuntimeError, match='(?i)(timestamp|timing|fps|frame rate)'):
        runner.run()
    assert runtime['frames_seen'] == expected_frames
    assert runner.snapshot()['status'] == 'failed'
    assert runtime['captures'][0].released and runtime['apps'][0].closed


def test_video_auto_processes_all_frames_even_when_media_timing_is_unavailable(tmp_path, video_runtime):
    runtime = video_runtime
    runtime.update(frames=3, video_fps=0, video_pts=[float('nan')] * 3)
    runner = module.PersonAnalysisRunner(make_job(tmp_path, analysis_max_fps=0))
    result = runner.run()
    assert result['status'] == 'completed' and result['frames_processed'] == 3
    assert runtime['frames_seen'] == [0, 1, 2]
    _, preview = runner.take_preview()
    assert preview.time_basis == 'video' and preview.timestamp is None


@pytest.mark.parametrize('cancel_via', ['request_stop', 'callback'])
def test_video_cancellation_is_checked_while_skipping_frames(tmp_path, video_runtime, cancel_via):
    runtime = video_runtime
    runtime.update(frames=60, video_fps=30)
    runner = module.PersonAnalysisRunner(make_job(tmp_path, analysis_max_fps=.1))
    cancel = threading.Event()

    def before_read(index):
        if index == 4:
            runner.request_stop() if cancel_via == 'request_stop' else cancel.set()

    runtime['before_read'] = before_read
    result = runner.run(is_cancelled=cancel.is_set)
    assert result['status'] == 'stopped' and result['frames_processed'] == 1
    assert runtime['frames_seen'] == [0]
    assert runtime['captures'][0].index == 5
    assert runtime['captures'][0].released and runtime['apps'][0].closed


@pytest.mark.parametrize('kind,source', [('camera', 0), ('rtsp', 'rtsp://host.test/live')])
@pytest.mark.parametrize('cancel_via', ['request_stop', 'callback'])
def test_stop_waiting_for_first_frame_does_not_release_native_read(tmp_path, runtime, kind, source, cancel_via):
    runtime['block'] = True
    runner = module.PersonAnalysisRunner(make_job(tmp_path, source_kind=kind, source=source))
    cancel = threading.Event()
    cancellation_observed = threading.Event()
    def is_cancelled():
        if cancel.is_set():
            cancellation_observed.set()
            return True
        return False
    outcome = start_runner(runner, is_cancelled=is_cancelled)
    try:
        assert runtime['started'].wait(3)
        runner.request_stop() if cancel_via == 'request_stop' else cancel.set()
        assert not runtime['apps'][0].closed and not runtime['captures'][0].released
        assert outcome.thread.is_alive()
        if cancel_via == 'callback':
            assert cancellation_observed.wait(3), 'cancellation was not checked while waiting'
    finally:
        runner.request_stop()
        runtime['release'].set()
        outcome.thread.join(3)
    result = finish_runner(outcome)
    assert result['status'] == 'stopped' and result['frames_processed'] == 0
    assert runtime['apps'][0].closed and runtime['captures'][0].released


@pytest.mark.parametrize('kind,source', [('camera', 0), ('rtsp', 'rtsp://host.test/live')])
def test_stop_during_inference_waits_for_native_read_before_closing_sdk(tmp_path, runtime, kind, source):
    inference_started = threading.Event()
    inference_finished = threading.Event()
    release_inference = threading.Event()
    read_started = threading.Event()
    release_read = threading.Event()
    runtime['frames'] = 1
    runner = module.PersonAnalysisRunner(make_job(tmp_path, source_kind=kind, source=source))

    def before_read(index):
        if index == 1:
            assert inference_started.wait(3)
            read_started.set()
            assert release_read.wait(3), 'native read was not unblocked'

    def on_get(_):
        inference_started.set()
        assert release_inference.wait(3), 'inference was not unblocked'
        inference_finished.set()

    runtime.update(before_read=before_read, on_get=on_get)
    outcome = start_runner(runner)
    try:
        assert read_started.wait(3), 'capture did not read while inference was running'
        runner.request_stop()
        assert not runtime['captures'][0].released and not runtime['apps'][0].closed
        release_inference.set()
        assert inference_finished.wait(3)
        assert outcome.thread.is_alive()
        assert not runtime['captures'][0].released and not runtime['apps'][0].closed
    finally:
        runner.request_stop()
        release_inference.set()
        release_read.set()
        outcome.thread.join(3)
    result = finish_runner(outcome)
    assert result['status'] == 'stopped'
    assert runtime['frames_seen'] == [0]
    assert [call for call in runtime['calls'] if call in ('release', 'close')] == ['release', 'close']
    assert runtime['calls'][-1] == 'close'


@pytest.mark.parametrize('kind,source', [('camera', 0), ('rtsp', 'rtsp://host.test/live')])
@pytest.mark.parametrize('ending', ['eof', 'error'])
def test_live_pending_frame_is_processed_before_capture_failure(tmp_path, runtime, kind, source, ending):
    first_inference = threading.Event()
    failed_read_started = threading.Event()
    runner = module.PersonAnalysisRunner(make_job(tmp_path, source_kind=kind, source=source))

    def before_read(index):
        if index == 1:
            assert first_inference.wait(3)
        if index == 3:
            if ending == 'error':
                runtime['read_error'] = 'injected live capture failure'
            failed_read_started.set()

    def on_get(_):
        if len(runtime['frames_seen']) == 1:
            first_inference.set()
            assert failed_read_started.wait(3), 'capture did not drain pending frames'

    runtime.update(before_read=before_read, on_get=on_get)
    message = 'injected live capture failure' if ending == 'error' else 'stopped delivering'
    with pytest.raises(RuntimeError, match=message):
        runner.run()
    assert runtime['frames_seen'] == [0, 2]
    snapshot = runner.snapshot()
    assert snapshot['status'] == 'failed' and snapshot['frames_processed'] == 2
    image, preview = runner.take_preview()
    assert np.all(image == 2) and preview.frame_index == 2
    assert runtime['captures'][0].released and runtime['apps'][0].closed


def test_local_camera_can_open_when_timeout_options_are_unsupported(tmp_path, runtime):
    runtime['reject_timeout_options'] = True
    runner = module.PersonAnalysisRunner(make_job(tmp_path, source_kind='camera', source=0))
    runtime['after_match'] = runner.request_stop
    assert runner.run()['status'] == 'stopped'
    assert len(runtime['captures']) == 2 and all(x.released for x in runtime['captures'])
    assert len(runtime['open_args']) == 2


def test_rtsp_open_failure_never_retries_without_timeout(tmp_path, runtime):
    runtime['reject_timeout_options'] = True
    runner = module.PersonAnalysisRunner(make_job(tmp_path, source_kind='rtsp', source='rtsp://host.test/live'))
    with pytest.raises(RuntimeError, match='Unable to open'):
        runner.run()
    assert len(runtime['captures']) == 1 and runtime['captures'][0].released
    assert len(runtime['open_args']) == 3 and runtime['apps'][0].closed


@pytest.mark.parametrize('kind', ['video', 'camera', 'rtsp'])
@pytest.mark.parametrize('stage', ['prepare', 'open', 'read', 'get', 'update'])
def test_failures_release_resources_and_remain_visible(tmp_path, runtime, stage, kind):
    runtime[stage + '_error'] = 'injected failure'
    changes = dict(source_kind=kind)
    if kind != 'video':
        changes['source'] = 0 if kind == 'camera' else 'rtsp://host.test/live'
    runner = module.PersonAnalysisRunner(make_job(tmp_path, **changes))
    with pytest.raises(RuntimeError, match='injected failure'):
        runner.run()
    assert runtime['apps'][0].closed and all(x.released for x in runtime['captures'])
    assert runner.snapshot()['status'] == 'failed'


def test_rtsp_error_redacts_url_and_separately_quoted_credentials(tmp_path, runtime):
    source = 'rtsp://operator:pass%20word@host.test/live?token=mysecret'
    runtime['read_error'] = f'failed {source}; operator pass word mysecret'
    runner = module.PersonAnalysisRunner(make_job(tmp_path, source_kind='rtsp', source=source))
    with pytest.raises(RuntimeError) as error:
        runner.run()
    message = str(error.value)
    for secret in ('operator', 'pass%20word', 'pass word', 'mysecret', 'host.test'):
        assert secret not in message and secret not in str(runner.snapshot())
    assert runner.snapshot()['status'] == 'failed'
    assert runtime['captures'][0].released and runtime['apps'][0].closed


def test_camera_eof_is_reported_as_disconnection(tmp_path, runtime):
    runtime['frames'] = 0
    runner = module.PersonAnalysisRunner(make_job(tmp_path, source_kind='camera', source=0))
    with pytest.raises(RuntimeError, match='stopped delivering'):
        runner.run()


def test_zero_frame_video_is_an_error_not_a_successful_demo(tmp_path, runtime):
    runtime['frames'] = 0
    runner = module.PersonAnalysisRunner(make_job(tmp_path))
    with pytest.raises(RuntimeError, match='No readable frames'):
        runner.run()
    assert runner.snapshot()['status'] == 'failed'
    assert runtime['apps'][0].closed and runtime['captures'][0].released


def test_registration_rejection_and_early_cancel(tmp_path, runtime):
    runtime['registration_result'] = SimpleNamespace(accepted=0, rejected=[{'reason': 'no_face'}])
    runner = module.PersonAnalysisRunner(make_job(tmp_path))
    with pytest.raises(RuntimeError, match='no_face'):
        runner.run()
    assert runtime['apps'][0].closed and not runtime['captures']
    runtime['calls'].clear()
    other = module.PersonAnalysisRunner(make_job(tmp_path))
    other.request_stop()
    assert other.run()['status'] == 'stopped' and not runtime['calls']


@pytest.mark.parametrize('changes', [{'source_kind': 'camera', 'source': True},
    {'source_kind': 'rtsp', 'source': 'https://example.test/live'}, {'source_kind': 'bad'},
    {'auto_update': 1}, {'person_config': {'face_interval_ms': 100}},
    {'person_config': {'body_det_size': 321}}, {'person_config': {'body_det_size': True}},
    {'person_config': {'face_det_size': 641}}, {'person_config': {'face_det_size': True}},
    {'references': ((' ', Path('photo.jpg')),)}])
def test_job_rejects_invalid_sources_and_old_options(tmp_path, changes):
    with pytest.raises(ValueError):
        make_job(tmp_path, **changes)


@pytest.mark.parametrize('analysis_max_fps', [-1, -.01, float('nan'), float('inf'),
                                         -float('inf'), True, False, '5', None, []])
def test_job_rejects_invalid_analysis_fps_caps(tmp_path, analysis_max_fps):
    with pytest.raises(ValueError, match='analysis_max_fps'):
        make_job(tmp_path, analysis_max_fps=analysis_max_fps)


@pytest.mark.parametrize('analysis_max_fps', [0, .25, 5, 15.5])
def test_job_accepts_auto_and_finite_positive_analysis_caps(tmp_path, analysis_max_fps):
    assert make_job(tmp_path, analysis_max_fps=analysis_max_fps).analysis_max_fps == analysis_max_fps


def test_job_analysis_cap_defaults_to_auto(tmp_path):
    assert make_job(tmp_path).analysis_max_fps == 0

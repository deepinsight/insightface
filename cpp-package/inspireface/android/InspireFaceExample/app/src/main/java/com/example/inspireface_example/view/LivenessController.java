package com.example.inspireface_example.view;

import android.content.Context;
import android.graphics.Color;
import android.os.SystemClock;
import android.text.SpannableStringBuilder;
import android.text.Spanned;
import android.text.style.ForegroundColorSpan;

import androidx.core.content.ContextCompat;

import com.example.inspireface_example.R;
import com.insightface.sdk.inspireface.InspireFace;
import com.insightface.sdk.inspireface.base.CustomParameter;
import com.insightface.sdk.inspireface.base.FaceInteractionsActions;
import com.insightface.sdk.inspireface.base.FaceRect;
import com.insightface.sdk.inspireface.base.ImageStream;
import com.insightface.sdk.inspireface.base.MultipleFaceData;
import com.insightface.sdk.inspireface.base.RGBLivenessConfidence;
import com.insightface.sdk.inspireface.base.Session;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Per-frame liveness logic for both modes. All {@link #onFrame} calls must come from the
 * single analysis thread; mode switches from the UI thread are serialized with
 * {@code synchronized}.
 */
final class LivenessController {

    enum Mode { SILENT, ACTION, POSE }

    /** Cooperative challenges the SDK can recognize. */
    enum ActionType {
        BLINK(R.string.action_blink, R.string.name_blink),
        SHAKE(R.string.action_shake, R.string.name_shake),
        JAW_OPEN(R.string.action_jaw_open, R.string.name_jaw_open),
        HEAD_RAISE(R.string.action_head_raise, R.string.name_head_raise);

        final int promptRes;
        final int nameRes;

        ActionType(int promptRes, int nameRes) {
            this.promptRes = promptRes;
            this.nameRes = nameRes;
        }
    }

    /** Snapshot the UI renders; immutable, built on the analysis thread. */
    static final class UiState {
        final String title;
        final CharSequence subtitle;   // null hides the row
        final int progress;            // 0..100, -1 hides the bar
        final boolean showRestart;
        final int boxColor;

        UiState(String title, CharSequence subtitle, int progress, boolean showRestart, int boxColor) {
            this.title = title;
            this.subtitle = subtitle;
            this.progress = progress;
            this.showRestart = showRestart;
            this.boxColor = boxColor;
        }

        boolean sameContent(UiState o) {
            return o != null
                    && progress == o.progress
                    && showRestart == o.showRestart
                    && boxColor == o.boxColor
                    && title.equals(o.title)
                    && (subtitle == null ? o.subtitle == null
                        : o.subtitle != null && subtitle.toString().equals(o.subtitle.toString()));
        }
    }

    // ---- Silent-mode tunables ----
    /** RGB anti-spoofing score above which a face counts as live (SDK author default). */
    private static final float RGB_LIVENESS_THRESHOLD = 0.88f;
    /** Frames averaged before giving a verdict; smooths single-frame noise. */
    private static final int SCORE_WINDOW = 8;
    private static final int MIN_VERDICT_SAMPLES = 3;
    /**
     * Run the anti-spoofing pipeline on every Nth tracked frame. Each call converts the
     * full frame internally (the SDK's own acknowledged hot spot), so skipping frames
     * buys preview smoothness at no visible cost given the score averaging.
     */
    private static final int SILENT_PIPELINE_INTERVAL = 2;

    // ---- Action-mode tunables ----
    private static final int ACTIONS_PER_RUN = 3;
    private static final long ACTION_TIMEOUT_MS = 8_000;
    /**
     * Consecutive good frames required before the challenge sequence starts. Must be
     * >= the SDK's 9-call action warm-up so flags are live for the first challenge.
     */
    private static final int STABLE_FRAMES_TO_START = 10;
    /**
     * Shake latches while both yaw extremes sit in the SDK's rolling 10-slot window, so
     * a single 0 read can still hide a half-shake residue from the previous step. Only
     * arm SHAKE after the window has provably flushed.
     */
    private static final int SHAKE_ARM_ZERO_FRAMES = 10;
    private static final long FACE_LOST_GRACE_MS = 800;
    private static final float MAX_START_YAW_DEG = 20f;
    private static final float MAX_START_PITCH_DEG = 15f;

    // ---- Pose-mode tunables ----
    /** Latest action shown large plus up to 5 smaller history entries. */
    private static final int POSE_HISTORY_MAX = 6;

    // ---- Shared gating ----
    /** Faces narrower than this fraction of the frame width are too far away to judge. */
    private static final float MIN_FACE_WIDTH_RATIO = 0.18f;

    private enum Phase { WAIT_FACE, CHALLENGE, PASSED, FAILED }

    private final Context context;
    private final Random random = new Random();

    private final CustomParameter silentParam = InspireFace.CreateCustomParameter().enableLiveness(true);
    private final CustomParameter actionParam = InspireFace.CreateCustomParameter().enableInteractionLiveness(true);

    private final int colorNeutral;
    private final int colorWarn;
    private final int colorFail;

    private Mode mode = Mode.SILENT;

    // Silent state
    private final ArrayDeque<Float> scoreWindow = new ArrayDeque<>();
    private int silentFrameCounter;
    private UiState lastSilentState;
    private int silentTrackId = -1;

    // Pose state
    private final ArrayDeque<ActionType> poseHistory = new ArrayDeque<>();
    private final int[] posePrevFlags = new int[ActionType.values().length];
    private int poseTrackId = -1;

    // Action state
    private Phase phase = Phase.WAIT_FACE;
    private final List<ActionType> sequence = new ArrayList<>();
    private int actionIndex;
    private long actionDeadline;
    /**
     * Edge trigger: jawOpen/headRaise stay 1 while the pose is held and shake latches for
     * ~10 pipeline calls, so a leftover from the previous step could complete the next one
     * instantly. Each step first has to observe the flag at 0 before a 1 counts —
     * SHAKE needs a full window of zeros (see SHAKE_ARM_ZERO_FRAMES).
     */
    private boolean actionArmed;
    private int armZeroStreak;
    private int stableFrames;
    private int lockedTrackId = -1;
    private long faceLostSince;
    private int failReasonRes;

    LivenessController(Context context) {
        // Deliberately the activity context, not the application one: per-app locales
        // (AppCompatDelegate) only wrap activity contexts below API 33, and every prompt
        // string is resolved through this reference. Lifetime matches the activity.
        this.context = context;
        colorNeutral = ContextCompat.getColor(context, R.color.liveness_accent);
        colorWarn = ContextCompat.getColor(context, R.color.liveness_warn);
        colorFail = ContextCompat.getColor(context, R.color.liveness_fail);
    }

    synchronized Mode getMode() {
        return mode;
    }

    synchronized void setMode(Mode newMode) {
        if (mode != newMode) {
            mode = newMode;
            reset();
        }
    }

    synchronized void restart() {
        reset();
    }

    private void reset() {
        clearSilentState();
        phase = Phase.WAIT_FACE;
        sequence.clear();
        actionIndex = 0;
        stableFrames = 0;
        lockedTrackId = -1;
        faceLostSince = 0;
        poseHistory.clear();
        resetPoseEdges();
    }

    /**
     * Consumes one tracked frame. Runs the pipeline appropriate for the current mode while
     * {@code stream} is still alive and returns what the UI should display.
     */
    synchronized UiState onFrame(Session session, ImageStream stream,
                                 MultipleFaceData faces, int uprightWidth, int uprightHeight) {
        switch (mode) {
            case SILENT:
                return onSilentFrame(session, stream, faces, uprightWidth);
            case POSE:
                return onPoseFrame(session, stream, faces);
            case ACTION:
            default:
                return onActionFrame(session, stream, faces, uprightWidth);
        }
    }

    // ------------------------------------------------------------------
    // Silent (RGB anti-spoofing) mode
    // ------------------------------------------------------------------

    private UiState onSilentFrame(Session session, ImageStream stream,
                                  MultipleFaceData faces, int uprightWidth) {
        if (faces.detectedNum == 0) {
            clearSilentState();
            return new UiState(str(R.string.msg_no_face), null, -1, false, colorNeutral);
        }
        if (faces.detectedNum > 1) {
            clearSilentState();
            return new UiState(str(R.string.msg_multiple_faces), null, -1, false, colorWarn);
        }
        int idx = largestFaceIndex(faces);
        if (!isFaceBigEnough(faces.rects[idx], uprightWidth)) {
            clearSilentState();
            return new UiState(str(R.string.msg_face_too_small), null, -1, false, colorWarn);
        }
        // A different track id means a different subject — never blend its scores or show
        // the previous subject's cached verdict.
        if (faces.trackIds[idx] != silentTrackId) {
            clearSilentState();
            silentTrackId = faces.trackIds[idx];
        }

        if (silentFrameCounter++ % SILENT_PIPELINE_INTERVAL != 0 && lastSilentState != null) {
            return lastSilentState;
        }
        if (!InspireFace.MultipleFacePipelineProcess(session, stream, faces, silentParam)) {
            return new UiState(str(R.string.silent_analyzing), null, -1, false, colorNeutral);
        }
        RGBLivenessConfidence liveness = InspireFace.GetRGBLivenessConfidence(session);
        if (liveness == null || liveness.num <= idx) {
            return new UiState(str(R.string.silent_analyzing), null, -1, false, colorNeutral);
        }

        scoreWindow.addLast(liveness.confidence[idx]);
        while (scoreWindow.size() > SCORE_WINDOW) {
            scoreWindow.removeFirst();
        }
        float sum = 0f;
        for (float v : scoreWindow) {
            sum += v;
        }
        float avg = sum / scoreWindow.size();
        String scoreText = str(R.string.silent_score, avg);
        int progress = Math.round(avg * 100);

        UiState state;
        if (scoreWindow.size() < MIN_VERDICT_SAMPLES) {
            state = new UiState(str(R.string.silent_analyzing), scoreText, progress, false, colorNeutral);
        } else {
            boolean live = avg >= RGB_LIVENESS_THRESHOLD;
            state = new UiState(
                    str(live ? R.string.silent_real : R.string.silent_fake),
                    scoreText, progress, false, live ? colorNeutral : colorFail);
        }
        lastSilentState = state;
        return state;
    }

    private void clearSilentState() {
        scoreWindow.clear();
        silentFrameCounter = 0;
        lastSilentState = null;
        silentTrackId = -1;
    }

    // ------------------------------------------------------------------
    // Cooperative action mode
    // ------------------------------------------------------------------

    private UiState onActionFrame(Session session, ImageStream stream,
                                  MultipleFaceData faces, int uprightWidth) {
        long now = SystemClock.elapsedRealtime();
        switch (phase) {
            case PASSED:
                return new UiState(str(R.string.action_passed), null, 100, true, colorNeutral);
            case FAILED:
                return new UiState(str(failReasonRes), null, -1, true, colorFail);
            case WAIT_FACE:
                return onWaitFace(session, stream, faces, uprightWidth);
            case CHALLENGE:
            default:
                return onChallenge(session, stream, faces, now);
        }
    }

    private UiState onWaitFace(Session session, ImageStream stream,
                               MultipleFaceData faces, int uprightWidth) {
        if (faces.detectedNum == 0) {
            stableFrames = 0;
            return new UiState(str(R.string.msg_no_face), null, -1, false, colorNeutral);
        }
        if (faces.detectedNum > 1) {
            stableFrames = 0;
            return new UiState(str(R.string.msg_multiple_faces), null, -1, false, colorWarn);
        }
        int idx = largestFaceIndex(faces);
        if (!isFaceBigEnough(faces.rects[idx], uprightWidth)) {
            stableFrames = 0;
            return new UiState(str(R.string.msg_face_too_small), null, -1, false, colorWarn);
        }
        // Pre-warm the SDK's 10-call action window during get-ready so the first
        // challenge is detectable the moment it is shown.
        InspireFace.MultipleFacePipelineProcess(session, stream, faces, actionParam);
        // angles[] is only trustworthy at index 0 — the 1.2.0 JNI writes face[0]'s angles
        // into every slot. We require exactly one face here, so idx is always 0.
        boolean frontal = Math.abs(faces.angles[idx].yaw) <= MAX_START_YAW_DEG
                && Math.abs(faces.angles[idx].pitch) <= MAX_START_PITCH_DEG;
        if (!frontal) {
            stableFrames = 0;
            return new UiState(str(R.string.action_get_ready), null, -1, false, colorWarn);
        }
        stableFrames++;
        if (stableFrames < STABLE_FRAMES_TO_START) {
            int progress = stableFrames * 100 / STABLE_FRAMES_TO_START;
            return new UiState(str(R.string.action_get_ready), null, progress, false, colorNeutral);
        }
        startChallenge(faces.trackIds[idx]);
        return challengeState(SystemClock.elapsedRealtime(), null);
    }

    private void startChallenge(int trackId) {
        List<ActionType> pool = new ArrayList<>(Arrays.asList(ActionType.values()));
        Collections.shuffle(pool, random);
        sequence.clear();
        sequence.addAll(pool.subList(0, Math.min(ACTIONS_PER_RUN, pool.size())));
        actionIndex = 0;
        lockedTrackId = trackId;
        faceLostSince = 0;
        armCurrentAction();
        phase = Phase.CHALLENGE;
    }

    private void armCurrentAction() {
        actionDeadline = SystemClock.elapsedRealtime() + ACTION_TIMEOUT_MS;
        actionArmed = false;
        armZeroStreak = 0;
    }

    private UiState onChallenge(Session session, ImageStream stream,
                                MultipleFaceData faces, long now) {
        int idx = indexOfTrackId(faces, lockedTrackId);
        if (idx < 0) {
            if (faceLostSince == 0) {
                faceLostSince = now;
            }
            if (now - faceLostSince > FACE_LOST_GRACE_MS) {
                return fail(R.string.action_face_lost);
            }
            return new UiState(str(R.string.msg_no_face), stepText(), -1, false, colorWarn);
        }
        faceLostSince = 0;

        if (now > actionDeadline) {
            return fail(R.string.action_failed_timeout);
        }

        if (InspireFace.MultipleFacePipelineProcess(session, stream, faces, actionParam)) {
            FaceInteractionsActions actions = InspireFace.GetFaceInteractionActionsResult(session);
            if (actions != null && actions.num > idx) {
                ActionType current = sequence.get(actionIndex);
                int flag = actionFlag(actions, idx, current);
                // normal==1 marks the SDK's warm-up (fresh track or post-blink window
                // reset) during which every flag reads a placeholder 0. Those zeros must
                // not open the gate, or a pose held through a blink would count as fresh.
                boolean warmingUp = actions.normal[idx] == 1;
                if (!actionArmed) {
                    // -1 means "not evaluated" — only a real 0 opens the gate.
                    armZeroStreak = !warmingUp && flag == 0 ? armZeroStreak + 1 : 0;
                    actionArmed = armZeroStreak
                            >= (current == ActionType.SHAKE ? SHAKE_ARM_ZERO_FRAMES : 1);
                } else if (flag > 0 && !warmingUp) {
                    actionIndex++;
                    if (actionIndex >= sequence.size()) {
                        phase = Phase.PASSED;
                        lockedTrackId = -1;
                        return new UiState(str(R.string.action_passed), null, 100, true, colorNeutral);
                    }
                    armCurrentAction();
                }
            }
        }
        String warn = faces.detectedNum > 1 ? str(R.string.msg_multiple_faces) : null;
        return challengeState(now, warn);
    }

    private UiState challengeState(long now, String warnOverride) {
        long msLeft = Math.max(0, actionDeadline - now);
        String subtitle = warnOverride != null
                ? warnOverride
                : stepText() + " · " + str(R.string.action_time_left, (int) ((msLeft + 999) / 1000));
        int progress = (int) (msLeft * 100 / ACTION_TIMEOUT_MS);
        return new UiState(str(sequence.get(actionIndex).promptRes), subtitle, progress, false, colorNeutral);
    }

    private String stepText() {
        return str(R.string.action_step, actionIndex + 1, sequence.size());
    }

    private UiState fail(int reasonRes) {
        phase = Phase.FAILED;
        failReasonRes = reasonRes;
        lockedTrackId = -1;
        return new UiState(str(reasonRes), null, -1, true, colorFail);
    }

    private static int actionFlag(FaceInteractionsActions actions, int idx, ActionType type) {
        switch (type) {
            case BLINK:
                return actions.blink[idx];
            case SHAKE:
                return actions.shake[idx];
            case JAW_OPEN:
                return actions.jawOpen[idx];
            case HEAD_RAISE:
            default:
                return actions.headRaise[idx];
        }
    }

    // ------------------------------------------------------------------
    // Pose recognition mode — display whatever action the user performs
    // ------------------------------------------------------------------

    private UiState onPoseFrame(Session session, ImageStream stream, MultipleFaceData faces) {
        if (faces.detectedNum == 0) {
            resetPoseEdges();
            poseTrackId = -1;
            // Without a face there is no "current" action — show the full history small.
            return new UiState(str(R.string.msg_no_face), poseHistoryLine(true), -1, false, colorNeutral);
        }
        if (faces.detectedNum > 1) {
            // Edge state is keyed to one face; following the "largest" of two similar
            // faces would flap and fabricate rising edges.
            return new UiState(str(R.string.msg_multiple_faces), poseHistoryLine(true), -1, false, colorWarn);
        }
        int idx = largestFaceIndex(faces);
        if (faces.trackIds[idx] != poseTrackId) {
            resetPoseEdges();
            poseTrackId = faces.trackIds[idx];
        }
        if (InspireFace.MultipleFacePipelineProcess(session, stream, faces, actionParam)) {
            FaceInteractionsActions actions = InspireFace.GetFaceInteractionActionsResult(session);
            // Skip the SDK warm-up (normal==1: fresh track or post-blink window reset):
            // its placeholder zeros would read as "pose released" and make a pose held
            // through a natural blink re-register as a duplicate entry.
            if (actions != null && actions.num > idx && actions.normal[idx] != 1) {
                for (ActionType type : ActionType.values()) {
                    int flag = actionFlag(actions, idx, type);
                    if (flag < 0) {
                        continue; // not evaluated this call — keep the previous edge state
                    }
                    // Rising edge only: level-triggered flags (jawOpen/headRaise) and the
                    // window-latched shake must return to 0 before they register again.
                    if (flag == 1 && posePrevFlags[type.ordinal()] == 0) {
                        poseHistory.addFirst(type);
                        while (poseHistory.size() > POSE_HISTORY_MAX) {
                            poseHistory.removeLast();
                        }
                    }
                    posePrevFlags[type.ordinal()] = flag;
                }
            }
        }
        String title = poseHistory.isEmpty()
                ? str(R.string.pose_hint)
                : str(poseHistory.peekFirst().nameRes);
        return new UiState(title, poseHistoryLine(false), -1, false, colorNeutral);
    }

    /** Edge-detector state only; the visible history is kept until reset/mode switch. */
    private void resetPoseEdges() {
        Arrays.fill(posePrevFlags, 0);
    }

    /**
     * History entries joined oldest-last with decreasing opacity, so past actions visually
     * fade out as new ones push them toward the tail.
     */
    private CharSequence poseHistoryLine(boolean includeHead) {
        int skip = includeHead ? 0 : 1;
        if (poseHistory.size() <= skip) {
            return null;
        }
        SpannableStringBuilder line = new SpannableStringBuilder();
        int i = 0;
        int shown = 0;
        int total = poseHistory.size() - skip;
        for (ActionType type : poseHistory) {
            if (i++ < skip) {
                continue;
            }
            if (line.length() > 0) {
                line.append("  ·  ");
            }
            int start = line.length();
            line.append(str(type.nameRes));
            int alpha = 230 - (total <= 1 ? 0 : 160 * shown / (total - 1));
            line.setSpan(new ForegroundColorSpan(Color.argb(alpha, 255, 255, 255)),
                    start, line.length(), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
            shown++;
        }
        return line;
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private static int largestFaceIndex(MultipleFaceData faces) {
        int best = 0;
        long bestArea = -1;
        for (int i = 0; i < faces.detectedNum; i++) {
            FaceRect r = faces.rects[i];
            long area = (long) r.width * r.height;
            if (area > bestArea) {
                bestArea = area;
                best = i;
            }
        }
        return best;
    }

    private static int indexOfTrackId(MultipleFaceData faces, int trackId) {
        for (int i = 0; i < faces.detectedNum; i++) {
            if (faces.trackIds[i] == trackId) {
                return i;
            }
        }
        return -1;
    }

    private static boolean isFaceBigEnough(FaceRect rect, int uprightWidth) {
        return rect.width >= uprightWidth * MIN_FACE_WIDTH_RATIO;
    }

    private String str(int res, Object... args) {
        return args.length == 0 ? context.getString(res) : context.getString(res, args);
    }
}

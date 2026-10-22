package com.example.inspireface_example.view;

import android.content.Context;
import android.util.Size;

import androidx.annotation.StringRes;
import androidx.camera.core.CameraInfoUnavailableException;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.Preview;
import androidx.camera.core.resolutionselector.AspectRatioStrategy;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.example.inspireface_example.R;
import com.google.common.util.concurrent.ListenableFuture;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;

/**
 * Reusable CameraX preview + analysis component shared by live demos and enrollment.
 * It owns lens fallback/switching and keeps both use cases on the same 4:3 crop.
 */
public final class CameraPreviewController {

    public interface Listener {
        /** Initial lens is ready; false means the device fell back to the rear camera. */
        void onCameraReady(boolean frontCamera);

        /** Called only after a user-requested lens switch succeeds. */
        void onLensChanged(boolean frontCamera);

        void onCameraError(@StringRes int messageRes);
    }

    private final Context context;
    private final LifecycleOwner lifecycleOwner;
    private final PreviewView previewView;
    private final Executor analysisExecutor;
    private final ImageAnalysis.Analyzer analyzer;
    private final Listener listener;

    private ProcessCameraProvider cameraProvider;
    private Preview preview;
    private ImageAnalysis analysis;
    private boolean useFrontCamera = true;
    private boolean stopped;

    public CameraPreviewController(Context context, LifecycleOwner lifecycleOwner,
                                   PreviewView previewView, Executor analysisExecutor,
                                   ImageAnalysis.Analyzer analyzer, Listener listener) {
        this.context = context.getApplicationContext();
        this.lifecycleOwner = lifecycleOwner;
        this.previewView = previewView;
        this.analysisExecutor = analysisExecutor;
        this.analyzer = analyzer;
        this.listener = listener;
    }

    /** Asynchronously acquires CameraX and binds a front camera, with rear fallback. */
    public void start() {
        start(true);
    }

    /** Asynchronously acquires CameraX and binds the requested lens, with opposite fallback. */
    public void start(boolean preferFrontCamera) {
        stopped = false;
        useFrontCamera = preferFrontCamera;
        ListenableFuture<ProcessCameraProvider> future =
                ProcessCameraProvider.getInstance(context);
        future.addListener(() -> {
            if (stopped) {
                return;
            }
            try {
                cameraProvider = future.get();
            } catch (ExecutionException e) {
                listener.onCameraError(R.string.msg_engine_failed);
                return;
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                listener.onCameraError(R.string.msg_engine_failed);
                return;
            }
            CameraSelector requested = useFrontCamera
                    ? CameraSelector.DEFAULT_FRONT_CAMERA
                    : CameraSelector.DEFAULT_BACK_CAMERA;
            if (!hasCamera(requested)) {
                CameraSelector fallback = useFrontCamera
                        ? CameraSelector.DEFAULT_BACK_CAMERA
                        : CameraSelector.DEFAULT_FRONT_CAMERA;
                if (hasCamera(fallback)) {
                    useFrontCamera = !useFrontCamera;
                } else {
                    listener.onCameraError(R.string.msg_no_front_camera);
                    return;
                }
            }
            createUseCases();
            if (bindCurrentLens()) {
                listener.onCameraReady(useFrontCamera);
            }
        }, ContextCompat.getMainExecutor(context));
    }

    /** Returns false if CameraX is not ready or the opposite lens does not exist. */
    public boolean flipCamera() {
        if (cameraProvider == null || stopped) {
            return false;
        }
        boolean targetFront = !useFrontCamera;
        if (!hasCamera(targetFront ? CameraSelector.DEFAULT_FRONT_CAMERA
                : CameraSelector.DEFAULT_BACK_CAMERA)) {
            return false;
        }
        useFrontCamera = targetFront;
        if (!bindCurrentLens()) {
            useFrontCamera = !targetFront;
            bindCurrentLens();
            return false;
        }
        listener.onLensChanged(useFrontCamera);
        return true;
    }

    /** Stops only this component's use cases; it does not disturb another Activity. */
    public void stop() {
        stopped = true;
        if (analysis != null) {
            analysis.clearAnalyzer();
        }
        if (cameraProvider != null && preview != null && analysis != null) {
            cameraProvider.unbind(preview, analysis);
        }
    }

    private void createUseCases() {
        ResolutionSelector analysisResolution = new ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(new ResolutionStrategy(new Size(640, 480),
                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                .build();
        analysis = new ImageAnalysis.Builder()
                .setResolutionSelector(analysisResolution)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();
        analysis.setAnalyzer(analysisExecutor, analyzer);

        preview = new Preview.Builder()
                .setResolutionSelector(new ResolutionSelector.Builder()
                        .setAspectRatioStrategy(
                                AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                        .build())
                .build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());
    }

    private boolean bindCurrentLens() {
        if (cameraProvider == null || preview == null || analysis == null || stopped) {
            return false;
        }
        try {
            cameraProvider.unbind(preview, analysis);
            cameraProvider.bindToLifecycle(lifecycleOwner,
                    useFrontCamera ? CameraSelector.DEFAULT_FRONT_CAMERA
                            : CameraSelector.DEFAULT_BACK_CAMERA,
                    preview, analysis);
            return true;
        } catch (RuntimeException e) {
            listener.onCameraError(R.string.msg_camera_unavailable);
            return false;
        }
    }

    private boolean hasCamera(CameraSelector selector) {
        try {
            return cameraProvider != null && cameraProvider.hasCamera(selector);
        } catch (CameraInfoUnavailableException e) {
            return false;
        }
    }
}

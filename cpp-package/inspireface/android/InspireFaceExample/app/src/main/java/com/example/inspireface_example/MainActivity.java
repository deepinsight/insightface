package com.example.inspireface_example;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.insightface.sdk.inspireface.InspireFace;
import com.insightface.sdk.inspireface.base.*;
import com.insightface.sdk.inspireface.utils.SDKUtils;

import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    private final String TAG = "InspireFace";

    void test() {
        InspireFaceVersion version = InspireFace.QueryInspireFaceVersion();
        Log.i(TAG, "InspireFace Version: " + version.major + "." + version.minor + "." + version.patch);
        String dbPath = "/storage/emulated/0/Android/data/com.example.inspireface_example/files/f.db";
        FeatureHubConfiguration configuration = InspireFace.CreateFeatureHubConfiguration()
                .setEnablePersistence(false)
                .setPersistenceDbPath(dbPath)
                .setSearchThreshold(0.42f)
                .setSearchMode(InspireFace.SEARCH_MODE_EXHAUSTIVE)
                .setPrimaryKeyMode(InspireFace.PK_AUTO_INCREMENT);

        boolean enableStatus = InspireFace.FeatureHubDataEnable(configuration);
        Log.d(TAG, "Enable feature hub data status: " + enableStatus);
        InspireFace.FeatureHubFaceSearchThresholdSetting(0.42f);

        boolean launchStatus = InspireFace.GlobalLaunch(this, InspireFace.PIKACHU);
        Log.d(TAG, "Launch status: " + launchStatus);
        if (!launchStatus) {
            Log.e(TAG, "Failed to launch InspireFace");
            return;
        }
        CustomParameter parameter = InspireFace.CreateCustomParameter()
                .enableRecognition(true)
                .enableFaceQuality(true)
                .enableFaceAttribute(true)
                .enableInteractionLiveness(true)
                .enableLiveness(true)
                .enableMaskDetect(true);
        Session session = InspireFace.CreateSession(parameter, InspireFace.DETECT_MODE_ALWAYS_DETECT, 10, -1, -1);
        Log.i(TAG, "session handle: " + session.handle);
        InspireFace.SetTrackPreviewSize(session, 320);
        InspireFace.SetFaceDetectThreshold(session, 0.5f);
        InspireFace.SetFilterMinimumFacePixelSize(session, 0);

        Bitmap img = SDKUtils.getImageFromAssetsFile(this, "inspireface/kun.jpg");
        ImageStream stream = InspireFace.CreateImageStreamFromBitmap(img, InspireFace.CAMERA_ROTATION_0);
        Log.i(TAG, "stream handle: " + stream.handle);
        InspireFace.WriteImageStreamToFile(stream, "/storage/emulated/0/Android/data/com.example.inspireface_example/files/out.jpg");

        MultipleFaceData multipleFaceData = InspireFace.ExecuteFaceTrack(session, stream);
        Log.i(TAG, "Face num: " + multipleFaceData.detectedNum);

        if (multipleFaceData.detectedNum > 0) {
            Point2f[] lmk = InspireFace.GetFaceDenseLandmarkFromFaceToken(multipleFaceData.tokens[0]);
            for (Point2f p : lmk) {
                Log.i(TAG, p.x + ", " + p.y);
            }
            FaceFeature feature = InspireFace.ExtractFaceFeature(session, stream, multipleFaceData.tokens[0]);
            Log.i(TAG, "Feature size: " + feature.data.length);
            String strFt = "";
            for (int i = 0; i < feature.data.length; i++) {
                strFt = strFt + feature.data[i] + ", ";
            }
            Log.i(TAG, strFt);

            for (int i = 0; i < 10; i++) {
                FaceFeatureIdentity identity = FaceFeatureIdentity.create(-1, feature);
                boolean succ = InspireFace.FeatureHubInsertFeature(identity);
                if (succ) {
                    Log.i(TAG, "Allocation ID: " + identity.id);
                }
            }

            FaceFeatureIdentity searched = InspireFace.FeatureHubFaceSearch(feature);
            Log.i(TAG, "Searched id: " + searched.id + ", Confidence: " + searched.searchConfidence);

            SearchTopKResults topKResults = InspireFace.FeatureHubFaceSearchTopK(feature, 10);
            for (int i = 0; i < topKResults.num; i++) {
                Log.i(TAG, "TopK id: " + topKResults.ids[i] + ", Confidence: " + topKResults.confidence[i]);
            }

            FaceFeature newFeature = new FaceFeature();
            Log.i(TAG, "Feature length: " + InspireFace.GetFeatureLength());
            newFeature.data = new float[InspireFace.GetFeatureLength()];
            FaceFeatureIdentity identity = FaceFeatureIdentity.create(8, newFeature);
            boolean updateSucc = InspireFace.FeatureHubFaceUpdate(identity);
            if (updateSucc) {
                Log.i(TAG, "Update feature success: " + 8);
            }
            boolean removeSucc = InspireFace.FeatureHubFaceRemove(4);
            if (removeSucc) {
                Log.i(TAG, "Remove feature success: " + 4);
            }
            SearchTopKResults topkAgn = InspireFace.FeatureHubFaceSearchTopK(feature, 10);
            for (int i = 0; i < topkAgn.num; i++) {
                Log.i(TAG, "Agn TopK id: " + topkAgn.ids[i] + ", Confidence: " + topKResults.confidence[i]);
            }

            FaceFeatureIdentity queryIdentity = InspireFace.FeatureHubGetFaceIdentity(4);
            if (queryIdentity != null) {
                Log.e(TAG, "query id: " + queryIdentity.id);
            }
            queryIdentity = InspireFace.FeatureHubGetFaceIdentity(2);
            if (queryIdentity != null) {
                strFt = "";
                for (int i = 0; i < queryIdentity.feature.data.length; i++) {
                    strFt = strFt + queryIdentity.feature.data[i] + ", ";
                }
                Log.i(TAG, "query id: " + queryIdentity.id);
                Log.i(TAG, strFt);

                float comp = InspireFace.FaceComparison(queryIdentity.feature, feature);
                Log.i(TAG, "Comparison: " + comp);
            }
            CustomParameter pipelineNeedParam = InspireFace.CreateCustomParameter()
                    .enableFaceQuality(true)
                    .enableLiveness(true)
                    .enableMaskDetect(true)
                    .enableFaceAttribute(true)
                    .enableInteractionLiveness(true);
            boolean succPipe = InspireFace.MultipleFacePipelineProcess(session, stream, multipleFaceData, pipelineNeedParam);
            if (succPipe) {
                Log.i(TAG, "Exec pipeline success");
                RGBLivenessConfidence rgbLivenessConfidence = InspireFace.GetRGBLivenessConfidence(session);
                Log.i(TAG, "rgbLivenessConfidence: " + rgbLivenessConfidence.confidence[0]);
                FaceQualityConfidence faceQualityConfidence = InspireFace.GetFaceQualityConfidence(session);
                Log.i(TAG, "faceQualityConfidence: " + faceQualityConfidence.confidence[0]);
                FaceMaskConfidence faceMaskConfidence = InspireFace.GetFaceMaskConfidence(session);
                Log.i(TAG, "faceMaskConfidence: " + faceMaskConfidence.confidence[0]);
                FaceInteractionState faceInteractionState = InspireFace.GetFaceInteractionStateResult(session);
                Log.i(TAG, "Left eye status confidence: " + faceInteractionState.leftEyeStatusConfidence[0]);
                Log.i(TAG, "Right eye status confidence: " + faceInteractionState.rightEyeStatusConfidence[0]);
                FaceInteractionsActions faceInteractionsActions = InspireFace.GetFaceInteractionActionsResult(session);
                Log.i(TAG, "Normal: " + faceInteractionsActions.normal[0]);
                Log.i(TAG, "Shake: " + faceInteractionsActions.shake[0]);
                Log.i(TAG, "Jaw open: " + faceInteractionsActions.jawOpen[0]);
                Log.i(TAG, "Head raise: " + faceInteractionsActions.headRaise[0]);
                Log.i(TAG, "Blink: " + faceInteractionsActions.blink[0]);
                FaceAttributeResult faceAttributeResult = InspireFace.GetFaceAttributeResult(session);
                Log.i(TAG, "Race: " + faceAttributeResult.race[0]);
                Log.i(TAG, "Gender: " + faceAttributeResult.gender[0]);
                Log.i(TAG, "Age bracket: " + faceAttributeResult.ageBracket[0]);
            } else {

                Log.e(TAG, "Exec pipeline fail");
            }
        }

        int count = InspireFace.FeatureHubGetFaceCount();
        Log.i(TAG, "Face count: " + count);

        Bitmap crop = InspireFace.GetFaceAlignmentImage(session, stream, multipleFaceData.tokens[0]);
        try {
            SDKUtils.saveBitmap("/storage/emulated/0/Android/data/com.example.inspireface_example/files/", "crop", crop);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        InspireFace.ReleaseImageStream(stream);
        InspireFace.ReleaseSession(session);


        InspireFace.FeatureHubDataDisable();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        //

        test();
    }
}
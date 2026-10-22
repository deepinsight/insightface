import com.insightface.sdk.inspireface.jni.NativeTypes.*;
import static com.insightface.sdk.inspireface.jni.Native.*;
import static com.insightface.sdk.inspireface.jni.NativeConstants.*;
import static com.insightface.sdk.inspireface.jni.InspireFaceException.check;

/**
 * Non-Android JVM example. Run from the installed Java SDK directory:
 * javac -cp inspireface.jar examples/DetectFaces.java
 * java -Djava.library.path=native/macos-arm64 -cp inspireface.jar:examples DetectFaces /path/Pikachu /path/face.jpg
 * Select the native directory matching the JVM OS/architecture; Windows uses ; in the classpath.
 */
public final class DetectFaces {
    public static void main(String[] args) {
        if (args.length != 2) throw new IllegalArgumentException("Usage: DetectFaces MODEL_FILE IMAGE_FILE");
        check(HFLaunchInspireFace(args[0]));
        long[] session = new long[1], bitmap = new long[1], stream = new long[1];
        try {
            check(HFCreateInspireFaceSessionOptional(HF_ENABLE_NONE, HF_DETECT_MODE_ALWAYS_DETECT, 10, -1, -1, session));
            check(HFCreateImageBitmapFromFilePath(args[1], 3, bitmap));
            HFImageBitmapData pixels = new HFImageBitmapData();
            check(HFImageBitmapGetData(bitmap[0], pixels));
            HFImageData input = new HFImageData();
            input.data = pixels.data; input.width = pixels.width; input.height = pixels.height;
            input.format = HF_STREAM_BGR; input.rotation = HF_CAMERA_ROTATION_0;
            check(HFCreateImageStream(input, stream)); // Borrows pixels; keep bitmap alive.
            HFMultipleFaceData faces = new HFMultipleFaceData();
            check(HFExecuteFaceTrack(session[0], stream[0], faces));
            System.out.println("Detected " + faces.detectedNum + " face(s)");
            for (HFaceRect rect : faces.rects) {
                System.out.printf("x=%d y=%d width=%d height=%d%n", rect.x, rect.y, rect.width, rect.height);
            }
        } finally {
            if (stream[0] != 0) HFReleaseImageStream(stream[0]);
            if (bitmap[0] != 0) HFReleaseImageBitmap(bitmap[0]);
            if (session[0] != 0) HFReleaseInspireFaceSession(session[0]);
            HFTerminateInspireFace();
        }
    }
}

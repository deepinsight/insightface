import { DetectMode, Feature, ImageFormat, InspireFace, Rotation, Session } from '@hyperinspire/inspireface';

export function detectFirstFace(resourcePath: string, rgba: Uint8Array, width: number, height: number): number {
  InspireFace.launch(resourcePath);
  const session: Session = InspireFace.createSession({
    featureMask: Feature.FACE_RECOGNITION,
    detectMode: DetectMode.ALWAYS_DETECT,
    maxFaces: 5
  });
  const image = InspireFace.createImageStream(rgba, width, height, ImageFormat.RGBA, Rotation.DEGREE_0);
  try {
    const result = session.track(image);
    try {
      return result.detectedNum;
    } finally {
      session.releaseFaceResult(result);
    }
  } finally {
    image.close();
    session.close();
    InspireFace.terminate();
  }
}

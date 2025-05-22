import cv2
import numpy as np
from inspireface import ImageStream
import inspireface as isf


def read_nv21(file_path, width, height, rotate=0):
    with open(file_path, 'rb') as f:
        nv21_data = f.read()
    
    yuv = np.frombuffer(nv21_data, dtype=np.uint8)
    
    expected_size = width * height * 3 // 2
    if len(yuv) < expected_size:
        raise ValueError(f"NV21 data size is not enough: expected {expected_size} bytes, actual {len(yuv)} bytes")
    
    yuv_mat = np.zeros((height * 3 // 2, width), dtype=np.uint8)
    yuv_mat[:] = yuv[:height * width * 3 // 2].reshape(height * 3 // 2, width)
    
    bgr_mat = cv2.cvtColor(yuv_mat, cv2.COLOR_YUV2BGR_NV21)
    
    # add reverse rotate
    if rotate != 0:
        # calculate reverse rotate angle
        reverse_angle = (360 - rotate) % 360
        
        # select rotate method by angle
        if reverse_angle == 90:
            bgr_mat = cv2.rotate(bgr_mat, cv2.ROTATE_90_CLOCKWISE)
        elif reverse_angle == 180:
            bgr_mat = cv2.rotate(bgr_mat, cv2.ROTATE_180)
        elif reverse_angle == 270:
            bgr_mat = cv2.rotate(bgr_mat, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            # for non 90 degree angle, use affine transform
            center = (bgr_mat.shape[1] // 2, bgr_mat.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, reverse_angle, 1.0)
            bgr_mat = cv2.warpAffine(bgr_mat, rotation_matrix, (bgr_mat.shape[1], bgr_mat.shape[0]))
    
    return bgr_mat

# example usage
if __name__ == "__main__":
    image_width = 640
    image_height = 480
    nv21_file = ""
    pre_rotate = 90
    try:
        # read and reverse rotate image
        bgr_image = read_nv21(nv21_file, image_width, image_height, pre_rotate)
        
        cv2.imshow("origin", bgr_image)
        
        # show original image for comparison
        bgr_image_original = read_nv21(nv21_file, image_width, image_height)
        cv2.imshow("rotate ", bgr_image_original)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"error: {e}")

    with open(nv21_file, 'rb') as f:
        nv21_data = f.read()

    stream = ImageStream.load_from_buffer(nv21_data, image_width, image_height, isf.HF_STREAM_YUV_NV21, isf.HF_CAMERA_ROTATION_90)

    stream.debug_show()

    
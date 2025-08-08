/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#ifndef INSPIRE_FACE_HERROR_H
#define INSPIRE_FACE_HERROR_H

// [Anchor-Begin]

#define HSUCCEED (0)  // Success

// Basic error types (1-99)
#define HERR_BASIC_BASE 0x0001                                   // Basic error types
#define HERR_UNKNOWN HERR_BASIC_BASE                             // Unknown error (1)
#define HERR_INVALID_PARAM (HERR_BASIC_BASE + 1)                 // Invalid parameter (2)
#define HERR_INVALID_IMAGE_STREAM_HANDLE (HERR_BASIC_BASE + 2)   // Invalid image stream handle (3)
#define HERR_INVALID_CONTEXT_HANDLE (HERR_BASIC_BASE + 3)        // Invalid context handle (4)
#define HERR_INVALID_FACE_TOKEN (HERR_BASIC_BASE + 4)            // Invalid face token (5)
#define HERR_INVALID_FACE_FEATURE (HERR_BASIC_BASE + 5)          // Invalid face feature (6)
#define HERR_INVALID_FACE_LIST (HERR_BASIC_BASE + 6)             // Invalid face feature list (7)
#define HERR_INVALID_BUFFER_SIZE (HERR_BASIC_BASE + 7)           // Invalid copy token (8)
#define HERR_INVALID_IMAGE_STREAM_PARAM (HERR_BASIC_BASE + 8)    // Invalid image param (9)
#define HERR_INVALID_SERIALIZATION_FAILED (HERR_BASIC_BASE + 9)  // Invalid face serialization failed (10)
#define HERR_INVALID_DETECTION_INPUT (HERR_BASIC_BASE + 10)      // Failed to modify detector input size (11)
#define HERR_INVALID_IMAGE_BITMAP_HANDLE (HERR_BASIC_BASE + 11)  // Invalid image bitmap handle (12)
#define HERR_IMAGE_STREAM_DECODE_FAILED (HERR_BASIC_BASE + 12)  // ImageStream failed to decode the image (13)

// Session error types (100-199)
#define HERR_SESS_BASE 0x0064                                   // Session error types (100)
#define HERR_SESS_FUNCTION_UNUSABLE (HERR_SESS_BASE + 1)        // Function not usable (101)
#define HERR_SESS_TRACKER_FAILURE (HERR_SESS_BASE + 2)          // Tracker module not initialized (102)
#define HERR_SESS_PIPELINE_FAILURE (HERR_SESS_BASE + 3)         // Pipeline module not initialized (103)
#define HERR_SESS_INVALID_RESOURCE (HERR_SESS_BASE + 4)         // Invalid static resource (104)
#define HERR_SESS_LANDMARK_NUM_NOT_MATCH (HERR_SESS_BASE + 5)   // The number of input landmark points does not match (105)
#define HERR_SESS_LANDMARK_NOT_ENABLE (HERR_SESS_BASE + 6)      // The landmark model is not enabled (106)
#define HERR_SESS_KEY_POINT_NUM_NOT_MATCH (HERR_SESS_BASE + 7)  // The number of input key points does not match (107)
#define HERR_SESS_REC_EXTRACT_FAILURE (HERR_SESS_BASE + 8)      // Face feature extraction not registered (108)
#define HERR_SESS_REC_CONTRAST_FEAT_ERR (HERR_SESS_BASE + 9)    // Incorrect length of feature vector for comparison (109)
#define HERR_SESS_FACE_DATA_ERROR (HERR_SESS_BASE + 10)         // Face data parsing (110)
#define HERR_SESS_FACE_REC_OPTION_ERROR (HERR_SESS_BASE + 11)   // An optional parameter is incorrect (111)

// FeatureHub error types (200-249)
#define HERR_FT_HUB_BASE 0x00C8                               // FeatureHub error types (200)
#define HERR_FT_HUB_DISABLE (HERR_FT_HUB_BASE + 1)            // FeatureHub is disabled (201)
#define HERR_FT_HUB_INSERT_FAILURE (HERR_FT_HUB_BASE + 2)     // Data insertion error (202)
#define HERR_FT_HUB_NOT_FOUND_FEATURE (HERR_FT_HUB_BASE + 3)  // Get face feature error (203)

// Archive error types (250-299)
#define HERR_ARCHIVE_BASE 0x00FA                                 // Archive error types (250)
#define HERR_ARCHIVE_LOAD_FAILURE (HERR_ARCHIVE_BASE + 1)        // Archive load failure (251)
#define HERR_ARCHIVE_LOAD_MODEL_FAILURE (HERR_ARCHIVE_BASE + 2)  // Model load failure (252)
#define HERR_ARCHIVE_FILE_FORMAT_ERROR (HERR_ARCHIVE_BASE + 3)   // The archive format is incorrect (253)
#define HERR_ARCHIVE_REPETITION_LOAD (HERR_ARCHIVE_BASE + 4)     // Do not reload the model (254)
#define HERR_ARCHIVE_NOT_LOAD (HERR_ARCHIVE_BASE + 5)            // Model not loaded (255)

// Device/Hardware error types (300-349)
#define HERR_DEVICE_BASE 0x012C                                       // Hardware error types (300)
#define HERR_DEVICE_CUDA_NOT_SUPPORT (HERR_DEVICE_BASE + 1)           // CUDA not supported (301)
#define HERR_DEVICE_CUDA_TENSORRT_NOT_SUPPORT (HERR_DEVICE_BASE + 2)  // CUDA TensorRT not supported (302)
#define HERR_DEVICE_CUDA_UNKNOWN_ERROR (HERR_DEVICE_BASE + 3)         // CUDA unknown error (303)
#define HERR_DEVICE_CUDA_DISABLE (HERR_DEVICE_BASE + 4)               // CUDA support is disabled (304)

// Extension module error types (350-549)
#define HERR_EXTENSION_BASE 0x015E                                             // Extension module error types (350)
#define HERR_EXTENSION_ERROR (HERR_EXTENSION_BASE + 1)                         // Extension module error (351)
#define HERR_EXTENSION_MLMODEL_LOAD_FAILED (HERR_EXTENSION_BASE + 2)           // MLModel load failed (352)
#define HERR_EXTENSION_HETERO_MODEL_TAG_ERROR (HERR_EXTENSION_BASE + 3)        // Incorrect heterogeneous model tag (353)
#define HERR_EXTENSION_HETERO_REC_HEAD_CONFIG_ERROR (HERR_EXTENSION_BASE + 4)  // Rec head config error (354)
#define HERR_EXTENSION_HETERO_MODEL_NOT_MATCH (HERR_EXTENSION_BASE + 5)        // Heterogeneous model dimensions do not match (355)
#define HERR_EXTENSION_HETERO_MODEL_NOT_LOADED (HERR_EXTENSION_BASE + 6)       // Heterogeneous model dimensions not loaded (356)

// [Anchor-End]

#endif  // INSPIRE_FACE_HERROR_H

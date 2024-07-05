//
// Created by Tunm-Air13 on 2023/9/11.
//

#ifndef HYPERFACEREPO_HERROR_H
#define HYPERFACEREPO_HERROR_H


// [Anchor-Begin]

#define HSUCCEED						    (0)                             // Success
#define HERR_BASIC_BASE					    0X0001							// Basic error types
#define HERR_UNKNOWN					    HERR_BASIC_BASE					// Unknown error
#define HERR_INVALID_PARAM				    (HERR_BASIC_BASE+1)				// Invalid parameter
#define HERR_INVALID_IMAGE_STREAM_HANDLE	(HERR_BASIC_BASE+24)			// Invalid image stream handle
#define HERR_INVALID_CONTEXT_HANDLE	        (HERR_BASIC_BASE+25)			// Invalid context handle
#define HERR_INVALID_FACE_TOKEN     	    (HERR_BASIC_BASE+30)			// Invalid face token
#define HERR_INVALID_FACE_FEATURE    	    (HERR_BASIC_BASE+31)			// Invalid face feature
#define HERR_INVALID_FACE_LIST          	(HERR_BASIC_BASE+32)			// Invalid face feature list
#define HERR_INVALID_BUFFER_SIZE          	(HERR_BASIC_BASE+33)			// Invalid copy token
#define HERR_INVALID_IMAGE_STREAM_PARAM      (HERR_BASIC_BASE+34)			// Invalid image param
#define HERR_INVALID_SERIALIZATION_FAILED    (HERR_BASIC_BASE+35)			// Invalid face serialization failed
#define HERR_INVALID_DETECTION_INPUT        (HERR_BASIC_BASE+36)			// Failed to modify detector input size

#define HERR_SESS_BASE				        0X500					         // Session error types
#define HERR_SESS_FUNCTION_UNUSABLE           (HERR_SESS_BASE+2)               // Function not usable
#define HERR_SESS_TRACKER_FAILURE             (HERR_SESS_BASE+3)               // Tracker module not initialized
#define HERR_SESS_INVALID_RESOURCE            (HERR_SESS_BASE+10)              // Invalid static resource
#define HERR_SESS_NUM_OF_MODELS_NOT_MATCH     (HERR_SESS_BASE+11)              // Number of models does not match
#define HERR_SESS_LANDMARK_NUM_NOT_MATCH      (HERR_SESS_BASE+20)              // The number of input landmark points does not match

#define HERR_SESS_PIPELINE_FAILURE            (HERR_SESS_BASE+8)               // Pipeline module not initialized

#define HERR_SESS_REC_EXTRACT_FAILURE         (HERR_SESS_BASE+15)              // Face feature extraction not registered
#define HERR_SESS_REC_DEL_FAILURE             (HERR_SESS_BASE+16)              // Face feature deletion failed due to out of range index
#define HERR_SESS_REC_UPDATE_FAILURE          (HERR_SESS_BASE+17)              // Face feature update failed due to out of range index
#define HERR_SESS_REC_ADD_FEAT_EMPTY          (HERR_SESS_BASE+18)              // Feature vector for registration cannot be empty
#define HERR_SESS_REC_FEAT_SIZE_ERR           (HERR_SESS_BASE+19)              // Incorrect length of feature vector for registration
#define HERR_SESS_REC_INVALID_INDEX           (HERR_SESS_BASE+20)              // Invalid index number
#define HERR_SESS_REC_CONTRAST_FEAT_ERR       (HERR_SESS_BASE+23)              // Incorrect length of feature vector for comparison
#define HERR_SESS_REC_BLOCK_FULL              (HERR_SESS_BASE+24)              // Feature vector block full
#define HERR_SESS_REC_BLOCK_DEL_FAILURE       (HERR_SESS_BASE+25)              // Deletion failed
#define HERR_SESS_REC_BLOCK_UPDATE_FAILURE    (HERR_SESS_BASE+26)              // Update failed
#define HERR_SESS_REC_ID_ALREADY_EXIST        (HERR_SESS_BASE+27)              // ID already exists

#define HERR_SESS_FACE_DATA_ERROR             (HERR_SESS_BASE+30)              // Face data parsing

#define HERR_SESS_FACE_REC_OPTION_ERROR       (HERR_SESS_BASE+40)              // An optional parameter is incorrect

#define HERR_FT_HUB_DISABLE                  (HERR_SESS_BASE+49)              // FeatureHub is disabled
#define HERR_FT_HUB_OPEN_ERROR               (HERR_SESS_BASE+50)              // Database open error
#define HERR_FT_HUB_NOT_OPENED               (HERR_SESS_BASE+51)              // Database not opened
#define HERR_FT_HUB_NO_RECORD_FOUND          (HERR_SESS_BASE+52)              // No record found
#define HERR_FT_HUB_CHECK_TABLE_ERROR        (HERR_SESS_BASE+53)              // Data table check error
#define HERR_FT_HUB_INSERT_FAILURE           (HERR_SESS_BASE+54)              // Data insertion error
#define HERR_FT_HUB_PREPARING_FAILURE        (HERR_SESS_BASE+55)              // Data preparation error
#define HERR_FT_HUB_EXECUTING_FAILURE        (HERR_SESS_BASE+56)              // SQL execution error
#define HERR_FT_HUB_NOT_VALID_FOLDER_PATH    (HERR_SESS_BASE+57)              // Invalid folder path
#define HERR_FT_HUB_ENABLE_REPETITION        (HERR_SESS_BASE+58)              // Enable db function repeatedly
#define HERR_FT_HUB_DISABLE_REPETITION       (HERR_SESS_BASE+59)              // Disable db function repeatedly

#define HERR_ARCHIVE_LOAD_FAILURE           (HERR_SESS_BASE+80)              // Archive load failure
#define HERR_ARCHIVE_LOAD_MODEL_FAILURE     (HERR_SESS_BASE+81)              // Model load failure
#define HERR_ARCHIVE_FILE_FORMAT_ERROR      (HERR_SESS_BASE+82)              // The archive format is incorrect
#define HERR_ARCHIVE_REPETITION_LOAD      (HERR_SESS_BASE+83)              // Do not reload the model
#define HERR_ARCHIVE_NOT_LOAD             (HERR_SESS_BASE+84)              // Model not loaded

// [Anchor-End]

#endif //HYPERFACEREPO_HERROR_H

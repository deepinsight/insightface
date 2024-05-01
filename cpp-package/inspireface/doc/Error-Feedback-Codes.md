# Error Feedback Codes

During the use of InspireFace, some error feedback codes may be generated. Here is a table of error feedback codes.

| **Index** | **Name** | **Code** | **Description** | 
| --- | --- | --- | --- | 
| 1 | HSUCCEED | 0 | Success | 
| 2 | HERR_BASIC_BASE | 1 | Basic error types | 
| 3 | HERR_UNKNOWN | 1 | Unknown error | 
| 4 | HERR_INVALID_PARAM | 2 | Invalid parameter | 
| 5 | HERR_INVALID_IMAGE_STREAM_HANDLE | 25 | Invalid image stream handle | 
| 6 | HERR_INVALID_CONTEXT_HANDLE | 26 | Invalid context handle | 
| 7 | HERR_INVALID_FACE_TOKEN | 31 | Invalid face token | 
| 8 | HERR_INVALID_FACE_FEATURE | 32 | Invalid face feature | 
| 9 | HERR_INVALID_FACE_LIST | 33 | Invalid face feature list | 
| 10 | HERR_INVALID_BUFFER_SIZE | 34 | Invalid copy token | 
| 11 | HERR_INVALID_IMAGE_STREAM_PARAM | 35 | Invalid image param | 
| 12 | HERR_INVALID_SERIALIZATION_FAILED | 36 | Invalid face serialization failed | 
| 13 | HERR_SESS_BASE | 1280 | Session error types | 
| 14 | HERR_SESS_FUNCTION_UNUSABLE | 1282 | Function not usable | 
| 15 | HERR_SESS_TRACKER_FAILURE | 1283 | Tracker module not initialized | 
| 16 | HERR_SESS_INVALID_RESOURCE | 1290 | Invalid static resource | 
| 17 | HERR_SESS_NUM_OF_MODELS_NOT_MATCH | 1291 | Number of models does not match | 
| 18 | HERR_SESS_PIPELINE_FAILURE | 1288 | Pipeline module not initialized | 
| 19 | HERR_SESS_REC_EXTRACT_FAILURE | 1295 | Face feature extraction not registered | 
| 20 | HERR_SESS_REC_DEL_FAILURE | 1296 | Face feature deletion failed due to out of range index | 
| 21 | HERR_SESS_REC_UPDATE_FAILURE | 1297 | Face feature update failed due to out of range index | 
| 22 | HERR_SESS_REC_ADD_FEAT_EMPTY | 1298 | Feature vector for registration cannot be empty | 
| 23 | HERR_SESS_REC_FEAT_SIZE_ERR | 1299 | Incorrect length of feature vector for registration | 
| 24 | HERR_SESS_REC_INVALID_INDEX | 1300 | Invalid index number | 
| 25 | HERR_SESS_REC_CONTRAST_FEAT_ERR | 1303 | Incorrect length of feature vector for comparison | 
| 26 | HERR_SESS_REC_BLOCK_FULL | 1304 | Feature vector block full | 
| 27 | HERR_SESS_REC_BLOCK_DEL_FAILURE | 1305 | Deletion failed | 
| 28 | HERR_SESS_REC_BLOCK_UPDATE_FAILURE | 1306 | Update failed | 
| 29 | HERR_SESS_REC_ID_ALREADY_EXIST | 1307 | ID already exists | 
| 30 | HERR_SESS_FACE_DATA_ERROR | 1310 | Face data parsing | 
| 31 | HERR_SESS_FACE_REC_OPTION_ERROR | 1320 | An optional parameter is incorrect | 
| 32 | HERR_FT_HUB_DISABLE | 1329 | FeatureHub is disabled | 
| 33 | HERR_FT_HUB_OPEN_ERROR | 1330 | Database open error | 
| 34 | HERR_FT_HUB_NOT_OPENED | 1331 | Database not opened | 
| 35 | HERR_FT_HUB_NO_RECORD_FOUND | 1332 | No record found | 
| 36 | HERR_FT_HUB_CHECK_TABLE_ERROR | 1333 | Data table check error | 
| 37 | HERR_FT_HUB_INSERT_FAILURE | 1334 | Data insertion error | 
| 38 | HERR_FT_HUB_PREPARING_FAILURE | 1335 | Data preparation error | 
| 39 | HERR_FT_HUB_EXECUTING_FAILURE | 1336 | SQL execution error | 
| 40 | HERR_FT_HUB_NOT_VALID_FOLDER_PATH | 1337 | Invalid folder path | 
| 41 | HERR_FT_HUB_ENABLE_REPETITION | 1338 | Enable db function repeatedly | 
| 42 | HERR_FT_HUB_DISABLE_REPETITION | 1339 | Disable db function repeatedly | 
| 43 | HERR_ARCHIVE_LOAD_FAILURE | 1360 | Archive load failure | 
| 44 | HERR_ARCHIVE_LOAD_MODEL_FAILURE | 1361 | Model load failure | 
| 45 | HERR_ARCHIVE_FILE_FORMAT_ERROR | 1362 | The archive format is incorrect | 
| 46 | HERR_ARCHIVE_REPETITION_LOAD | 1363 | Do not reload the model | 
| 47 | HERR_ARCHIVE_NOT_LOAD | 1364 | Model not loaded |

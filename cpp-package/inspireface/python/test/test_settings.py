import os
import sys
import inspireface as ifac

# ++ OPTIONAL ++

# Enabling will run all the benchmark tests, which takes time
ENABLE_BENCHMARK_TEST = True

# Enabling will run all the CRUD tests, which will take time
ENABLE_CRUD_TEST = False

# Enabling will run the face search benchmark, which takes time and must be configured with the correct
# 'LFW_FUNNELED_DIR_PATH' parameter
ENABLE_SEARCH_BENCHMARK_TEST = True

# Enabling will run the LFW dataset precision test, which will take time
ENABLE_LFW_PRECISION_TEST = False

# Testing model name
TEST_MODEL_NAME = "Pikachu"
# TEST_MODEL_NAME = "Megatron"

# Testing length of face feature
TEST_MODEL_FACE_FEATURE_LENGTH = 512

# Testing face comparison image threshold
TEST_FACE_COMPARISON_IMAGE_THRESHOLD = 0.45

# ++ END OPTIONAL ++

# Current project path
TEST_PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

# Current project path
CURRENT_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Main project path
MAIN_PROJECT_PATH = os.path.dirname(CURRENT_PROJECT_PATH)

# Model zip path
MODEL_ZIP_PATH = os.path.join(MAIN_PROJECT_PATH, "test_res/pack/")

# Testing model full path
TEST_MODEL_PATH = os.path.join(MODEL_ZIP_PATH, TEST_MODEL_NAME)

# Python test data folder
PYTHON_TEST_DATA_FOLDER = os.path.join(TEST_PROJECT_PATH, "data/")

# Stores some temporary file data generated during testing
TMP_FOLDER = os.path.join(CURRENT_PROJECT_PATH, "tmp")

# Default db file path
DEFAULT_DB_PATH = os.path.join(TMP_FOLDER, ".E63520A95DD5B3892C56DA38C3B28E551D8173FD")

# Create tmp if not exist
os.makedirs(TMP_FOLDER, exist_ok=True)

# lfw_funneled Dataset dir path
LFW_FUNNELED_DIR_PATH = "/Users/tunm/datasets/lfw_funneled/"

# The LFW data predicted by the algorithm is used and cached to save time in the next prediction, and it can be
# re-predicted by manually deleting it
LFW_PREDICT_DATA_CACHE_PATH = os.path.join(TMP_FOLDER, "LFW_PRED.npy")

assert os.path.exists(LFW_FUNNELED_DIR_PATH), "'LFW_FUNNELED_DIR_PATH' is not found."

ifac.launch(TEST_MODEL_PATH)
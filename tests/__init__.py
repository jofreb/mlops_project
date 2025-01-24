import os

_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data/processed")  # root of data
_PATH_MODELS = os.path.join(_PROJECT_ROOT, "models")  # root of models
_PATH_SRC = os.path.join(_PROJECT_ROOT, "src")  # root of src
_PATH_UTILS = os.path.join(_PROJECT_ROOT, "src/nrms_ml_ops/utils")  # root of utils
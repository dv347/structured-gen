import os


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)

CONFIGS_DIR = os.path.join(ROOT_DIR, "configs")
TRAIN_CONFIGS_DIR = os.path.join(CONFIGS_DIR, "train")
EVAL_CONFIGS_DIR = os.path.join(CONFIGS_DIR, "evaluate")
DATA_DIR = os.path.join(ROOT_DIR, "data/smcalflow")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
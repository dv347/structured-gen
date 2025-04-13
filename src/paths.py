import os


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)

CONFIGS_DIR = os.path.join(ROOT_DIR, "configs")
TRAIN_CONFIGS_DIR = os.path.join(CONFIGS_DIR, "train")
EVAL_CONFIGS_DIR = os.path.join(CONFIGS_DIR, "evaluate")
DATA_DIR = os.path.join(ROOT_DIR, "data")
GRAMMARS_DIR = os.path.join(ROOT_DIR, "grammars")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

_DATASET_NAME = None
_DATASET_DIR = None


def set_dataset(name: str) -> None:
    global _DATASET_NAME, _DATASET_DIR
    _DATASET_NAME = name
    _DATASET_DIR = os.path.join(DATA_DIR, name)


def get_dataset_name() -> str:
    if _DATASET_NAME is None:
        raise ValueError("Dataset name not set. Call `set_dataset(name)` first.")
    return _DATASET_NAME


def get_dataset_dir() -> str:
    if _DATASET_DIR is None:
        raise ValueError("Dataset not set. Call `set_dataset(name)` first.")
    return _DATASET_DIR


def get_grammar_path() -> str:
    return os.path.join(GRAMMARS_DIR, f"{get_dataset_name()}.lark")


def get_model_dir(path: str) -> str:
    return os.path.join(MODELS_DIR, get_dataset_name(), path)


def get_merged_model_dir(path: str) -> str:
    return os.path.join(get_model_dir(path), "merged_model")
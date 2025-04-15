import os
import shutil

from logger import logger
from paths import MODELS_DIR

PRESERVE_FILES = {"loss_plot.png", "loss_history.json"}


def safe_listdir(path):
    return os.listdir(path) if os.path.isdir(path) else []


def cleanup_model_dir():
    for dataset_dir in safe_listdir(MODELS_DIR):
        dataset_dir_path = os.path.join(MODELS_DIR, dataset_dir)

        for stage_dir in safe_listdir(dataset_dir_path):
            stage_dir_path = os.path.join(dataset_dir_path, stage_dir)

            for model_dir in safe_listdir(stage_dir_path):
                model_dir_path = os.path.join(stage_dir_path, model_dir)

                for item in safe_listdir(model_dir_path):
                    item_path = os.path.join(model_dir_path, item)
                    if os.path.isdir(item_path):
                        if item.startswith("checkpoint") or item == "merged_model":
                            logger.info(f"Deleting directory: {item_path}")
                            shutil.rmtree(item_path)

                    elif os.path.isfile(item_path) and os.path.basename(item_path) not in PRESERVE_FILES:
                        logger.info(f"Deleting file: {item_path}")
                        os.remove(item_path)


if __name__ == "__main__":
    cleanup_model_dir()
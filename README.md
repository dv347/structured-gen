# structured-gen

## Overview

This repository contains the replication package for the paper Structure-Aware Learning with Minimal Specialized Grammars, currently under review. The implementation supports training, evaluation, and analysis of structure-aware fine-tuning approaches for semantic parsing benchmarks.

## Setup

```bash
pip install -r requirements.txt
```

- This project uses Hugging Face Transformers. You will need to configure your Hugging Face tokens ([docs](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). Some models are gated and require requesting access.
- CUDA-enabled GPUs are recommended for training and evaluation.
- Apple Silicon (MPS backend) is supported for small-scale local experiments.

## Running Experiments

```plaintext
python src/main.py --mode {train|eval} --config CONFIG_FILE --dataset {smcalflow|geoquery|blocks} [--multi_seed]
```

Arguments:
- **`--mode`**:
    - `train`: trains a model and saves it to `models/[dataset]/[output_dir]_[seed]` (where `output_dir` is defined in the config).
    - `eval`: loads a trained model from the `path` specified in the config. Checks both local and Hugging Face Hub. Results are saved to `results/[experiment_name]`.
- **`--config`**: Name of the JSON config file for the experiment. Specify only the relative path:
    - Training configs: `configs/train/`
    - Evaluation configs: `configs/eval/`
- **`--dataset`**: One of `smcalflow`, `geoquery`, or `blocks`.
- **`--multi_seed`** (optional): Runs experiments with 3 random seeds. If not specified, defaults to seed `42`.


Example:
```plaintext
python src/main.py --mode train --config stral.json --dataset smcalflow --multi_seed
```

## Configuration Files

- Sample training and evaluation configs are located in the `/configs` folder. 
- To view all available parameters, see `configs.py`.
- **Note**: Using the `cleanup_weights` parameter will delete model weights after evaluation while preserving predictions, loss plots, and history.

## Code Structure

- Training pipeline implementation: `src/pipelines/training_pipeline.py`
- Inference logic: `src/experiment.py`
- Algorithms for generating minimal, semi-minimal and abstract grammars from inputs are in `src/grammar_generator.py`
- Prompt templates are in `src/prompt.py`
- All datasets used in the paper are provided in the `/data` folder. This includes the generalization splits introduced in the paper for SMCalFlow.
- Corresponding grammars are included in `/grammars`.

The repository includes some additional experimental modules (e.g., `UnifiedPipeline`, in-context learning, grammar embeddings) that are not used in the experiments reported in the paper.
<!-- The repository includes some exploratory features not described in the paper, such as:
- `UnifiedPipeline`
- In-context learning support implementing Grammar Prompting (Wang et al. 2024).
- Grammar embeddings. -->

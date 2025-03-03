import os
from typing import Dict

from datasets import load_dataset
from peft import get_peft_model, LoraConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, EvalPrediction
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

from logger import logger
from paths import DATA_DIR, MODELS_DIR


# TODO: Sort padding token


output_dir = os.path.join(MODELS_DIR, "sft_output")
# r: rank dimension for LoRA update matrices (smaller = more compression)
rank_dimension = 16
# lora_alpha: scaling factor for LoRA layers (higher = stronger adaptation)
lora_alpha = 32
# lora_dropout: dropout probability for LoRA layers (helps prevent overfitting)
lora_dropout = 0.05

lora_config = LoraConfig(
    r=rank_dimension,  # Rank dimension - typically between 4-32
    lora_alpha=lora_alpha,  # LoRA scaling factor - typically 2x rank
    lora_dropout=lora_dropout,  # Dropout probability for LoRA layers
    bias="none",  # Bias type for LoRA. the corresponding biases will be updated during training.
    target_modules="all-linear",  # Which modules to apply LoRA to
    task_type="CAUSAL_LM",  # Task type for model architecture
)

training_args = SFTConfig(
    output_dir=output_dir,
    max_steps=1000,
    per_device_train_batch_size=4,
    bf16=True,
    learning_rate=5e-5,
    logging_steps=10,
    eval_strategy="steps",
    save_steps=100,
    eval_steps=200
)

# Load dataset
train_path = os.path.join(DATA_DIR, "train.json")
val_path = os.path.join(DATA_DIR, "valid_8.json")
dataset = load_dataset("json", data_files={"train": train_path, "validation": val_path}, field="data")

# Load model and tokenizer
model_name = "google/codegemma-2b"
model = model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
tokenizer = AutoTokenizer.from_pretrained(model_name)
lora_model = get_peft_model(model, lora_config)

def preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits, dim=-1)
    masked_preds = torch.where(labels != -100, pred_ids, -100)
    return masked_preds

def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, any]:
    logger.info("===============================================================================================================\n")

    predictions = eval_pred.predictions
    label_ids = eval_pred.label_ids

    predictions = eval_pred.predictions
    label_ids = eval_pred.label_ids

    total_correct = 0
    total_samples = predictions.shape[0]

    counter = 0
    for i in range(total_samples):
        valid_label_ids = label_ids[i][label_ids[i] >= 0]
        valid_predicted_ids = predictions[i][predictions[i] >= 0]

        predicted_text = tokenizer.decode(valid_predicted_ids, skip_special_tokens=True)
        label_text = tokenizer.decode(valid_label_ids, skip_special_tokens=True)

        if counter < 8:
            logger.info(f"y_pred: {predicted_text}")
            logger.info(f"y_true: {label_text}\n")
            counter += 1

        if predicted_text.strip() == label_text.strip():
            total_correct += 1

    accuracy = total_correct / total_samples

    return {"accuracy": accuracy}

# Data collator
def formatting_prompts_func(example) -> str:
    assert type(example['query']) == str
    assert type(example['program']) == str
    return f"### Query: {example['query']}\n ### Program: {example['program']}"
response_template = " ### Program:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    lora_model,
    args=training_args,
    data_collator=collator,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    peft_config=lora_config,
    formatting_func=formatting_prompts_func,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    compute_metrics=compute_metrics
)

results = trainer.evaluate()
logger.info(f"Results before training: {results}")

trainer.train()

results = trainer.evaluate()
logger.info(f"Results after training: {results}")

merged_model = lora_model.merge_and_unload()
model_dir = os.path.join(output_dir, "merged_model")
merged_model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
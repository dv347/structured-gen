import os
import time
from typing import Dict

from datasets import load_dataset
from peft import get_peft_model, LoraConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, EvalPrediction
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

from config import LoraArgs, TrainingArgs, TrainingConfig
from logger import logger
from paths import DATA_DIR, MODELS_DIR


class TrainingPipeline:
    FORMATTERS = {
        "baseline": (lambda e: f"### Query: {e['query']}\n ### Program: {e['program']}", " ### Program:"),
        "bnf_generation": (lambda e: f"### Query: {e['query']}\n ### BNF Grammar: {e['minimal_grammar']}", " ### BNF Grammar:")
    }
    
    def __init__(
        self,
        pipeline_type: str,
        model_name: str,
        train_path: str,
        val_path: str,
        output_dir: str,
        lora_args: LoraArgs,
        training_args: TrainingArgs
    ):
        self.pipeline_type = pipeline_type
        self.formatting_function, self.response_template = TrainingPipeline.FORMATTERS[self.pipeline_type]
        self.model_name = model_name
        self.output_dir = os.path.join(MODELS_DIR, f'{pipeline_type}/{output_dir}')
        self.lora_config = LoraConfig(
            r=lora_args.rank_dimension,  
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.bias,  
            target_modules=lora_args.target_modules,
            task_type="CAUSAL_LM"
        )
        self.training_args = SFTConfig(
            output_dir=self.output_dir,
            max_steps=training_args.max_steps,
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            bf16=training_args.bf16,
            learning_rate=training_args.learning_rate,
            logging_steps=training_args.logging_steps,
            eval_strategy=training_args.eval_strategy,
            save_steps=training_args.save_steps,
            eval_steps=training_args.eval_steps
        )
        self.train_path = os.path.join(DATA_DIR, train_path)
        self.val_path = os.path.join(DATA_DIR, val_path)

        self.dataset = None
        self.model = None
        self.tokenizer = None
        self.lora_model = None

    @classmethod
    def from_config(cls, config: TrainingConfig) -> "TrainingPipeline":
        return cls(**vars(config))
    
    def load_model(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.lora_model = get_peft_model(self.model, self.lora_config)

    def save_model(self) -> None:
        merged_model = self.lora_model.merge_and_unload()
        model_dir = os.path.join(self.output_dir, "merged_model")
        os.makedirs(model_dir, exist_ok=True)
        merged_model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)

    @staticmethod
    def preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor):
        """
        Original Trainer may have a memory leak.
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        pred_ids = torch.argmax(logits, dim=-1)
        masked_preds = torch.where(labels != -100, pred_ids, -100)
        return masked_preds
    
    def get_compute_accuracy_fn(self):
        def compute_accuracy(eval_pred: EvalPrediction) -> Dict[str, any]:
            predictions = eval_pred.predictions
            label_ids = eval_pred.label_ids

            total_correct = 0
            total_samples = predictions.shape[0]

            for i in range(total_samples):
                valid_label_ids = label_ids[i][label_ids[i] >= 0]
                valid_predicted_ids = predictions[i][predictions[i] >= 0]

                predicted_text = self.tokenizer.decode(valid_predicted_ids, skip_special_tokens=True)
                label_text = self.tokenizer.decode(valid_label_ids, skip_special_tokens=True)

                if predicted_text.strip() == label_text.strip():
                    total_correct += 1

            accuracy = total_correct / total_samples

            return {"accuracy": accuracy}
        return compute_accuracy

    def run(self) -> None:
        start_time = time.time()
        logger.info("Loading dataset.")
        dataset = load_dataset("json", data_files={"train": self.train_path, "validation": self.val_path}, field="data")

        logger.info(f"Loading model {self.model_name}.")
        self.load_model()
        
        collator = DataCollatorForCompletionOnlyLM(self.response_template, tokenizer=self.tokenizer)

        trainer = SFTTrainer(
            self.lora_model,
            args=self.training_args,
            data_collator=collator,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            peft_config=self.lora_config,
            formatting_func=self.formatting_function,
            preprocess_logits_for_metrics=TrainingPipeline.preprocess_logits_for_metrics,
            compute_metrics=self.get_compute_accuracy_fn()
        )

        logger.info("Beginning training.")
        trainer.train()

        logger.info(f"Saving model to {self.output_dir}.")
        self.save_model()

        end_time = time.time()
        time_taken_minutes = (end_time - start_time) / 60
        time_taken_hours = time_taken_minutes / 60
        logger.info(f"Time taken: {time_taken_minutes:.2f} minutes ({time_taken_hours:.2f} hours).")
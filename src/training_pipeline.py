import os
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
    def __init__(
        self,
        model_name: str,
        train_path: str,
        val_path: str,
        output_dir: str,
        lora_args: LoraArgs,
        training_args: TrainingArgs
    ):
        self.model_name = model_name
        self.output_dir = os.path.join(MODELS_DIR, output_dir)
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

    @classmethod
    def from_config(cls, config: TrainingConfig) -> "TrainingPipeline":
        return cls(**vars(config))

    def run(self) -> None:
        dataset = load_dataset("json", data_files={"train": self.train_path, "validation": self.val_path}, field="data")

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        lora_model = get_peft_model(model, self.lora_config)

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
        
        def formatting_prompts_func(example) -> str:
            assert type(example['query']) == str
            assert type(example['program']) == str
            return f"### Query: {example['query']}\n ### Program: {example['program']}"
        response_template = " ### Program:"
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

        trainer = SFTTrainer(
            lora_model,
            args=self.training_args,
            data_collator=collator,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            peft_config=self.lora_config,
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
        model_dir = os.path.join(self.output_dir, "merged_model")
        merged_model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
from dataclasses import asdict
import json
import os
import random
import time
from typing import Any, Dict

from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from peft import get_peft_model, LoraConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, EvalPrediction
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

from config import DatasetPaths, LoraArgs, StageConfig, TrainingArgs, TrainingConfig
from grammar_loader import GrammarLoader
from llm import LargeLanguageModel
from logger import logger
from paths import get_dataset_dir, get_model_dir


class TrainingPipeline:
    FORMATTERS = {
        "baseline": (lambda e: f"### Query: {e['query']}\n ### Program: {e['program']}", " ### Program:"),
        "baseline_bnf": (lambda e: f"### Query: {e['query']}\n ### BNF Grammar: {e['grammar']}\n ### Program: {e['program']}", " ### BNF Grammar:"),
        "induction": (lambda e: f"### Query: {e['query']}\n ### BNF Grammar: {e['grammar']}", " ### BNF Grammar:"),
        "structured_reasoning": {
            "default": (lambda e: f"### Query: {e['query']}\n ### BNF Grammar: {e['grammar']}\n ### Program: {e['program']}", " ### Program:"),
            "with_embedding": (lambda e: f"### Query: {e['query']}\n ### BNF Grammar: {e['grammar']}\n ### Grammar Embedding: {e['embedding']}\n ### Program: {e['program']}", " ### Program:")
        }
    }
    
    def __init__(
        self,
        stage_config: StageConfig,
        model_path: str,
        dataset_paths: DatasetPaths,
        output_dir: str,
        lora_args: LoraArgs,
        training_args: TrainingArgs,
        seed: int = 42
    ):
        self.stage = stage_config.name
        self.stage_config = stage_config
        if self.stage == "structured_reasoning":
            variant = "with_embedding" if stage_config.use_embeddings else "default"
            self.formatting_function, self.response_template = TrainingPipeline.FORMATTERS[self.stage][variant]
        else:
            self.formatting_function, self.response_template = TrainingPipeline.FORMATTERS[self.stage]
        self.model_path = LargeLanguageModel.resolve_model_path(model_path)
        self.output_dir = get_model_dir(output_dir)
        self.set_seed(seed)
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
            **asdict(training_args)
        )
        dataset_dir = get_dataset_dir()
        self.train_path = os.path.join(dataset_dir, dataset_paths.train_path)
        self.val_path = os.path.join(dataset_dir, dataset_paths.val_path)

        self.dataset = None
        self.model = None
        self.tokenizer = None
        self.lora_model = None

    @classmethod
    def from_config(cls, config: TrainingConfig) -> "TrainingPipeline":
        return cls(**vars(config))
    
    def set_seed(self, seed) -> None:
        logger.info(f"Setting seed to {seed}.")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

        # Optional — only if you need 100% repeatability
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    
    def load_dataset(self) -> None:
        self.dataset = load_dataset("json", data_files={"train": self.train_path, "validation": self.val_path}, field="data")
        if self.stage in ["baseline_bnf", "induction", "structured_reasoning"]:
            loader = GrammarLoader(grammar_source=self.stage_config.grammar_source)
            grammars_train = loader.load_grammars(self.train_path)
            grammars_val = loader.load_grammars(self.val_path)
            self.dataset["train"] = self.dataset["train"].add_column("grammar", grammars_train)
            self.dataset["validation"] = self.dataset["validation"].add_column("grammar", grammars_val)
            if self.stage == "structured_reasoning" and self.stage_config.use_embeddings:
                embeddings_train = loader.load_encodings(grammars_train)
                embeddings_val = loader.load_encodings(grammars_val)
                self.dataset["train"] = self.dataset["train"].add_column("embedding", embeddings_train)
                self.dataset["validation"] = self.dataset["validation"].add_column("embedding", embeddings_val)
    
    def load_model(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=not torch.backends.mps.is_available())
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
        def compute_accuracy(eval_pred: EvalPrediction) -> Dict[str, Any]:
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
    
    def get_loss_history(self) -> Dict[str, list]:
        training_steps = []
        training_loss = []
        validation_steps = []
        validation_loss = []

        for log in self.trainer.state.log_history:
            if "loss" in log:
                training_steps.append(log["step"])
                training_loss.append(log["loss"])
            if "eval_loss" in log:
                validation_steps.append(log["step"])
                validation_loss.append(log["eval_loss"])

        return {
            "training_steps": training_steps,
            "training_loss": training_loss,
            "validation_steps": validation_steps,
            "validation_loss": validation_loss
        }
    
    def save_loss_history(self, loss_data: Dict[str, list]) -> None:
        loss_file = os.path.join(self.output_dir, "loss_history.json")

        with open(loss_file, "w", encoding="utf-8") as f:
            json.dump(loss_data, f, indent=4)

    def plot_loss_history(self, loss_data: dict) -> None:
        training_steps = loss_data.get("training_steps", [])
        training_loss = loss_data.get("training_loss", [])
        validation_steps = loss_data.get("validation_steps", [])
        validation_loss = loss_data.get("validation_loss", [])

        plt.figure(figsize=(10, 6))
        plt.plot(training_steps, training_loss, label="Training Loss", linestyle="-", marker="o")
        plt.plot(validation_steps, validation_loss, label="Validation Loss", linestyle="--", marker="s")

        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()

        plot_path = os.path.join(self.output_dir, "loss_plot.png")
        plt.savefig(plot_path)
        plt.close()

    def run(self) -> None:
        start_time = time.time()
        logger.info("Loading dataset.")
        self.load_dataset()

        logger.info(f"Loading model {self.model_path}.")
        self.load_model()

        # Strip leading/trailing spaces from the response template for LLaMA-based models.
        # Llama-2 and Code Llama tokenizers handle newlines and whitespace inconsistently,
        # which can break response template matching unless the prefix is stripped.
        if "Llama-2" in self.model_path or "CodeLlama" in self.model_path:
            self.response_template = self.response_template.strip()
        
        collator = DataCollatorForCompletionOnlyLM(self.response_template, tokenizer=self.tokenizer)

        self.trainer = SFTTrainer(
            self.lora_model,
            args=self.training_args,
            data_collator=collator,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['validation'],
            peft_config=self.lora_config,
            formatting_func=self.formatting_function,
            preprocess_logits_for_metrics=TrainingPipeline.preprocess_logits_for_metrics,
            compute_metrics=self.get_compute_accuracy_fn()
        )

        logger.info("Beginning training.")
        self.trainer.train()

        logger.info("Saving loss history.")
        loss_history = self.get_loss_history()
        self.save_loss_history(loss_history)
        self.plot_loss_history(loss_history)

        logger.info(f"Saving model to {self.output_dir}.")
        self.save_model()

        end_time = time.time()
        time_taken_minutes = (end_time - start_time) / 60
        time_taken_hours = time_taken_minutes / 60
        logger.info(f"Time taken: {time_taken_minutes:.2f} minutes ({time_taken_hours:.2f} hours).")
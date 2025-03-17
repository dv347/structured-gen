import os
from typing import List
from tqdm import tqdm

from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList, PreTrainedTokenizerBase, StoppingCriteria, StoppingCriteriaList
import torch

from config import ModelConfig
from paths import MODELS_DIR


class StopOnDoubleNewline(StoppingCriteria):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, lookback: int = 15):
        self.tokenizer = tokenizer
        self.lookback = lookback

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        recent_tokens = input_ids[0, -self.lookback:]
        decoded_text = self.tokenizer.decode(recent_tokens, skip_special_tokens=True)
        return "\n\n" in decoded_text
    

class LogitsDoubleNewline(LogitsProcessor):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Modify logits to force stopping when a double newline (\n\n) appears in the generated sequence.
        """
        batch_size = input_ids.shape[0] 

        for i in range(batch_size):
            last_tokens = input_ids[i][-10:].tolist()  # Get last 10 tokens
            decoded_last_tokens = self.tokenizer.decode(last_tokens, skip_special_tokens=True)

            if "\n\n" in decoded_last_tokens:
                scores[i, :] = -1e9  # Block all tokens
                scores[i, self.tokenizer.eos_token_id] = 1e9  # Force EOS

        return scores


class LargeLanguageModel:
    def __init__(
        self, 
        path: str,
        batch_size: int,
        assistant_model: str | None
    ):
        assert not (batch_size > 1 and assistant_model), "Batch inference does not support assistant models"
        self.batch_size = batch_size
        resolved_path = LargeLanguageModel.resolve_model_path(path)
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_path, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            resolved_path,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        self.generate_kwargs = {
            "max_new_tokens": 500,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        if self.batch_size > 1:
            self.generate_kwargs["logits_processor"] = LogitsProcessorList([LogitsDoubleNewline(self.tokenizer)])
        else:
            self.generate_kwargs["stopping_criteria"] = StoppingCriteriaList([StopOnDoubleNewline(self.tokenizer)])
            if assistant_model:
                self.assistant_model = AutoModelForCausalLM.from_pretrained(assistant_model, torch_dtype=torch.bfloat16, device_map="auto")
                self.generate_kwargs["assistant_model"] = self.assistant_model
                self.generate_kwargs["prompt_lookup_num_tokens"] = 15

    @staticmethod
    def resolve_model_path(path: str) -> str:
        """Check if the model is on Hugging Face. If not, assume it's a local fine-tuned model."""
        try:
            model_info(path)
            return path
        except Exception:
            return os.path.join(MODELS_DIR, path)

    @classmethod
    def from_config(cls, config: ModelConfig) -> "LargeLanguageModel":
        return cls(**vars(config))
    
    @staticmethod
    def process_response(response: str) -> str:
        response = response.split("<|file_separator|>", 1)[0]
        response = response.split("\n\n", 1)[0]
        response = response.strip()
        return response

    def batch_prompt(self, prompts: List[str]) -> List[str]:
        batched_prompts = [prompts[i:i + self.batch_size] for i in range(0, len(prompts), self.batch_size)]
        output = []
        for batch in tqdm(batched_prompts, desc="Generating predictions", unit="batch"):
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True).to(self.model.device)
            input_lengths = [len(input_ids) for input_ids in inputs["input_ids"]]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self.generate_kwargs
                )

            outputs = [output[input_length:] for output, input_length in zip(outputs, input_lengths)]
            output += [LargeLanguageModel.process_response(resp) for resp in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)]
        return output

    def single_prompt(self, prompts: List[str]) -> List[str]:
        output = []
        for prompt in tqdm(prompts, desc="Generating predictions", unit="case"):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            input_length = inputs["input_ids"].shape[1]
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    **self.generate_kwargs
                )

            response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
            output.append(LargeLanguageModel.process_response(response))
        return output
    
    def prompt(self, prompts: List[str]) -> List[str]:
        if self.batch_size > 1:
            return self.batch_prompt(prompts)
        return self.single_prompt(prompts)
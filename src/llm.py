from typing import List
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList, PreTrainedTokenizerBase, StoppingCriteria, StoppingCriteriaList
import torch

from config import ModelConfig


class StopOnDoubleNewline(StoppingCriteria):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, lookback: int = 15):
        self.tokenizer = tokenizer
        self.lookback = lookback

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        recent_tokens = input_ids[0, -self.lookback:]
        decoded_text = self.tokenizer.decode(recent_tokens, skip_special_tokens=True)
        return "\n\n" in decoded_text
    

# class LogitsDoubleNewline(LogitsProcessor):
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
#         """
#         Modify logits to force stopping when a double newline (\n\n) appears in the generated sequence.
#         """
#         batch_size = input_ids.shape[0] 

#         for i in range(batch_size):
#             last_tokens = input_ids[i][-10:].tolist()  # Get last 10 tokens
#             decoded_last_tokens = self.tokenizer.decode(last_tokens, skip_special_tokens=True)

#             if "\n\n" in decoded_last_tokens:
#                 scores[i, :] = -1e9  # Block all tokens
#                 scores[i, self.tokenizer.eos_token_id] = 1e9  # Force EOS

#         return scores


class LargeLanguageModel:
    def __init__(
        self, 
        name: str, 
        assistant_model: str | None
    ):
        model_name = name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        if assistant_model:
            self.assistant_model = AutoModelForCausalLM.from_pretrained(assistant_model, torch_dtype=torch.bfloat16, device_map="auto")
        else:
            self.assistant_model = None

    @classmethod
    def from_config(cls, config: ModelConfig) -> "LargeLanguageModel":
        return cls(config.name, config.assistant_model)

    # def batch_prompt(self, prompts: List[str]) -> List[str]:
    #     self.batch_size = 8 # TODO: Make this configurable
    #     logits_processor = LogitsProcessorList([LogitsDoubleNewline(self.tokenizer)])
    #     batched_prompts = [prompts[i:i + self.batch_size] for i in range(0, len(prompts), self.batch_size)]
    #     output = []

    #     for batch in tqdm(batched_prompts, desc="Generating predictions", unit="batch"):
    #         inputs = self.tokenizer(batch, return_tensors="pt", padding=True).to(self.model.device)
    #         input_lengths = [len(input_ids) for input_ids in inputs["input_ids"]]

    #         with torch.no_grad():
    #             outputs = self.model.generate(
    #                 **inputs,
    #                 max_new_tokens=500,
    #                 pad_token_id=self.tokenizer.eos_token_id,
    #                 logits_processor=logits_processor
    #             )

    #         outputs = [output[input_length:] for output, input_length in zip(outputs, input_lengths)]
    #         output += [resp.strip() for resp in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)]
    #     return output

    def prompt(self, prompt: str) -> str:
        stopping_criteria = StoppingCriteriaList([StopOnDoubleNewline(self.tokenizer)])
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]

        generate_kwargs = {
            "max_new_tokens": 500,
            "pad_token_id": self.tokenizer.eos_token_id,
            "stopping_criteria": stopping_criteria,
            "assistant_model": self.assistant_model,
        }
        if self.assistant_model is not None:
            generate_kwargs["prompt_lookup_num_tokens"] = 15

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                **generate_kwargs
            )

        response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        response = response.split("<|file_separator|>", 1)[0]
        response = response.split("\n\n", 1)[0]
        response = response.strip()
        return response
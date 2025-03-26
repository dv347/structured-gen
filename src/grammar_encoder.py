import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class GrammarEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased", embedding_dim=64): # Bert has a hidden size of 768
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.encoder.config.hidden_size, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, grammar_strs):
        """
        Args:
            grammar_strs (List[str]): batch of BNF grammars as text
        Returns:
            torch.Tensor: [batch_size, embedding_dim]
        """
        tokens = self.tokenizer(grammar_strs, padding=True, truncation=True, return_tensors="pt").to(self.encoder.device)
        outputs = self.encoder(**tokens)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.projection(cls_embedding)

    def encode_single(self, grammar_str: str) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward([grammar_str])[0].cpu()
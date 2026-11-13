"""A causal Transformer workload with deterministic, generated token sequences."""

import torch
from torch import nn

VOCABULARY = 128
SEQUENCE_LENGTH = 64
WIDTH = 128
LAYERS = 2


class CausalTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens = nn.Embedding(VOCABULARY, WIDTH)
        self.positions = nn.Embedding(SEQUENCE_LENGTH, WIDTH)
        layer = nn.TransformerEncoderLayer(
            WIDTH, 4, WIDTH * 4, dropout=0.1, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
        self.output = nn.Linear(WIDTH, VOCABULARY)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(SEQUENCE_LENGTH, SEQUENCE_LENGTH, dtype=torch.bool), diagonal=1),
        )

    def forward(self, inputs):
        positions = torch.arange(inputs.size(1), device=inputs.device)
        hidden = self.tokens(inputs) + self.positions(positions)
        hidden = self.transformer(hidden, mask=self.causal_mask, is_causal=True)
        return self.output(hidden)


def batch(seed, rank, step, local_batch, device):
    generator = torch.Generator().manual_seed(seed + rank * 1_000_003 + step * 97)
    starts = torch.randint(VOCABULARY, (local_batch, 1), generator=generator)
    tokens = (starts + torch.arange(SEQUENCE_LENGTH + 1)) % VOCABULARY
    return tokens[:, :-1].to(device), tokens[:, 1:].to(device)

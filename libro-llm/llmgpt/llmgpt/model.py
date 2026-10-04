import torch
from torch import nn

from llmgpt.config import Config
from llmgpt.model_blocks import LayerNorm, TransformerBlock


class GptNN(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.token_embedding = nn.Embedding(
            config.gpt_config.vocab_size, config.gpt_config.emb_dim
        )

        self.position_embedding = nn.Embedding(
            config.gpt_config.context_length, config.gpt_config.emb_dim
        )

        self.drop_embeddining = nn.Dropout(config.gpt_config.drop_rate)

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(gptr_config=config.gpt_config)
                for _ in range(config.gpt_config.n_layers)
            ]
        )
        self.final_normalization = LayerNorm(normalized_shape=config.gpt_config.emb_dim)

        self.out_head = nn.Linear(
            config.gpt_config.emb_dim, config.gpt_config.vocab_size, bias=False
        )

        self.net = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))

    def forward(self, in_idx: torch.Tensor):
        __batch_size, sequence_length = in_idx.shape
        token_embeddings: torch.Tensor = self.token_embedding(
            in_idx, device=in_idx.device
        )
        position_embeddings: torch.Tensor = self.position_embedding(
            torch.arange(sequence_length, device=in_idx.device)
        )
        intermediate: torch.Tensor = token_embeddings + position_embeddings
        intermediate = self.drop_embeddining(intermediate)
        intermediate = self.transformer_blocks(intermediate)
        intermediate = self.final_normalization(intermediate)
        logits = self.out_head(intermediate)

        return logits

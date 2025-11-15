# SPDX-License-Identifier: Apache-2.0

"""Transformer encoder."""

from __future__ import annotations

from typing import Optional

import flax.nnx as nnx
import jax.numpy as jnp

from .config import TransformerConfig
from .embeddings import TokenEmbedding, PositionalEncoding
from .blocks import EncoderBlock


class TransformerEncoder(nnx.Module):
    """Transformer encoder with token embeddings and stacked encoder blocks."""

    def __init__(self, rngs: nnx.Rngs, config: TransformerConfig):
        self.config = config

        self.token_embed = TokenEmbedding(rngs, config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config.max_len, config.d_model)
        self.dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)

        self.layers = nnx.List(
            [
                EncoderBlock(
                    rngs,
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    d_ff=config.d_ff,
                    dropout_rate=config.dropout_rate,
                    pre_norm=config.use_pre_norm,
                    activation=config.activation,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_ln = nnx.LayerNorm(config.d_model, rngs=rngs)

    def __call__(
        self,
        input_ids: jnp.ndarray,
        *,
        padding_mask: Optional[jnp.ndarray] = None,
        deterministic: Optional[bool] = None,
    ) -> jnp.ndarray:
        """Applies transformer encoder.

        Args:
          input_ids: Token IDs of shape (batch, seq_len).
          padding_mask: Optional padding mask.
          deterministic: If True, disables dropout.

        Returns:
          Encoded representations of shape (batch, seq_len, d_model).
        """
        x = self.token_embed(input_ids)
        x = self.pos_encoding(x)
        x = self.dropout(x, deterministic=deterministic)

        for layer in self.layers:
            x = layer(x, mask=padding_mask, deterministic=deterministic)

        return self.final_ln(x)

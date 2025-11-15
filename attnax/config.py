# SPDX-License-Identifier: Apache-2.0

"""Transformer configuration."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True, kw_only=True)
class TransformerConfig:
    """Transformer encoder configuration.

    Attributes:
      vocab_size: Size of the vocabulary.
      d_model: Dimension of embeddings and hidden states.
      num_heads: Number of attention heads.
      num_layers: Number of transformer blocks.
      d_ff: Dimension of feed-forward inner layer.
      dropout_rate: Dropout probability.
      max_len: Maximum sequence length for positional encoding.
      use_pre_norm: Whether to apply layer norm before attention/FFN.
      use_sinusoidal_positional_embeddings: Use sinusoidal vs learned positions.
      activation: Activation function name ('relu' or 'gelu').
      pad_token_id: Token ID used for padding.
    """

    vocab_size: int
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    d_ff: int = 2048
    dropout_rate: float = 0.1
    max_len: int = 512
    use_pre_norm: bool = True
    use_sinusoidal_positional_embeddings: bool = True
    activation: str = "relu"
    pad_token_id: int = 0

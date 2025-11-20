# SPDX-License-Identifier: Apache-2.0

"""Transformer configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AttentionType(str, Enum):
  """Attention implementation variants.

  STANDARD: Flax default multi-head attention (balanced performance).
  MEMORY_EFFICIENT: Block-wise attention reducing memory from O(n²) to O(n).
  Works on both GPU and TPU.
  FLASH: Optimized attention kernel. Uses JAX's dot_product_attention when
  available, falls back to memory-efficient on TPU.
  LITE: Simplified attention using element-wise product instead of QK^T matmul.

  Better for OOD length generalization on algorithmic tasks. Acts as an
  information gate rather than full attention. Based on ReAct architecture.
  """

  STANDARD = "standard"
  MEMORY_EFFICIENT = "memory_efficient"
  FLASH = "flash"
  LITE = "lite"


@dataclass(frozen=True, kw_only=True)
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
  attention_type: Type of attention implementation to use.
  attention_block_size: Block size for memory-efficient attention.
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
  attention_type: AttentionType = AttentionType.STANDARD
  attention_block_size: int = 512

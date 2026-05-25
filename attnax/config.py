# SPDX-License-Identifier: Apache-2.0

"""Transformer configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class AttentionType(str, Enum):
  """Built-in attention backend selectors.

  Attributes:
    STANDARD: Scaled dot-product attention with :math:`O(n^2)` memory.
    MEMORY_EFFICIENT: Block-wise online-softmax attention.
    FLASH: :func:`jax.nn.dot_product_attention` on GPU,
      :attr:`MEMORY_EFFICIENT` elsewhere.
    PALLAS_FLASH: Pallas-lowered FlashAttention with ``score_mod`` in
      the inner loop; falls back to :attr:`MEMORY_EFFICIENT`.
    LINEAR: Chunkwise-parallel linear attention. Ignores
      ``score_mod``.
    LITE: Element-wise gated attention.

  Ring and paged attention are not enum entries because they require
  additional arguments (``axis_name`` and ``PagedKVCache``); pass
  :func:`~attnax.kernels.ring_attention` or
  :func:`~attnax.kernels.paged_attention` directly via
  ``attention_fn=``.
  """

  STANDARD = "standard"
  MEMORY_EFFICIENT = "memory_efficient"
  FLASH = "flash"
  PALLAS_FLASH = "pallas_flash"
  LINEAR = "linear"
  LITE = "lite"


PosEmbType = Literal["sinusoidal", "rope", "alibi", "none"]
NormKind = Literal["layer", "rms"]
FfActivation = Literal["gelu", "relu", "swiglu", "geglu"]
Pool = Literal["cls", "mean"]


@dataclass(frozen=True, kw_only=True)
class TransformerConfig:
  """Transformer stack hyperparameters.

  Args:
    vocab_size: Vocabulary size.
    d_model: Hidden size.
    num_heads: Number of query heads.
    num_layers: Number of transformer blocks.
    d_ff: Feed-forward hidden width.
    dropout_rate: Dropout probability.
    max_len: Maximum sequence length.
    use_pre_norm: Pre-norm when ``True``, post-norm otherwise.
    pos_emb_type: Positional embedding variant.
    ff_activation: Feed-forward activation.
    norm_type: Normalisation kind.
    pad_token_id: Token id treated as padding.
    attention_type: Attention backend.
    attention_block_size: Block size for memory-efficient, flash and
      pallas_flash backends.
    linear_attention_chunk_size: Chunk size for the linear backend.
    rope_base: RoPE base :math:`\\theta`.
    rope_max_positions: Length of the RoPE table. Defaults to
      ``max_len``.
    num_kv_heads: KV heads for GQA/MQA. Defaults to ``num_heads``.
    attention_window: Causal sliding-window size.
    kv_cache_max_len: Maximum length of the KV cache. Defaults to
      ``max_len``.
  """

  vocab_size: int
  d_model: int = 512
  num_heads: int = 8
  num_layers: int = 6
  d_ff: int = 2048
  dropout_rate: float = 0.1
  max_len: int = 512
  use_pre_norm: bool = True
  pos_emb_type: PosEmbType = "sinusoidal"
  ff_activation: FfActivation = "gelu"
  norm_type: NormKind = "layer"
  pad_token_id: int = 0
  attention_type: AttentionType = AttentionType.STANDARD
  attention_block_size: int = 512
  linear_attention_chunk_size: int = 64
  rope_base: float = 10000.0
  rope_max_positions: int | None = None
  num_kv_heads: int | None = None
  attention_window: int | None = None
  kv_cache_max_len: int | None = None


@dataclass(frozen=True, kw_only=True)
class VisionTransformerConfig:
  """Vision Transformer hyperparameters.

  Defaults reproduce ViT-Base/16 on 224x224 RGB images.

  Args:
    image_size: ``int`` or ``(height, width)``. Must be divisible by
      ``patch_size``.
    patch_size: ``int`` or ``(patch_height, patch_width)``.
    num_channels: Number of image channels.
    num_classes: Output classes. ``None`` disables the head and
      returns the token sequence.
    use_cls_token: Prepend a learnable ``[CLS]`` token.
    pool: Pooling strategy before the classification head.
    d_model: Hidden size.
    num_heads: Number of query heads.
    num_layers: Number of transformer blocks.
    d_ff: Feed-forward hidden width.
    dropout_rate: Dropout probability.
    use_pre_norm: Pre-norm when ``True``, post-norm otherwise.
    ff_activation: Feed-forward activation.
    norm_type: Normalisation kind.
    attention_type: Attention backend.
    attention_block_size: Block size for memory-efficient, flash and
      pallas_flash backends.
    linear_attention_chunk_size: Chunk size for the linear backend.
    num_kv_heads: KV heads for GQA/MQA.
    attention_window: Causal sliding-window size.

  Raises:
    ValueError: If ``image_size`` is not divisible by ``patch_size``,
      or if ``pool == "cls"`` while ``use_cls_token`` is ``False``.

  References:
    Dosovitskiy et al., `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_,
    2020.
  """

  image_size: int | tuple[int, int] = 224
  patch_size: int | tuple[int, int] = 16
  num_channels: int = 3
  num_classes: int | None = None
  use_cls_token: bool = True
  pool: Pool = "cls"
  d_model: int = 768
  num_heads: int = 12
  num_layers: int = 12
  d_ff: int = 3072
  dropout_rate: float = 0.0
  use_pre_norm: bool = True
  ff_activation: FfActivation = "gelu"
  norm_type: NormKind = "layer"
  attention_type: AttentionType = AttentionType.STANDARD
  attention_block_size: int = 512
  linear_attention_chunk_size: int = 64
  num_kv_heads: int | None = None
  attention_window: int | None = None

  def __post_init__(self) -> None:
    img_h, img_w = _pair(self.image_size)
    patch_h, patch_w = _pair(self.patch_size)
    if img_h % patch_h != 0 or img_w % patch_w != 0:
      raise ValueError(
        f"image_size {(img_h, img_w)} must be divisible by patch_size "
        f"{(patch_h, patch_w)} along both axes."
      )
    if self.pool == "cls" and not self.use_cls_token:
      raise ValueError(
        "pool='cls' requires use_cls_token=True; set pool='mean' or "
        "enable the CLS token."
      )


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
  """Returns ``(value, value)`` for an int, or ``value`` for a tuple."""
  if isinstance(value, int):
    return (value, value)
  return value

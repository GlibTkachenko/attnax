# SPDX-License-Identifier: Apache-2.0

"""Attention kernels and transformer components for JAX."""

__version__ = "0.2.0"

from . import kernels
from .attention import MultiHeadAttention
from .blocks import DecoderBlock, EncoderBlock
from .cache import (
  KVLayerCache,
  init_decoder_kv_caches,
  init_decoder_kv_caches_from_config,
  init_kv_layer_cache,
  update_kv_layer_cache,
)
from .config import (
  AttentionType,
  FfActivation,
  NormKind,
  Pool,
  PosEmbType,
  TransformerConfig,
  VisionTransformerConfig,
)
from .embeddings import (
  LearnedPositionalEmbedding,
  PatchEmbedding,
  PositionalEncoding,
  RotaryEmbedding,
  TokenEmbedding,
  apply_rope,
  rope_cos_sin_from_positions,
  rope_cos_sin_table,
  rope_inv_freq,
)
from .feedforward import FeedForward, MixtureOfExperts
from .kernels import (
  AttentionFn,
  MaskMod,
  ScoreMod,
  additive_bias_mod,
  alibi_mod,
  alibi_slopes,
  causal_mod,
  compose_score_mods,
  document_mask_mod,
  flash_attention,
  linear_attention,
  mask_mod_to_boolean_mask,
  mask_mod_to_score_mod,
  memory_efficient_attention,
  paged_attention,
  pallas_flash_attention,
  prefix_lm_mod,
  ring_attention,
  ring_attention_reference,
  sliding_window_mod,
  standard_attention,
)
from .paged_cache import (
  PagedKVCache,
  allocate_blocks,
  append_kv,
  gather_kv,
  init_paged_kv_cache,
)
from .masking import (
  combine_masks,
  make_causal_mask,
  make_document_mask,
  make_padding_mask,
  make_sliding_window_mask,
)
from .norms import RMSNorm, create_norm
from .transformer import TransformerDecoder, TransformerEncoder
from .vision import VisionTransformer

__all__ = [
  "__version__",
  "AttentionType",
  "FfActivation",
  "NormKind",
  "Pool",
  "PosEmbType",
  "MultiHeadAttention",
  "EncoderBlock",
  "DecoderBlock",
  "KVLayerCache",
  "TransformerEncoder",
  "TransformerDecoder",
  "FeedForward",
  "TokenEmbedding",
  "PositionalEncoding",
  "RotaryEmbedding",
  "PatchEmbedding",
  "LearnedPositionalEmbedding",
  "TransformerConfig",
  "VisionTransformerConfig",
  "VisionTransformer",
  "make_padding_mask",
  "make_causal_mask",
  "make_sliding_window_mask",
  "make_document_mask",
  "combine_masks",
  "RMSNorm",
  "create_norm",
  "apply_rope",
  "rope_cos_sin_from_positions",
  "rope_cos_sin_table",
  "rope_inv_freq",
  "init_kv_layer_cache",
  "update_kv_layer_cache",
  "init_decoder_kv_caches",
  "init_decoder_kv_caches_from_config",
  "kernels",
  "AttentionFn",
  "ScoreMod",
  "MaskMod",
  "standard_attention",
  "memory_efficient_attention",
  "flash_attention",
  "linear_attention",
  "ring_attention",
  "ring_attention_reference",
  "paged_attention",
  "pallas_flash_attention",
  "alibi_mod",
  "alibi_slopes",
  "causal_mod",
  "sliding_window_mod",
  "prefix_lm_mod",
  "document_mask_mod",
  "additive_bias_mod",
  "compose_score_mods",
  "mask_mod_to_score_mod",
  "mask_mod_to_boolean_mask",
  "PagedKVCache",
  "init_paged_kv_cache",
  "allocate_blocks",
  "append_kv",
  "gather_kv",
  "MixtureOfExperts",
]

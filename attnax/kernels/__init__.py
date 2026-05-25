# SPDX-License-Identifier: Apache-2.0

"""Attention kernels and score-mod constructors.

Each kernel is a pure JAX function conforming to the
:data:`AttentionFn` protocol and can be passed to
:class:`~attnax.MultiHeadAttention` via ``attention_fn=`` or called
standalone on pre-projected ``(batch, num_heads, seq, head_dim)``
tensors. :mod:`attnax.kernels.score_mods` provides prebuilt
:data:`ScoreMod` and :data:`MaskMod` constructors (ALiBi, sliding
window, prefix-LM, document masks).
"""

from . import score_mods
from ._api import (
  AttentionFn,
  MaskMod,
  ScoreMod,
  compose_score_mods,
  mask_mod_to_boolean_mask,
)
from .attention import (
  flash_attention,
  linear_attention,
  lite_attention,
  memory_efficient_attention,
  ring_attention,
  ring_attention_reference,
  standard_attention,
)
from .paged import paged_attention
from .pallas import pallas_flash_attention
from .score_mods import (
  additive_bias_mod,
  alibi_mod,
  alibi_slopes,
  causal_mod,
  document_mask_mod,
  mask_mod_to_score_mod,
  prefix_lm_mod,
  sliding_window_mod,
)

__all__ = [
  "AttentionFn",
  "MaskMod",
  "ScoreMod",
  "additive_bias_mod",
  "alibi_mod",
  "alibi_slopes",
  "causal_mod",
  "compose_score_mods",
  "document_mask_mod",
  "flash_attention",
  "linear_attention",
  "lite_attention",
  "mask_mod_to_boolean_mask",
  "mask_mod_to_score_mod",
  "memory_efficient_attention",
  "paged_attention",
  "pallas_flash_attention",
  "prefix_lm_mod",
  "ring_attention",
  "ring_attention_reference",
  "score_mods",
  "sliding_window_mod",
  "standard_attention",
]

# SPDX-License-Identifier: Apache-2.0

"""Paged attention against a block-table KV cache."""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

from ..paged_cache import PagedKVCache, gather_kv
from ._api import ScoreMod
from .attention import standard_attention


def paged_attention(
  query: jnp.ndarray,
  cache: PagedKVCache,
  sequence_idx: int,
  *,
  mask: Optional[jnp.ndarray] = None,
  score_mod: Optional[ScoreMod] = None,
  dropout_rng: Optional[jax.Array] = None,
  dropout_rate: float = 0.0,
  deterministic: bool = True,
) -> jnp.ndarray:
  """Attention against a paged KV cache for one sequence.

  Gathers the keys and values pointed to by the block table for
  ``sequence_idx`` and delegates to
  :func:`~attnax.kernels.standard_attention`. Repeats KV heads if
  ``cache.num_kv_heads < num_heads`` (grouped-query attention).

  Args:
    query: Array of shape ``(num_heads, seq_q, head_dim)`` for a
      single sequence. Use :func:`jax.vmap` to batch.
    cache: :class:`PagedKVCache` storing keys and values.
    sequence_idx: Row of the block table to attend against.
    mask: Boolean mask broadcastable to
      ``(num_heads, seq_q, seq_kv)`` where ``seq_kv`` is the current
      sequence length.
    score_mod: Callable applied to the pre-softmax scores; key indices
      are cache positions starting at zero.
    dropout_rng: PRNG key for attention dropout.
    dropout_rate: Attention dropout probability.
    deterministic: If ``True``, disables dropout.

  Returns:
    Array of shape ``(num_heads, seq_q, head_dim)``.
  """
  keys, values, _ = gather_kv(cache, sequence_idx)
  if keys.shape[0] == 0:
    return jnp.zeros_like(query)

  num_heads = query.shape[0]
  q = query[None, ...]
  k = jnp.transpose(keys, (1, 0, 2))[None, ...]
  v = jnp.transpose(values, (1, 0, 2))[None, ...]
  num_kv_heads = k.shape[1]
  if num_kv_heads != num_heads:
    if num_heads % num_kv_heads != 0:
      raise ValueError(
        f"num_heads ({num_heads}) must be divisible by num_kv_heads "
        f"({num_kv_heads}) for paged attention"
      )
    rep = num_heads // num_kv_heads
    k = jnp.repeat(k, rep, axis=1)
    v = jnp.repeat(v, rep, axis=1)

  if mask is not None:
    mask = mask[None, ...]

  out = standard_attention(
    q,
    k,
    v,
    mask=mask,
    score_mod=score_mod,
    dropout_rng=dropout_rng,
    dropout_rate=dropout_rate,
    deterministic=deterministic,
  )
  return out[0]

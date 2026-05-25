# SPDX-License-Identifier: Apache-2.0

"""Protocols and helpers for attention kernels."""

from __future__ import annotations

from typing import Optional, Protocol

import jax
import jax.numpy as jnp


class ScoreMod(Protocol):
  """Callable that modifies pre-softmax attention scores.

  Returns ``scores + bias`` (or any other position-conditioned
  transformation) of the same shape as ``scores``. Position indices
  broadcast over the batch, head, query and key axes respectively.
  """

  def __call__(
    self,
    scores: jnp.ndarray,
    b_idx: jnp.ndarray,
    h_idx: jnp.ndarray,
    q_idx: jnp.ndarray,
    kv_idx: jnp.ndarray,
  ) -> jnp.ndarray:
    """Modifies scores.

    Args:
      scores: Pre-softmax scores of shape
        ``(batch, num_heads, seq_q, seq_kv)``.
      b_idx: Batch indices of shape ``(batch, 1, 1, 1)``.
      h_idx: Head indices of shape ``(1, num_heads, 1, 1)``.
      q_idx: Query positions of shape ``(1, 1, seq_q, 1)``.
      kv_idx: Key positions of shape ``(1, 1, 1, seq_kv)``.

    Returns:
      Modified scores of the same shape as ``scores``.
    """
    ...


class MaskMod(Protocol):
  """Callable that returns a boolean attention mask.

  Reads only the position indices (no scores) and returns a boolean
  array that broadcasts to ``(batch, num_heads, seq_q, seq_kv)``.
  ``True`` means attend.
  """

  def __call__(
    self,
    b_idx: jnp.ndarray,
    h_idx: jnp.ndarray,
    q_idx: jnp.ndarray,
    kv_idx: jnp.ndarray,
  ) -> jnp.ndarray:
    """Returns a boolean mask.

    Args:
      b_idx: Batch indices of shape ``(batch, 1, 1, 1)``.
      h_idx: Head indices of shape ``(1, num_heads, 1, 1)``.
      q_idx: Query positions of shape ``(1, 1, seq_q, 1)``.
      kv_idx: Key positions of shape ``(1, 1, 1, seq_kv)``.

    Returns:
      Boolean array broadcastable to
      ``(batch, num_heads, seq_q, seq_kv)``.
    """
    ...


class AttentionFn(Protocol):
  """Signature for an attention kernel.

  A pure JAX callable mapping pre-projected
  ``(batch, num_heads, seq, head_dim)`` query / key / value tensors to
  the attended output in the same layout. Carries no trainable
  parameters.
  """

  def __call__(
    self,
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    *,
    mask: Optional[jnp.ndarray] = None,
    score_mod: Optional[ScoreMod] = None,
    dropout_rng: Optional[jax.Array] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = True,
  ) -> jnp.ndarray:
    """Computes the attention output.

    Args:
      query: Array of shape ``(batch, num_heads, seq_q, head_dim)``.
      key: Array of shape ``(batch, num_heads, seq_kv, head_dim)``.
      value: Array of shape ``(batch, num_heads, seq_kv, head_dim)``.
      mask: Boolean mask broadcastable to
        ``(batch, num_heads, seq_q, seq_kv)``.
      score_mod: Callable applied to the pre-softmax scores.
      dropout_rng: PRNG key for attention-weight dropout.
      dropout_rate: Dropout probability applied to attention weights.
      deterministic: If ``True``, disables dropout.

    Returns:
      Array of shape ``(batch, num_heads, seq_q, head_dim)``.
    """
    ...


def _position_indices(
  batch: int, num_heads: int, seq_q: int, seq_kv: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Returns ``(b_idx, h_idx, q_idx, kv_idx)`` broadcast index arrays."""
  b_idx = jnp.arange(batch, dtype=jnp.int32)[:, None, None, None]
  h_idx = jnp.arange(num_heads, dtype=jnp.int32)[None, :, None, None]
  q_idx = jnp.arange(seq_q, dtype=jnp.int32)[None, None, :, None]
  kv_idx = jnp.arange(seq_kv, dtype=jnp.int32)[None, None, None, :]
  return b_idx, h_idx, q_idx, kv_idx


def _apply_score_mod(
  scores: jnp.ndarray, score_mod: Optional[ScoreMod]
) -> jnp.ndarray:
  """Applies ``score_mod`` to ``scores`` if non-``None``."""
  if score_mod is None:
    return scores
  batch, num_heads, seq_q, seq_kv = scores.shape
  b_idx, h_idx, q_idx, kv_idx = _position_indices(
    batch, num_heads, seq_q, seq_kv
  )
  return score_mod(scores, b_idx, h_idx, q_idx, kv_idx)


def mask_mod_to_boolean_mask(
  mask_mod: MaskMod,
  *,
  batch: int,
  num_heads: int,
  seq_q: int,
  seq_kv: int,
) -> jnp.ndarray:
  """Materialises a :data:`MaskMod` into a boolean mask tensor.

  Args:
    mask_mod: Callable conforming to :data:`MaskMod`.
    batch: Batch dimension.
    num_heads: Number of attention heads.
    seq_q: Query sequence length.
    seq_kv: Key/value sequence length.

  Returns:
    Boolean array of shape ``(batch, num_heads, seq_q, seq_kv)``.
  """
  b_idx, h_idx, q_idx, kv_idx = _position_indices(
    batch, num_heads, seq_q, seq_kv
  )
  out = mask_mod(b_idx, h_idx, q_idx, kv_idx)
  return jnp.broadcast_to(out, (batch, num_heads, seq_q, seq_kv))


def compose_score_mods(*score_mods: Optional[ScoreMod]) -> Optional[ScoreMod]:
  """Composes :data:`ScoreMod` callables left-to-right.

  ``None`` arguments are skipped. Returns ``None`` when every argument
  is ``None``.

  Args:
    *score_mods: Zero or more :data:`ScoreMod` callables or ``None``.

  Returns:
    A :data:`ScoreMod` applying each non-``None`` argument in order,
    or ``None``.
  """
  active = [m for m in score_mods if m is not None]
  if not active:
    return None
  if len(active) == 1:
    return active[0]

  def composed(
    scores: jnp.ndarray,
    b_idx: jnp.ndarray,
    h_idx: jnp.ndarray,
    q_idx: jnp.ndarray,
    kv_idx: jnp.ndarray,
  ) -> jnp.ndarray:
    for m in active:
      scores = m(scores, b_idx, h_idx, q_idx, kv_idx)
    return scores

  return composed

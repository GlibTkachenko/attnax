# SPDX-License-Identifier: Apache-2.0

"""Prebuilt :data:`ScoreMod` and :data:`MaskMod` constructors."""

from __future__ import annotations

import math

import jax.numpy as jnp

from ._api import MaskMod, ScoreMod, compose_score_mods as compose_score_mods  # noqa: F401


def alibi_slopes(num_heads: int) -> jnp.ndarray:
  """Per-head ALiBi slopes.

  Geometric slope schedule from the ALiBi paper: powers of two for
  ``num_heads`` that is a power of two, interleaved truncation
  otherwise.

  Args:
    num_heads: Number of attention heads.

  Returns:
    ``float32`` array of shape ``(num_heads,)`` with positive slopes
    :math:`m_h`. The bias added to the score between query position
    :math:`i` and key position :math:`j` is :math:`-m_h |i - j|`.

  References:
    Press et al., `Train Short, Test Long: Attention with Linear
    Biases Enables Input Length Extrapolation
    <https://arxiv.org/abs/2108.12409>`_, 2022.
  """

  def _power_of_two_slopes(n: int) -> list[float]:
    start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3)))
    return [start * (start**i) for i in range(n)]

  if (num_heads & (num_heads - 1)) == 0:
    return jnp.asarray(_power_of_two_slopes(num_heads), dtype=jnp.float32)

  closest = 2 ** math.floor(math.log2(num_heads))
  base = _power_of_two_slopes(closest)
  extra = _power_of_two_slopes(2 * closest)[0::2][: num_heads - closest]
  return jnp.asarray(base + extra, dtype=jnp.float32)


def alibi_mod(num_heads: int) -> ScoreMod:
  """ALiBi additive bias score-mod.

  Adds :math:`-m_h |q\\_idx - kv\\_idx|` to the pre-softmax scores,
  where :math:`m_h` is the per-head slope returned by
  :func:`alibi_slopes`.

  Args:
    num_heads: Number of attention heads.

  Returns:
    A :data:`ScoreMod` callable.

  References:
    Press et al., `Train Short, Test Long: Attention with Linear
    Biases Enables Input Length Extrapolation
    <https://arxiv.org/abs/2108.12409>`_, 2022.
  """
  slopes = alibi_slopes(num_heads)[None, :, None, None]

  def _mod(
    scores: jnp.ndarray,
    b_idx: jnp.ndarray,
    h_idx: jnp.ndarray,
    q_idx: jnp.ndarray,
    kv_idx: jnp.ndarray,
  ) -> jnp.ndarray:
    del b_idx, h_idx
    distance = jnp.abs(
      q_idx.astype(scores.dtype) - kv_idx.astype(scores.dtype)
    )
    return scores - slopes.astype(scores.dtype) * distance

  return _mod


def causal_mod() -> ScoreMod:
  """Causal score-mod.

  Sets the score to :math:`-\\infty` whenever ``kv_idx > q_idx``.

  Returns:
    A :data:`ScoreMod` callable.
  """

  def _mod(
    scores: jnp.ndarray,
    b_idx: jnp.ndarray,
    h_idx: jnp.ndarray,
    q_idx: jnp.ndarray,
    kv_idx: jnp.ndarray,
  ) -> jnp.ndarray:
    del b_idx, h_idx
    large_neg = jnp.finfo(scores.dtype).min
    return jnp.where(kv_idx <= q_idx, scores, large_neg)

  return _mod


def sliding_window_mod(
  window_size: int, *, causal: bool = True
) -> ScoreMod:
  """Sliding-window score-mod.

  Restricts attention to keys within ``window_size`` of each query.
  When ``causal`` is ``True``, attends only to positions
  ``q_idx - window_size < kv_idx <= q_idx``; otherwise the window is
  symmetric around the query.

  Args:
    window_size: Maximum query-key distance (positive integer).
    causal: If ``True``, restrict to past keys.

  Returns:
    A :data:`ScoreMod` callable.

  Raises:
    ValueError: If ``window_size <= 0``.
  """
  if window_size <= 0:
    raise ValueError(f"window_size must be positive, got {window_size}")

  def _mod(
    scores: jnp.ndarray,
    b_idx: jnp.ndarray,
    h_idx: jnp.ndarray,
    q_idx: jnp.ndarray,
    kv_idx: jnp.ndarray,
  ) -> jnp.ndarray:
    del b_idx, h_idx
    large_neg = jnp.finfo(scores.dtype).min
    if causal:
      in_window = (kv_idx <= q_idx) & (q_idx - kv_idx < window_size)
    else:
      in_window = jnp.abs(q_idx - kv_idx) < window_size
    return jnp.where(in_window, scores, large_neg)

  return _mod


def prefix_lm_mod(prefix_lengths: jnp.ndarray) -> ScoreMod:
  """Prefix-LM score-mod.

  Attention is bidirectional for ``kv_idx < prefix_lengths[b]`` and
  causal elsewhere.

  Args:
    prefix_lengths: Integer array of shape ``(batch,)`` with the
      per-sequence prefix length.

  Returns:
    A :data:`ScoreMod` callable.
  """
  prefix_lengths = jnp.asarray(prefix_lengths, dtype=jnp.int32)

  def _mod(
    scores: jnp.ndarray,
    b_idx: jnp.ndarray,
    h_idx: jnp.ndarray,
    q_idx: jnp.ndarray,
    kv_idx: jnp.ndarray,
  ) -> jnp.ndarray:
    del h_idx
    pref = prefix_lengths[b_idx[..., 0, 0, 0]][:, None, None, None]
    in_prefix = kv_idx < pref
    causal_ok = kv_idx <= q_idx
    allowed = in_prefix | causal_ok
    large_neg = jnp.finfo(scores.dtype).min
    return jnp.where(allowed, scores, large_neg)

  return _mod


def document_mask_mod(document_ids: jnp.ndarray) -> ScoreMod:
  """Document-boundary score-mod for sequence packing.

  Masks attention between tokens in different documents. A query at
  position :math:`i` only attends to keys at positions :math:`j` with
  ``document_ids[b, j] == document_ids[b, i]``.

  Args:
    document_ids: Integer array of shape ``(batch, seq_len)``.

  Returns:
    A :data:`ScoreMod` callable.
  """
  document_ids = jnp.asarray(document_ids, dtype=jnp.int32)
  same_doc = (
    document_ids[:, :, None] == document_ids[:, None, :]
  )[:, None, :, :]

  def _mod(
    scores: jnp.ndarray,
    b_idx: jnp.ndarray,
    h_idx: jnp.ndarray,
    q_idx: jnp.ndarray,
    kv_idx: jnp.ndarray,
  ) -> jnp.ndarray:
    del b_idx, h_idx
    q_positions = q_idx[0, 0, :, 0]
    kv_positions = kv_idx[0, 0, 0, :]
    allowed = same_doc[:, :, q_positions[:, None], kv_positions[None, :]]
    large_neg = jnp.finfo(scores.dtype).min
    return jnp.where(allowed, scores, large_neg)

  return _mod


def additive_bias_mod(bias: jnp.ndarray) -> ScoreMod:
  """Additive score-mod from a precomputed bias tensor.

  Args:
    bias: Float array broadcastable to
      ``(batch, num_heads, seq_q, seq_kv)``.

  Returns:
    A :data:`ScoreMod` callable.
  """

  def _mod(
    scores: jnp.ndarray,
    b_idx: jnp.ndarray,
    h_idx: jnp.ndarray,
    q_idx: jnp.ndarray,
    kv_idx: jnp.ndarray,
  ) -> jnp.ndarray:
    del b_idx, h_idx, q_idx, kv_idx
    return scores + bias.astype(scores.dtype)

  return _mod


def mask_mod_to_score_mod(mask_mod: MaskMod) -> ScoreMod:
  """Converts a :data:`MaskMod` into an equivalent :data:`ScoreMod`.

  Sets the score to :math:`-\\infty` where ``mask_mod`` returns
  ``False`` and leaves other entries untouched.

  Args:
    mask_mod: Callable conforming to :data:`MaskMod`.

  Returns:
    A :data:`ScoreMod` callable.
  """

  def _mod(
    scores: jnp.ndarray,
    b_idx: jnp.ndarray,
    h_idx: jnp.ndarray,
    q_idx: jnp.ndarray,
    kv_idx: jnp.ndarray,
  ) -> jnp.ndarray:
    keep = mask_mod(b_idx, h_idx, q_idx, kv_idx)
    large_neg = jnp.finfo(scores.dtype).min
    return jnp.where(keep, scores, large_neg)

  return _mod



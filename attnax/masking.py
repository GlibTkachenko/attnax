# SPDX-License-Identifier: Apache-2.0

"""Attention masking utilities."""

from __future__ import annotations

import jax.numpy as jnp


def make_padding_mask(
  input_ids: jnp.ndarray, pad_token_id: int = 0
) -> jnp.ndarray:
  """Returns a key-padding mask.

  Args:
    input_ids: Integer ids of shape ``(batch, seq_len)``.
    pad_token_id: Token id treated as padding.

  Returns:
    Boolean array of shape ``(batch, 1, 1, seq_len)``; ``True`` for
    non-padding positions.
  """
  mask = input_ids != pad_token_id
  return mask[:, None, None, :]


def make_causal_mask(seq_len: int) -> jnp.ndarray:
  """Returns a lower-triangular causal mask.

  Args:
    seq_len: Sequence length.

  Returns:
    Boolean array of shape ``(1, 1, seq_len, seq_len)``; ``True`` at
    positions ``(i, j)`` with ``j <= i``.
  """
  mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
  return mask[None, None, :, :]


def make_sliding_window_mask(
  seq_q: int,
  seq_kv: int | None = None,
  *,
  window_size: int,
  causal: bool = True,
) -> jnp.ndarray:
  """Returns a sliding-window attention mask.

  Args:
    seq_q: Query sequence length.
    seq_kv: Key/value sequence length. Defaults to ``seq_q``.
    window_size: Maximum query-key distance (positive integer).
    causal: If ``True``, restrict to past keys.

  Returns:
    Boolean array of shape ``(1, 1, seq_q, seq_kv)``.

  Raises:
    ValueError: If ``window_size <= 0``.
  """
  if window_size <= 0:
    raise ValueError(f"window_size must be positive, got {window_size}")
  if seq_kv is None:
    seq_kv = seq_q
  q_idx = jnp.arange(seq_q)[:, None]
  kv_idx = jnp.arange(seq_kv)[None, :]
  if causal:
    mask = (kv_idx <= q_idx) & (q_idx - kv_idx < window_size)
  else:
    mask = jnp.abs(q_idx - kv_idx) < window_size
  return mask[None, None, :, :]


def make_document_mask(document_ids: jnp.ndarray) -> jnp.ndarray:
  """Returns a document-boundary mask for sequence packing.

  ``mask[b, 0, i, j]`` is ``True`` iff
  ``document_ids[b, i] == document_ids[b, j]``.

  Args:
    document_ids: Integer array of shape ``(batch, seq_len)``.

  Returns:
    Boolean array of shape ``(batch, 1, seq_len, seq_len)``.
  """
  doc_ids = jnp.asarray(document_ids)
  same_doc = doc_ids[:, :, None] == doc_ids[:, None, :]
  return same_doc[:, None, :, :]


def combine_masks(*masks: jnp.ndarray | None) -> jnp.ndarray | None:
  """Element-wise logical AND of boolean masks.

  ``None`` arguments are skipped; returns ``None`` if every argument
  is ``None``.

  Args:
    *masks: Boolean arrays or ``None``.

  Returns:
    Combined boolean array, or ``None``.
  """
  result = None
  for mask in masks:
    if mask is None:
      continue
    result = mask if result is None else (result & mask)
  return result

# SPDX-License-Identifier: Apache-2.0

"""Paged KV cache backed by a block-table address space."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class PagedKVCache:
  """Paged KV cache storage and per-sequence block table.

  Attributes:
    key_pool: Float array of shape
      ``(num_blocks, block_size, num_kv_heads, head_dim)`` holding the
      physical key blocks.
    value_pool: Float array of the same shape holding the value
      blocks.
    block_table: ``int32`` array of shape
      ``(batch, max_blocks_per_seq)`` mapping each logical block to a
      physical block index. Unused slots are ``-1``.
    seq_lengths: ``int32`` array of shape ``(batch,)`` giving the
      number of valid tokens per sequence.
  """

  key_pool: jnp.ndarray
  value_pool: jnp.ndarray
  block_table: jnp.ndarray
  seq_lengths: jnp.ndarray

  @property
  def num_blocks(self) -> int:
    return int(self.key_pool.shape[0])

  @property
  def block_size(self) -> int:
    return int(self.key_pool.shape[1])

  @property
  def num_kv_heads(self) -> int:
    return int(self.key_pool.shape[2])

  @property
  def head_dim(self) -> int:
    return int(self.key_pool.shape[3])


def init_paged_kv_cache(
  *,
  num_blocks: int,
  block_size: int,
  num_kv_heads: int,
  head_dim: int,
  batch_size: int,
  max_blocks_per_seq: int,
  dtype: jnp.dtype,
) -> PagedKVCache:
  """Allocates an empty :class:`PagedKVCache`.

  Args:
    num_blocks: Total physical blocks in the pool.
    block_size: Tokens per physical block.
    num_kv_heads: Number of KV heads.
    head_dim: Per-head feature dimensionality.
    batch_size: Number of sequences.
    max_blocks_per_seq: Maximum logical blocks per sequence.
    dtype: Buffer dtype.

  Returns:
    :class:`PagedKVCache` with zero pools, ``-1`` block table, and
    ``seq_lengths = 0``.
  """
  key_pool = jnp.zeros(
    (num_blocks, block_size, num_kv_heads, head_dim), dtype=dtype
  )
  value_pool = jnp.zeros_like(key_pool)
  block_table = jnp.full(
    (batch_size, max_blocks_per_seq), -1, dtype=jnp.int32
  )
  seq_lengths = jnp.zeros((batch_size,), dtype=jnp.int32)
  return PagedKVCache(
    key_pool=key_pool,
    value_pool=value_pool,
    block_table=block_table,
    seq_lengths=seq_lengths,
  )


def allocate_blocks(
  cache: PagedKVCache,
  *,
  sequence_idx: int,
  num_new_tokens: int,
  free_block_ids: jnp.ndarray,
) -> tuple[PagedKVCache, int]:
  """Reserves blocks for ``num_new_tokens`` additional tokens.

  Args:
    cache: Current cache.
    sequence_idx: Row of the block table to update.
    num_new_tokens: Tokens about to be appended.
    free_block_ids: 1-D array of free physical block indices.

  Returns:
    Tuple ``(new_cache, blocks_consumed)``.

  Raises:
    ValueError: If ``free_block_ids`` is shorter than the number of
      blocks needed.
  """
  block_size = cache.block_size
  current_len = int(cache.seq_lengths[sequence_idx])
  current_blocks = (current_len + block_size - 1) // block_size
  needed_blocks = (
    (current_len + num_new_tokens + block_size - 1) // block_size
  )
  new_blocks = needed_blocks - current_blocks
  if new_blocks > int(free_block_ids.shape[0]):
    raise ValueError(
      f"free_block_ids has {int(free_block_ids.shape[0])} entries but "
      f"{new_blocks} are needed"
    )

  new_block_table = cache.block_table
  for i in range(new_blocks):
    new_block_table = new_block_table.at[sequence_idx, current_blocks + i].set(
      free_block_ids[i]
    )

  return (
    PagedKVCache(
      key_pool=cache.key_pool,
      value_pool=cache.value_pool,
      block_table=new_block_table,
      seq_lengths=cache.seq_lengths,
    ),
    new_blocks,
  )


def append_kv(
  cache: PagedKVCache,
  sequence_idx: int,
  keys_new: jnp.ndarray,
  values_new: jnp.ndarray,
) -> PagedKVCache:
  """Writes new keys/values for one sequence into the pool.

  The block table for ``sequence_idx`` must be populated for every
  logical block required by the new tokens.

  Args:
    cache: Current cache.
    sequence_idx: Sequence to append to.
    keys_new: Keys of shape ``(chunk, num_kv_heads, head_dim)``.
    values_new: Values of the same shape.

  Returns:
    Updated :class:`PagedKVCache` with ``seq_lengths[sequence_idx]``
    incremented by ``chunk``.

  Raises:
    ValueError: If shapes disagree or the block table is unallocated.
  """
  if keys_new.shape != values_new.shape:
    raise ValueError(
      f"keys_new shape {keys_new.shape} != values_new shape "
      f"{values_new.shape}"
    )
  chunk = keys_new.shape[0]
  block_size = cache.block_size
  start = int(cache.seq_lengths[sequence_idx])

  key_pool = cache.key_pool
  value_pool = cache.value_pool

  for t in range(chunk):
    logical = (start + t) // block_size
    offset = (start + t) % block_size
    physical = int(cache.block_table[sequence_idx, logical])
    if physical < 0:
      raise ValueError(
        f"block_table[{sequence_idx}, {logical}] is unallocated; call "
        "allocate_blocks before append_kv."
      )
    key_pool = key_pool.at[physical, offset].set(keys_new[t])
    value_pool = value_pool.at[physical, offset].set(values_new[t])

  new_seq_lengths = cache.seq_lengths.at[sequence_idx].add(chunk)
  return PagedKVCache(
    key_pool=key_pool,
    value_pool=value_pool,
    block_table=cache.block_table,
    seq_lengths=new_seq_lengths,
  )


def gather_kv(
  cache: PagedKVCache, sequence_idx: int
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
  """Materialises the contiguous KV view for one sequence.

  Args:
    cache: Current cache.
    sequence_idx: Sequence to materialise.

  Returns:
    Tuple ``(keys, values, seq_len)`` where ``keys`` and ``values``
    have shape ``(seq_len, num_kv_heads, head_dim)``.
  """
  seq_len = int(cache.seq_lengths[sequence_idx])
  block_size = cache.block_size
  num_blocks = (seq_len + block_size - 1) // block_size
  if num_blocks == 0:
    head_shape = (0, cache.num_kv_heads, cache.head_dim)
    empty = jnp.zeros(head_shape, dtype=cache.key_pool.dtype)
    return empty, empty, 0

  block_ids = cache.block_table[sequence_idx, :num_blocks]
  keys_blocks = cache.key_pool[block_ids]
  values_blocks = cache.value_pool[block_ids]
  keys = keys_blocks.reshape(
    num_blocks * block_size, cache.num_kv_heads, cache.head_dim
  )[:seq_len]
  values = values_blocks.reshape(
    num_blocks * block_size, cache.num_kv_heads, cache.head_dim
  )[:seq_len]
  return keys, values, seq_len

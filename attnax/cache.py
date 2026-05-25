# SPDX-License-Identifier: Apache-2.0

"""Key/value cache for autoregressive self-attention (inference)."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .config import TransformerConfig


@dataclass(frozen=True)
class KVLayerCache:
  """KV cache buffers for one attention layer.

  Stores keys and values after RoPE rotation in KV-head layout
  (``num_kv_heads``).

  Args:
    keys: Cached keys of shape
      ``(batch, num_kv_heads, max_len, head_dim)``.
    values: Cached values of the same shape as ``keys``.
    length: ``int32`` scalar number of valid positions.
  """

  keys: jnp.ndarray
  values: jnp.ndarray
  length: jnp.ndarray

  @property
  def max_len(self) -> int:
    """Maximum number of tokens the cache can hold."""
    return int(self.keys.shape[2])


def init_kv_layer_cache(
  batch_size: int,
  num_kv_heads: int,
  head_dim: int,
  max_len: int,
  dtype: jnp.dtype,
) -> KVLayerCache:
  """Creates a zero-filled :class:`KVLayerCache`.

  Args:
    batch_size: Batch dimension.
    num_kv_heads: Number of KV heads.
    head_dim: Per-head feature dimensionality.
    max_len: Maximum sequence length.
    dtype: Buffer dtype.

  Returns:
    :class:`KVLayerCache` with zero buffers and ``length = 0``.
  """
  shape = (batch_size, num_kv_heads, max_len, head_dim)
  z = jnp.zeros(shape, dtype=dtype)
  return KVLayerCache(
    keys=z,
    values=z,
    length=jnp.array(0, dtype=jnp.int32),
  )


def update_kv_layer_cache(
  cache: KVLayerCache,
  keys_new: jnp.ndarray,
  values_new: jnp.ndarray,
  start: int,
) -> KVLayerCache:
  """Writes new keys/values into ``cache`` at ``[start, start + chunk)``.

  Args:
    cache: Existing :class:`KVLayerCache`.
    keys_new: Keys of shape ``(batch, num_kv_heads, chunk, head_dim)``.
    values_new: Values of the same shape as ``keys_new``.
    start: Position to write at.

  Returns:
    Updated :class:`KVLayerCache` with ``length = start + chunk``.

  Raises:
    ValueError: If ``start + chunk > cache.max_len``.
  """
  chunk = keys_new.shape[2]
  end = start + chunk
  if end > cache.max_len:
    raise ValueError(
      f"KV cache overflow: end={end} exceeds max_len={cache.max_len}"
    )
  new_keys = cache.keys.at[:, :, start:end, :].set(keys_new)
  new_values = cache.values.at[:, :, start:end, :].set(values_new)
  return KVLayerCache(
    keys=new_keys,
    values=new_values,
    length=jnp.array(end, dtype=jnp.int32),
  )


def init_decoder_kv_caches(
  *,
  num_layers: int,
  batch_size: int,
  num_kv_heads: int,
  head_dim: int,
  max_len: int,
  dtype: jnp.dtype,
) -> tuple[KVLayerCache, ...]:
  """Creates one empty :class:`KVLayerCache` per layer.

  Args:
    num_layers: Number of layers.
    batch_size: Batch dimension.
    num_kv_heads: Number of KV heads.
    head_dim: Per-head feature dimensionality.
    max_len: Maximum cached sequence length.
    dtype: Buffer dtype.

  Returns:
    Tuple of ``num_layers`` :class:`KVLayerCache` objects.
  """
  return tuple(
    init_kv_layer_cache(
      batch_size, num_kv_heads, head_dim, max_len, dtype
    )
    for _ in range(num_layers)
  )


def init_decoder_kv_caches_from_config(
  config: TransformerConfig,
  *,
  batch_size: int,
  max_len: int | None = None,
  dtype: jnp.dtype = jnp.float32,
) -> tuple[KVLayerCache, ...]:
  """Builds per-layer KV caches from a :class:`TransformerConfig`.

  Args:
    config: Transformer hyperparameters.
    batch_size: Batch dimension.
    max_len: Maximum cached length. Defaults to
      ``config.kv_cache_max_len`` if set, otherwise ``config.max_len``.
    dtype: Buffer dtype.

  Returns:
    Tuple of ``config.num_layers`` :class:`KVLayerCache` objects.
  """
  cap = max_len if max_len is not None else (
    config.kv_cache_max_len or config.max_len
  )
  num_kv = config.num_kv_heads or config.num_heads
  head_dim = config.d_model // config.num_heads
  return init_decoder_kv_caches(
    num_layers=config.num_layers,
    batch_size=batch_size,
    num_kv_heads=num_kv,
    head_dim=head_dim,
    max_len=cap,
    dtype=dtype,
  )

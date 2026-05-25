# SPDX-License-Identifier: Apache-2.0

"""Token, positional, and rotary (RoPE) embeddings."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from .config import _pair


class TokenEmbedding(nnx.Module):
  """Token embedding lookup table.

  Args:
    rngs: Flax NNX random key container.
    vocab_size: Vocabulary size.
    d_model: Embedding dimension.
  """

  def __init__(self, rngs: nnx.Rngs, vocab_size: int, d_model: int):
    self.embed = nnx.Embed(
      num_embeddings=vocab_size, features=d_model, rngs=rngs
    )

  def __call__(self, token_ids: jnp.ndarray) -> jnp.ndarray:
    """Looks up embeddings.

    Args:
      token_ids: Integer ids of shape ``(batch, seq_len)``.

    Returns:
      Embeddings of shape ``(batch, seq_len, d_model)``.
    """
    return self.embed(token_ids)


class PositionalEncoding(nnx.Module):
  """Fixed sinusoidal positional encoding.

  .. math::

     \\mathrm{PE}(p, 2i) = \\sin\\!\\left(\\frac{p}{10000^{2i/d}}\\right),
     \\qquad
     \\mathrm{PE}(p, 2i+1) = \\cos\\!\\left(\\frac{p}{10000^{2i/d}}\\right)

  Args:
    max_len: Maximum sequence length.
    d_model: Embedding dimension.

  References:
    Vaswani et al., `Attention Is All You Need
    <https://arxiv.org/abs/1706.03762>`_, 2017.
  """

  def __init__(self, max_len: int, d_model: int):
    self.max_len = max_len
    self.d_model = d_model
    self.positional = self._create_sinusoidal_positions(max_len, d_model)

  @staticmethod
  def _create_sinusoidal_positions(max_len: int, d_model: int) -> jnp.ndarray:
    positions = jnp.arange(max_len)[:, None]
    dims = jnp.arange(d_model)[None, :]
    angle_rates = 1.0 / (10000 ** (2 * (dims // 2) / d_model))
    angles = positions * angle_rates
    return jnp.where(dims % 2 == 0, jnp.sin(angles), jnp.cos(angles))

  def __call__(self, x: jnp.ndarray, start: int = 0) -> jnp.ndarray:
    """Adds the positional encoding to ``x``.

    Args:
      x: Embeddings of shape ``(batch, seq_len, d_model)``.
      start: Offset of the first token, used when decoding past a KV
        cache.

    Returns:
      Array of the same shape as ``x``.
    """
    seq_len = x.shape[1]
    return x + self.positional[None, start : start + seq_len, :]


def rope_inv_freq(head_dim: int, base: float, dtype: jnp.dtype) -> jnp.ndarray:
  """RoPE inverse-frequency vector.

  Returns :math:`b^{-2i / d}` for :math:`i \\in [0, d/2)`.

  Args:
    head_dim: Per-head feature size; must be even.
    base: RoPE base :math:`\\theta`.
    dtype: Output dtype.

  Returns:
    Array of shape ``(head_dim // 2,)``.
  """
  half = head_dim // 2
  return 1.0 / (base ** (jnp.arange(0, half, dtype=dtype) / half))


def rope_cos_sin_table(
  max_seq_len: int,
  head_dim: int,
  base: float,
  dtype: jnp.dtype,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Precomputes the RoPE cos/sin table.

  Args:
    max_seq_len: Number of positions.
    head_dim: Per-head feature size; must be even.
    base: RoPE base :math:`\\theta`.
    dtype: Output dtype.

  Returns:
    ``(cos, sin)`` each of shape ``(max_seq_len, head_dim // 2)``.
  """
  inv_freq = rope_inv_freq(head_dim, base, dtype=dtype)
  t = jnp.arange(max_seq_len, dtype=dtype)
  freqs = jnp.outer(t, inv_freq)
  return jnp.cos(freqs).astype(dtype), jnp.sin(freqs).astype(dtype)


def apply_rope(
  x: jnp.ndarray,
  cos: jnp.ndarray,
  sin: jnp.ndarray,
) -> jnp.ndarray:
  """Applies RoPE rotation in the split-half formulation.

  Splits the last dimension into halves :math:`(x_1, x_2)` and returns

  .. math::

     \\bigl(x_1 \\cos\\theta - x_2 \\sin\\theta,\\;
            x_1 \\sin\\theta + x_2 \\cos\\theta\\bigr).

  Args:
    x: Array with last dimension ``head_dim``.
    cos: Cosine values broadcastable to ``x`` with last dim
      ``head_dim // 2``.
    sin: Sine values of the same shape as ``cos``.

  Returns:
    Array of the same shape as ``x``.

  References:
    Su et al., `RoFormer: Enhanced Transformer with Rotary Position
    Embedding <https://arxiv.org/abs/2104.09864>`_, 2021.
  """
  half = x.shape[-1] // 2
  x1 = x[..., :half]
  x2 = x[..., half:]
  return jnp.concatenate(
    [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
    axis=-1,
  )


def rope_cos_sin_from_positions(
  position_ids: jnp.ndarray,
  head_dim: int,
  base: float,
  table_len: int,
  out_dtype: jnp.dtype,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Gathers RoPE cos/sin values for given positions.

  Args:
    position_ids: Integer positions of shape ``(batch, seq_len)``.
    head_dim: Per-head feature size; must be even.
    base: RoPE base :math:`\\theta`.
    table_len: Length of the precomputed table.
    out_dtype: Output dtype.

  Returns:
    ``(cos, sin)`` each of shape ``(batch, seq_len, head_dim // 2)``.
  """
  cos_t, sin_t = rope_cos_sin_table(table_len, head_dim, base, dtype=out_dtype)
  cos = cos_t[position_ids]
  sin = sin_t[position_ids]
  return cos, sin


class RotaryEmbedding(nnx.Module):
  """Precomputed RoPE cos/sin table.

  Args:
    head_dim: Per-head feature size; must be even.
    max_positions: Length of the precomputed table.
    base: RoPE base :math:`\\theta`.
    dtype: Dtype of the cos/sin buffers.

  Raises:
    ValueError: If ``head_dim`` is odd.
  """

  def __init__(
    self,
    head_dim: int,
    max_positions: int,
    *,
    base: float = 10000.0,
    dtype: jnp.dtype = jnp.float32,
  ):
    if head_dim % 2 != 0:
      raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    self.head_dim = head_dim
    self.max_positions = max_positions
    self.base = base
    cos, sin = rope_cos_sin_table(max_positions, head_dim, base, dtype=dtype)
    self.cos_cached = cos
    self.sin_cached = sin

  def cos_sin_for_positions(
    self, position_ids: jnp.ndarray, *, out_dtype: jnp.dtype
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Gathers cos/sin for the given positions.

    Args:
      position_ids: Integer positions of shape ``(batch, seq_len)``.
      out_dtype: Output dtype.

    Returns:
      ``(cos, sin)`` each of shape
      ``(batch, seq_len, head_dim // 2)``.
    """
    cos = self.cos_cached[position_ids].astype(out_dtype)
    sin = self.sin_cached[position_ids].astype(out_dtype)
    return cos, sin


class PatchEmbedding(nnx.Module):
  """Patchify an image and project each patch to ``d_model``.

  Implemented as a strided 2-D convolution with kernel and stride
  equal to ``patch_size``.

  Args:
    rngs: Flax NNX random key container.
    image_size: ``int`` or ``(height, width)``.
    patch_size: ``int`` or ``(patch_height, patch_width)``.
    num_channels: Number of image channels.
    d_model: Output token dimension.

  Raises:
    ValueError: If ``image_size`` is not divisible by ``patch_size``.
  """

  def __init__(
    self,
    rngs: nnx.Rngs,
    *,
    image_size: int | tuple[int, int],
    patch_size: int | tuple[int, int],
    num_channels: int,
    d_model: int,
  ):
    img_h, img_w = _pair(image_size)
    patch_h, patch_w = _pair(patch_size)
    if img_h % patch_h != 0 or img_w % patch_w != 0:
      raise ValueError(
        f"image_size {(img_h, img_w)} must be divisible by patch_size "
        f"{(patch_h, patch_w)} along both axes."
      )
    self.image_size = (img_h, img_w)
    self.patch_size = (patch_h, patch_w)
    self.num_channels = num_channels
    self.d_model = d_model
    self.grid_size = (img_h // patch_h, img_w // patch_w)
    self.num_patches = self.grid_size[0] * self.grid_size[1]
    self.proj = nnx.Conv(
      in_features=num_channels,
      out_features=d_model,
      kernel_size=self.patch_size,
      strides=self.patch_size,
      padding="VALID",
      rngs=rngs,
    )

  def __call__(self, images: jnp.ndarray) -> jnp.ndarray:
    """Returns patch tokens.

    Args:
      images: Array of shape ``(batch, height, width, channels)``.

    Returns:
      Array of shape ``(batch, num_patches, d_model)``.
    """
    x = self.proj(images)
    batch, h, w, d = x.shape
    return x.reshape(batch, h * w, d)


class LearnedPositionalEmbedding(nnx.Module):
  """Additive learnable positional embedding.

  Initialised with truncated-normal noise (standard deviation
  ``init_std``).

  Args:
    rngs: Flax NNX random key container.
    num_positions: Maximum number of positions.
    d_model: Embedding dimension.
    init_std: Truncated-normal standard deviation.
  """

  def __init__(
    self,
    rngs: nnx.Rngs,
    *,
    num_positions: int,
    d_model: int,
    init_std: float = 0.02,
  ):
    self.num_positions = num_positions
    self.d_model = d_model
    key = rngs.params()
    init = jax.random.truncated_normal(
      key, lower=-2.0, upper=2.0, shape=(num_positions, d_model)
    ) * init_std
    self.embedding = nnx.Param(init)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Adds the learned positional embedding.

    Args:
      x: Array of shape ``(batch, seq_len, d_model)``.

    Returns:
      Array of the same shape as ``x``.

    Raises:
      ValueError: If ``seq_len > num_positions``.
    """
    seq_len = x.shape[1]
    if seq_len > self.num_positions:
      raise ValueError(
        f"Input sequence length {seq_len} exceeds the positional "
        f"embedding table size {self.num_positions}."
      )
    return x + self.embedding[...][None, :seq_len, :]

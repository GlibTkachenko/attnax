# SPDX-License-Identifier: Apache-2.0

"""Multi-head attention with multiple implementation variants."""

from __future__ import annotations

from typing import Optional
import functools

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from .config import AttentionType


def _standard_attention(
  query: jnp.ndarray,
  key: jnp.ndarray,
  value: jnp.ndarray,
  mask: Optional[jnp.ndarray] = None,
  dropout_rng: Optional[jax.Array] = None,
  dropout_rate: float = 0.0,
  deterministic: bool = True,
) -> jnp.ndarray:
  """Standard scaled dot-product attention.

  Args:
    query: Shape (batch, num_heads, seq_q, head_dim).
    key: Shape (batch, num_heads, seq_kv, head_dim).
    value: Shape (batch, num_heads, seq_kv, head_dim).
    mask: Optional mask broadcastable to (batch, num_heads, seq_q, seq_kv).
    dropout_rng: PRNGKey for dropout.
    dropout_rate: Dropout probability.
    deterministic: If True, disables dropout.

  Returns:
    Output of shape (batch, num_heads, seq_q, head_dim).
  """
  depth = query.shape[-1]
  scores = jnp.einsum("bhqd,bhkd->bhqk", query, key) / jnp.sqrt(depth)

  if mask is not None:
    large_neg = jnp.finfo(scores.dtype).min
    scores = jnp.where(mask, scores, large_neg)

  attn_weights = jax.nn.softmax(scores, axis=-1)

  if not deterministic and dropout_rate > 0.0:
    keep_prob = 1.0 - dropout_rate
    if dropout_rng is not None:
      dropout_shape = list(attn_weights.shape)
      keep = jax.random.bernoulli(dropout_rng, keep_prob, dropout_shape)
      attn_weights = jnp.where(keep, attn_weights / keep_prob, 0.0)

  return jnp.einsum("bhqk,bhkd->bhqd", attn_weights, value)


def _memory_efficient_attention(
  query: jnp.ndarray,
  key: jnp.ndarray,
  value: jnp.ndarray,
  mask: Optional[jnp.ndarray] = None,
  dropout_rng: Optional[jax.Array] = None,
  dropout_rate: float = 0.0,
  deterministic: bool = True,
  block_size: int = 512,
) -> jnp.ndarray:
  """Memory-efficient block-sparse attention.

  Computes attention in blocks to reduce memory from O(n²) to O(n).
  Works on both GPU and TPU.

  Args:
    query: Shape (batch, num_heads, seq_q, head_dim).
    key: Shape (batch, num_heads, seq_kv, head_dim).
    value: Shape (batch, num_heads, seq_kv, head_dim).
    mask: Optional mask.
    dropout_rng: PRNGKey for dropout.
    dropout_rate: Dropout probability.
    deterministic: If True, disables dropout.
    block_size: Size of blocks for block-wise computation.

  Returns:
    Output of shape (batch, num_heads, seq_q, head_dim).
  """
  batch, num_heads, seq_q, head_dim = query.shape
  _, _, seq_kv, _ = key.shape

  # If sequence is smaller than block size, use standard attention
  if seq_q <= block_size and seq_kv <= block_size:
    return _standard_attention(
      query, key, value, mask, dropout_rng, dropout_rate, deterministic
    )

  # Block-wise computation
  output = jnp.zeros_like(query)

  num_q_blocks = (seq_q + block_size - 1) // block_size
  num_kv_blocks = (seq_kv + block_size - 1) // block_size

  for q_block in range(num_q_blocks):
    q_start = q_block * block_size
    q_end = min(q_start + block_size, seq_q)
    query_block = query[:, :, q_start:q_end, :]

    block_output = jnp.zeros((batch, num_heads, q_end - q_start, head_dim))
    block_normalizer = jnp.zeros((batch, num_heads, q_end - q_start, 1))

    for kv_block in range(num_kv_blocks):
      kv_start = kv_block * block_size
      kv_end = min(kv_start + block_size, seq_kv)

      key_block = key[:, :, kv_start:kv_end, :]
      value_block = value[:, :, kv_start:kv_end, :]

      # Compute scores for this block
      depth = query_block.shape[-1]
      scores = jnp.einsum("bhqd,bhkd->bhqk", query_block, key_block) / jnp.sqrt(
        depth
      )

      if mask is not None:
        mask_block = mask[:, :, q_start:q_end, kv_start:kv_end]
        large_neg = jnp.finfo(scores.dtype).min
        scores = jnp.where(mask_block, scores, large_neg)

      # Compute block contribution
      block_max = jnp.max(scores, axis=-1, keepdims=True)
      exp_scores = jnp.exp(scores - block_max)
      block_sum = jnp.sum(exp_scores, axis=-1, keepdims=True)

      weighted_values = jnp.einsum("bhqk,bhkd->bhqd", exp_scores, value_block)

      # Update running statistics
      block_output = block_output * block_normalizer + weighted_values
      block_normalizer = block_normalizer + block_sum

    # Normalize and store
    output = output.at[:, :, q_start:q_end, :].set(
      block_output / (block_normalizer + 1e-10)
    )

  if not deterministic and dropout_rate > 0.0:
    keep_prob = 1.0 - dropout_rate
    if dropout_rng is not None:
      keep = jax.random.bernoulli(dropout_rng, keep_prob, output.shape)
      output = jnp.where(keep, output / keep_prob, 0.0)

  return output


def _flash_attention(
  query: jnp.ndarray,
  key: jnp.ndarray,
  value: jnp.ndarray,
  mask: Optional[jnp.ndarray] = None,
  dropout_rng: Optional[jax.Array] = None,
  dropout_rate: float = 0.0,
  deterministic: bool = True,
  block_size: int = 512,
) -> jnp.ndarray:
  """Flash attention using JAX's optimized implementation when available.

  Falls back to memory-efficient attention on TPU.

  Args:
    query: Shape (batch, num_heads, seq_q, head_dim).
    key: Shape (batch, num_heads, seq_kv, head_dim).
    value: Shape (batch, num_heads, seq_kv, head_dim).
    mask: Optional mask.
    dropout_rng: PRNGKey for dropout.
    dropout_rate: Dropout probability.
    deterministic: If True, disables dropout.
    block_size: Fallback block size for memory-efficient attention.

  Returns:
    Output of shape (batch, num_heads, seq_q, head_dim).
  """
  backend = jax.default_backend()

  # On GPU, use JAX's dot_product_attention if available
  if backend == "gpu" and hasattr(jax.nn, "dot_product_attention"):
    # Reshape for dot_product_attention: (batch, seq, num_heads, head_dim)
    batch, num_heads, seq_q, head_dim = query.shape
    q = jnp.transpose(query, (0, 2, 1, 3))
    k = jnp.transpose(key, (0, 2, 1, 3))
    v = jnp.transpose(value, (0, 2, 1, 3))

    if mask is not None:
      # Squeeze mask to (batch, num_heads, seq_q, seq_kv)
      mask = jnp.broadcast_to(mask, (batch, num_heads, seq_q, key.shape[2]))

    output = jax.nn.dot_product_attention(
      q, k, v, bias=None, mask=mask, scale=1.0 / jnp.sqrt(head_dim)
    )
    return jnp.transpose(output, (0, 2, 1, 3))

  # Fall back to memory-efficient attention
  return _memory_efficient_attention(
    query,
    key,
    value,
    mask,
    dropout_rng,
    dropout_rate,
    deterministic,
    block_size,
  )


def _lite_attention(
  query: jnp.ndarray,
  key: jnp.ndarray,
  value: jnp.ndarray,
  gate_proj: nnx.Linear,
  mask: Optional[jnp.ndarray] = None,
  dropout_rng: Optional[jax.Array] = None,
  dropout_rate: float = 0.0,
  deterministic: bool = True,
) -> jnp.ndarray:
  """Lite attention using element-wise product instead of QK^T matmul.

  Simplified attention mechanism that acts as an information gate rather than
  full attention. Better for OOD length generalization on algorithmic tasks.
  Replaces expensive matmul with hadamard product.

  Based on ReAct architecture: https://neel04.github.io/my-website/projects/react/

  Args:
    query: Shape (batch, num_heads, seq_q, head_dim).
    key: Shape (batch, num_heads, seq_kv, head_dim).
    value: Shape (batch, num_heads, seq_kv, head_dim).
    gate_proj: Linear layer for learnable gating (data-dependency).
    mask: Optional mask broadcastable to (batch, num_heads, seq_q, seq_kv).
    dropout_rng: PRNGKey for dropout.
    dropout_rate: Dropout probability.
    deterministic: If True, disables dropout.

  Returns:
    Output of shape (batch, num_heads, seq_q, head_dim).
  """
  batch, num_heads, seq_q, head_dim = query.shape
  _, _, seq_kv, _ = key.shape

  # Element-wise product instead of matmul: Q ⊙ K
  # Broadcast key to query shape for self-attention
  if seq_q == seq_kv:
    attention_scores = query * key  # (batch, num_heads, seq, head_dim)
  else:
    # Cross-attention: expand key to match query length
    key_expanded = jnp.repeat(
      key[:, :, :1, :], seq_q, axis=2
    )  # Repeat first key position
    attention_scores = query * key_expanded

  # Learnable projection for data-dependency
  # Reshape to (batch * num_heads * seq_q, head_dim) for linear layer
  scores_flat = attention_scores.reshape(-1, head_dim)
  gate_scores = gate_proj(scores_flat)  # (batch * num_heads * seq_q, 1)
  gate_scores = gate_scores.reshape(batch, num_heads, seq_q, 1)

  # Softmax to bound attention weights
  attn_weights = jax.nn.softmax(
    gate_scores, axis=-2
  )  # Across sequence dimension

  if mask is not None:
    # Apply mask by setting masked positions to 0
    mask_expanded = mask[..., :1]  # (batch, num_heads, seq_q, 1)
    attn_weights = attn_weights * mask_expanded

  # Apply attention weights to values
  # attn_weights: (batch, num_heads, seq_q, 1)
  # value: (batch, num_heads, seq_kv, head_dim)
  if seq_q == seq_kv:
    output = attn_weights * value  # Element-wise gating
  else:
    # Cross-attention: weight the values
    value_expanded = jnp.repeat(value[:, :, :1, :], seq_q, axis=2)
    output = attn_weights * value_expanded

  if not deterministic and dropout_rate > 0.0:
    keep_prob = 1.0 - dropout_rate
    if dropout_rng is not None:
      keep = jax.random.bernoulli(dropout_rng, keep_prob, output.shape)
      output = jnp.where(keep, output / keep_prob, 0.0)

  return output


class MultiHeadAttentionLayer(nnx.Module):
  """Multi-head attention with configurable implementation."""

  def __init__(
    self,
    rngs: nnx.Rngs,
    *,
    num_heads: int,
    in_features: int,
    qkv_features: Optional[int] = None,
    out_features: Optional[int] = None,
    dropout_rate: float = 0.0,
    broadcast_dropout: bool = True,
    decode: bool = False,
    attention_type: AttentionType = AttentionType.STANDARD,
    attention_block_size: int = 512,
  ):
    if out_features is None:
      out_features = in_features
    if qkv_features is None:
      qkv_features = in_features

    self.num_heads = num_heads
    self.in_features = in_features
    self.qkv_features = qkv_features
    self.out_features = out_features
    self.dropout_rate = dropout_rate
    self.broadcast_dropout = broadcast_dropout
    self.decode = decode
    self.attention_type = attention_type
    self.attention_block_size = attention_block_size

    head_dim = qkv_features // num_heads
    if qkv_features % num_heads != 0:
      raise ValueError(
        f"qkv_features ({qkv_features}) must be divisible by "
        f"num_heads ({num_heads})"
      )

    # Projection layers
    self.query_proj = nnx.Linear(in_features, qkv_features, rngs=rngs)
    self.key_proj = nnx.Linear(in_features, qkv_features, rngs=rngs)
    self.value_proj = nnx.Linear(in_features, qkv_features, rngs=rngs)
    self.output_proj = nnx.Linear(qkv_features, out_features, rngs=rngs)

    self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    # LiteAttention needs a gating projection
    if attention_type == AttentionType.LITE:
      self.gate_proj = nnx.Linear(head_dim, 1, rngs=rngs)
    else:
      self.gate_proj = None

    # Select attention implementation
    if attention_type == AttentionType.STANDARD:
      self._attention_fn = _standard_attention
    elif attention_type == AttentionType.MEMORY_EFFICIENT:
      self._attention_fn = functools.partial(
        _memory_efficient_attention, block_size=attention_block_size
      )
    elif attention_type == AttentionType.FLASH:
      self._attention_fn = functools.partial(
        _flash_attention, block_size=attention_block_size
      )
    elif attention_type == AttentionType.LITE:
      self._attention_fn = functools.partial(
        _lite_attention, gate_proj=self.gate_proj
      )
    else:
      raise ValueError(f"Unknown attention type: {attention_type}")

  def __call__(
    self,
    x: jnp.ndarray,
    *,
    context: Optional[jnp.ndarray] = None,
    mask: Optional[jnp.ndarray] = None,
    deterministic: Optional[bool] = None,
  ) -> jnp.ndarray:
    """Applies multi-head attention.

    Args:
      x: Query input of shape (batch, seq_q, in_features).
      context: Optional context for cross-attention of shape
        (batch, seq_kv, in_features). If None, performs self-attention.
      mask: Optional boolean mask broadcastable to
        (batch, num_heads, seq_q, seq_kv).
      deterministic: If True, disables dropout.

    Returns:
      Output of shape (batch, seq_q, out_features).
    """
    batch, seq_q, _ = x.shape

    # Project queries, keys, values
    query = self.query_proj(x)
    if context is not None:
      key = self.key_proj(context)
      value = self.value_proj(context)
      seq_kv = context.shape[1]
    else:
      key = self.key_proj(x)
      value = self.value_proj(x)
      seq_kv = seq_q

    # Reshape to (batch, num_heads, seq, head_dim)
    head_dim = self.qkv_features // self.num_heads
    query = query.reshape(batch, seq_q, self.num_heads, head_dim)
    query = jnp.transpose(query, (0, 2, 1, 3))

    key = key.reshape(batch, seq_kv, self.num_heads, head_dim)
    key = jnp.transpose(key, (0, 2, 1, 3))

    value = value.reshape(batch, seq_kv, self.num_heads, head_dim)
    value = jnp.transpose(value, (0, 2, 1, 3))

    # Apply attention (dropout_rng not needed - we use nnx.Dropout layer)
    output = self._attention_fn(
      query,
      key,
      value,
      mask=mask,
      dropout_rng=None,
      dropout_rate=self.dropout_rate,
      deterministic=deterministic if deterministic is not None else True,
    )

    # Reshape back to (batch, seq_q, qkv_features)
    output = jnp.transpose(output, (0, 2, 1, 3))
    output = output.reshape(batch, seq_q, self.qkv_features)

    # Output projection and dropout
    output = self.output_proj(output)
    output = self.dropout(output, deterministic=deterministic)

    return output

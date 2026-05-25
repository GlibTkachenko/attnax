# SPDX-License-Identifier: Apache-2.0

"""Vision Transformer."""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from .blocks import EncoderBlock
from .config import VisionTransformerConfig
from .embeddings import LearnedPositionalEmbedding, PatchEmbedding
from .norms import create_norm


class VisionTransformer(nnx.Module):
  """Vision Transformer with optional classification head.

  Patch embedding, optional ``[CLS]`` token, learnable positional
  embedding, ``config.num_layers`` encoder blocks, final
  normalisation, and an optional linear classification head.

  Args:
    rngs: Flax NNX random key container.
    config: Vision Transformer hyperparameters.

  References:
    Dosovitskiy et al., `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale
    <https://arxiv.org/abs/2010.11929>`_, 2020.
  """

  def __init__(self, rngs: nnx.Rngs, config: VisionTransformerConfig):
    self.config = config

    self.patch_embed = PatchEmbedding(
      rngs,
      image_size=config.image_size,
      patch_size=config.patch_size,
      num_channels=config.num_channels,
      d_model=config.d_model,
    )

    if config.use_cls_token:
      key = rngs.params()
      cls_init = jax.random.truncated_normal(
        key, lower=-2.0, upper=2.0, shape=(1, 1, config.d_model)
      ) * 0.02
      self.cls_token = nnx.Param(cls_init)
    else:
      self.cls_token = None

    num_positions = self.patch_embed.num_patches + int(config.use_cls_token)
    self.pos_embed = LearnedPositionalEmbedding(
      rngs, num_positions=num_positions, d_model=config.d_model
    )

    self.dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)

    self.layers = nnx.List(
      [
        EncoderBlock(
          rngs,
          d_model=config.d_model,
          num_heads=config.num_heads,
          d_ff=config.d_ff,
          dropout_rate=config.dropout_rate,
          pre_norm=config.use_pre_norm,
          attention_type=config.attention_type,
          attention_block_size=config.attention_block_size,
          linear_attention_chunk_size=config.linear_attention_chunk_size,
          norm_type=config.norm_type,
          ff_activation=config.ff_activation,
          num_kv_heads=config.num_kv_heads,
          attention_window=config.attention_window,
          use_rope=False,
        )
        for _ in range(config.num_layers)
      ]
    )

    self.final_norm = create_norm(config.norm_type, config.d_model, rngs=rngs)

    if config.num_classes is not None:
      self.head = nnx.Linear(config.d_model, config.num_classes, rngs=rngs)
    else:
      self.head = None

  def __call__(
    self,
    images: jnp.ndarray,
    *,
    deterministic: Optional[bool] = None,
  ) -> jnp.ndarray:
    """Applies the Vision Transformer.

    Args:
      images: Array of shape
        ``(batch, height, width, num_channels)``.
      deterministic: If ``True``, disables dropout.

    Returns:
      Logits of shape ``(batch, num_classes)`` when
      ``config.num_classes`` is set, otherwise tokens of shape
      ``(batch, num_patches + int(use_cls_token), d_model)``.
    """
    x = self.patch_embed(images)

    if self.cls_token is not None:
      batch = x.shape[0]
      cls = jnp.broadcast_to(
        self.cls_token[...], (batch, 1, self.config.d_model)
      )
      x = jnp.concatenate([cls, x], axis=1)

    x = self.pos_embed(x)
    x = self.dropout(x, deterministic=deterministic)

    for layer in self.layers:
      x = layer(x, deterministic=deterministic)

    x = self.final_norm(x)

    if self.head is None:
      return x

    if self.config.pool == "cls":
      pooled = x[:, 0, :]
    else:
      pooled = jnp.mean(x, axis=1)
    return self.head(pooled)

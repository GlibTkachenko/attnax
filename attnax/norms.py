# SPDX-License-Identifier: Apache-2.0

"""Normalization layers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from .config import NormKind


class RMSNorm(nnx.Module):
  """Root-mean-square normalisation with a learnable scale.

  .. math::

     y = \\gamma \\cdot
         \\frac{x}{\\sqrt{\\operatorname{mean}(x^2) + \\epsilon}}

  with mean over the last dimension.

  Args:
    num_features: Size of the last dimension.
    rngs: Flax NNX random key container (unused).
    epsilon: Numerical-stability term.

  References:
    Zhang and Sennrich, `Root Mean Square Layer Normalization
    <https://arxiv.org/abs/1910.07467>`_, 2019.
  """

  def __init__(
    self,
    num_features: int,
    *,
    rngs: nnx.Rngs,
    epsilon: float = 1e-6,
  ):
    del rngs
    self.scale = nnx.Param(jnp.ones((num_features,)))
    self.epsilon = epsilon

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies RMS normalisation along the last axis.

    Args:
      x: Input array.

    Returns:
      Array of the same shape as ``x``.
    """
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed = x * jax.lax.rsqrt(var + self.epsilon)
    return normed * self.scale


def create_norm(
  norm_type: NormKind,
  d_model: int,
  rngs: nnx.Rngs,
) -> nnx.Module:
  """Returns a normalisation module.

  Args:
    norm_type: ``'layer'`` for :class:`flax.nnx.LayerNorm`, ``'rms'``
      for :class:`RMSNorm`.
    d_model: Feature dimension.
    rngs: Flax NNX random key container.

  Returns:
    Normalisation module.

  Raises:
    ValueError: If ``norm_type`` is not ``'layer'`` or ``'rms'``.
  """
  if norm_type == "layer":
    return nnx.LayerNorm(d_model, rngs=rngs)
  if norm_type == "rms":
    return RMSNorm(d_model, rngs=rngs)
  raise ValueError(f"Unknown norm_type: {norm_type!r}")

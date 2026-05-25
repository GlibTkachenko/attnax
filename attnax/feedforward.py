# SPDX-License-Identifier: Apache-2.0

"""Position-wise feed-forward networks (dense and Mixture-of-Experts)."""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from .config import FfActivation


class FeedForward(nnx.Module):
  """Position-wise feed-forward network.

  For ``'gelu'`` and ``'relu'``:

  .. math::

     y = W_2 \\,\\sigma(W_1 x)

  For ``'swiglu'`` and ``'geglu'``:

  .. math::

     y = W_d \\,\\bigl(\\sigma(W_g x) \\odot W_u x\\bigr)

  Args:
    rngs: Flax NNX random key container.
    d_model: Input and output dimension.
    d_ff: Hidden width.
    dropout_rate: Dropout probability.
    ff_activation: Activation variant.

  Raises:
    ValueError: If ``ff_activation`` is not a recognised value.

  References:
    Shazeer, `GLU Variants Improve Transformer
    <https://arxiv.org/abs/2002.05202>`_, 2020.
  """

  def __init__(
    self,
    rngs: nnx.Rngs,
    d_model: int,
    d_ff: int,
    dropout_rate: float = 0.0,
    ff_activation: FfActivation = "gelu",
  ):
    self.ff_activation: FfActivation = ff_activation
    self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    if ff_activation in ("swiglu", "geglu"):
      self.gate_proj = nnx.Linear(d_model, d_ff, rngs=rngs)
      self.up_proj = nnx.Linear(d_model, d_ff, rngs=rngs)
      self.down_proj = nnx.Linear(d_ff, d_model, rngs=rngs)
      self.dense1 = None
      self.dense2 = None
    elif ff_activation in ("gelu", "relu"):
      self.dense1 = nnx.Linear(d_model, d_ff, rngs=rngs)
      self.dense2 = nnx.Linear(d_ff, d_model, rngs=rngs)
      self.gate_proj = None
      self.up_proj = None
      self.down_proj = None
    else:
      raise ValueError(f"Unknown ff_activation: {ff_activation!r}")

  def __call__(
    self, x: jnp.ndarray, *, deterministic: Optional[bool] = None
  ) -> jnp.ndarray:
    """Applies the feed-forward transformation.

    Args:
      x: Input of shape ``(batch, seq_len, d_model)``.
      deterministic: If ``True``, disables dropout.

    Returns:
      Output of shape ``(batch, seq_len, d_model)``.
    """
    if self.ff_activation in ("gelu", "relu"):
      assert self.dense1 is not None and self.dense2 is not None
      h = self.dense1(x)
      h = jax.nn.gelu(h) if self.ff_activation == "gelu" else jax.nn.relu(h)
      h = self.dropout(h, deterministic=deterministic)
      out = self.dense2(h)
      out = self.dropout(out, deterministic=deterministic)
      return out

    assert self.gate_proj is not None
    assert self.up_proj is not None
    assert self.down_proj is not None
    gate = self.gate_proj(x)
    up = self.up_proj(x)
    if self.ff_activation == "swiglu":
      h = jax.nn.silu(gate) * up
    else:
      h = jax.nn.gelu(gate) * up
    h = self.dropout(h, deterministic=deterministic)
    out = self.down_proj(h)
    out = self.dropout(out, deterministic=deterministic)
    return out


def _expert_activation(
  gate: jnp.ndarray, up: jnp.ndarray, ff_activation: FfActivation
) -> jnp.ndarray:
  """Per-expert activation matching :class:`FeedForward`."""
  if ff_activation == "swiglu":
    return jax.nn.silu(gate) * up
  if ff_activation == "geglu":
    return jax.nn.gelu(gate) * up
  if ff_activation == "gelu":
    return jax.nn.gelu(gate)
  if ff_activation == "relu":
    return jax.nn.relu(gate)
  raise ValueError(f"Unknown ff_activation: {ff_activation!r}")


class MixtureOfExperts(nnx.Module):
  """Top-:math:`k` routed Mixture-of-Experts feed-forward.

  Every token independently selects its top-:math:`k` of
  :math:`\\text{num\\_experts}` experts via a learned linear router.
  Each expert runs an MLP / SwiGLU / GeGLU block with width ``d_ff``;
  the outputs are combined weighted by the renormalised top-:math:`k`
  softmax of the router logits. Dispatch is dense — every token is
  multiplied by every expert weight and unselected contributions are
  zeroed by the gate.

  Args:
    rngs: Flax NNX random key container.
    d_model: Input and output dimension.
    d_ff: Per-expert hidden width.
    num_experts: Number of experts.
    top_k: Experts selected per token.
    dropout_rate: Output dropout probability.
    ff_activation: Activation variant.
    router_jitter: Standard deviation of multiplicative Gaussian
      noise on router logits at training time.
    capacity_factor: Reserved for sparse-dispatch implementations.

  Raises:
    ValueError: If ``top_k`` is not in ``[1, num_experts]`` or either
      is non-positive.

  References:
    Fedus et al., `Switch Transformers: Scaling to Trillion Parameter
    Models with Simple and Efficient Sparsity
    <https://arxiv.org/abs/2101.03961>`_, 2022.
  """

  def __init__(
    self,
    rngs: nnx.Rngs,
    *,
    d_model: int,
    d_ff: int,
    num_experts: int,
    top_k: int = 2,
    dropout_rate: float = 0.0,
    ff_activation: FfActivation = "swiglu",
    router_jitter: float = 0.0,
    capacity_factor: float = 1.25,
  ):
    if num_experts <= 0:
      raise ValueError(f"num_experts must be > 0, got {num_experts}")
    if top_k <= 0 or top_k > num_experts:
      raise ValueError(
        f"top_k must satisfy 1 <= top_k <= num_experts; got "
        f"top_k={top_k}, num_experts={num_experts}"
      )

    self.d_model = d_model
    self.d_ff = d_ff
    self.num_experts = num_experts
    self.top_k = top_k
    self.ff_activation: FfActivation = ff_activation
    self.router_jitter = router_jitter
    self.capacity_factor = capacity_factor

    init = nnx.initializers.lecun_normal()
    key_router, key_gate, key_up, key_down = jax.random.split(
      rngs.params(), 4
    )
    self.router = nnx.Param(init(key_router, (d_model, num_experts)))
    self.gate_proj = nnx.Param(init(key_gate, (num_experts, d_model, d_ff)))
    self.up_proj = nnx.Param(init(key_up, (num_experts, d_model, d_ff)))
    self.down_proj = nnx.Param(init(key_down, (num_experts, d_ff, d_model)))

    self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
    self._noise_rngs = rngs

  def __call__(
    self,
    x: jnp.ndarray,
    *,
    deterministic: Optional[bool] = None,
  ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Applies the MoE feed-forward.

    Args:
      x: Input of shape ``(batch, seq_len, d_model)``.
      deterministic: If ``True``, disables dropout and router noise.

    Returns:
      ``(output, aux)``. ``output`` has shape
      ``(batch, seq_len, d_model)``. ``aux`` contains:

      * ``"load_balance_loss"``: auxiliary load-balance loss.
      * ``"router_entropy"``: mean Shannon entropy of the routing
        distribution.
    """
    batch, seq_len, _ = x.shape
    x_flat = x.reshape(batch * seq_len, self.d_model)

    router_logits = jnp.einsum(
      "td,de->te", x_flat, self.router[...]
    )

    is_training = deterministic is False
    if is_training and self.router_jitter > 0.0:
      key = self._noise_rngs.dropout()
      noise = 1.0 + self.router_jitter * jax.random.normal(
        key, router_logits.shape, dtype=router_logits.dtype
      )
      router_logits = router_logits * noise

    router_probs = jax.nn.softmax(router_logits, axis=-1)

    top_k_weights, top_k_indices = jax.lax.top_k(router_probs, self.top_k)
    top_k_weights = top_k_weights / (
      jnp.sum(top_k_weights, axis=-1, keepdims=True) + 1e-9
    )

    one_hot = jax.nn.one_hot(
      top_k_indices, self.num_experts, dtype=router_probs.dtype
    )
    gate = jnp.sum(one_hot * top_k_weights[..., None], axis=1)

    gate_proj_out = jnp.einsum(
      "td,edm->tem", x_flat, self.gate_proj[...]
    )
    up_proj_out = jnp.einsum(
      "td,edm->tem", x_flat, self.up_proj[...]
    )
    hidden = _expert_activation(
      gate_proj_out, up_proj_out, self.ff_activation
    )
    expert_out = jnp.einsum(
      "tem,emd->ted", hidden, self.down_proj[...]
    )

    weighted = expert_out * gate[..., None]
    out_flat = jnp.sum(weighted, axis=1)
    output = out_flat.reshape(batch, seq_len, self.d_model)
    output = self.dropout(output, deterministic=deterministic)

    # Switch Transformer auxiliary load-balance loss (eq. 4).
    expert_fraction = jnp.mean(jnp.any(one_hot > 0, axis=1), axis=0)
    router_mean_prob = jnp.mean(router_probs, axis=0)
    load_balance_loss = self.num_experts * jnp.sum(
      expert_fraction * router_mean_prob
    )

    router_entropy = -jnp.mean(
      jnp.sum(
        router_probs * jnp.log(router_probs + 1e-9), axis=-1
      )
    )

    aux = {
      "load_balance_loss": load_balance_loss,
      "router_entropy": router_entropy,
    }
    return output, aux

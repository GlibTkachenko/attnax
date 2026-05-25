Feed-forward
============

.. currentmodule:: attnax

Position-wise feed-forward layers used inside :class:`EncoderBlock`
and :class:`DecoderBlock`. :class:`FeedForward` is the dense MLP and
gated variant; :class:`MixtureOfExperts` is the sparse top-:math:`k`
routed alternative and is a drop-in replacement at the same call site.

.. autosummary::

   FeedForward
   MixtureOfExperts

Dense feed-forward
------------------

.. autoclass:: FeedForward

Mixture of Experts
------------------

.. autoclass:: MixtureOfExperts
   :members:

Example
^^^^^^^

.. code-block:: python

   import flax.nnx as nnx, jax.numpy as jnp
   from attnax import MixtureOfExperts

   moe = MixtureOfExperts(
       nnx.Rngs(0), d_model=4096, d_ff=14336,
       num_experts=8, top_k=2, ff_activation="swiglu",
   )
   x = jnp.zeros((batch, seq_len, 4096))
   y, aux = moe(x, deterministic=False)
   loss = main_loss + 0.01 * aux["load_balance_loss"]

``aux["load_balance_loss"]`` is the auxiliary load-balance loss and
``aux["router_entropy"]`` is the mean Shannon entropy of the router
distribution.

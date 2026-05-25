Kernels
=======

.. currentmodule:: attnax

The :mod:`attnax.kernels` subpackage contains the attention kernels as
pure JAX functions conforming to the :data:`AttentionFn` protocol, and
:data:`ScoreMod` / :data:`MaskMod` constructors for common attention
biases. Each kernel can be passed to :class:`MultiHeadAttention` via
``attention_fn=`` or called standalone on pre-projected
``(batch, num_heads, seq, head_dim)`` tensors.

.. seealso::

   :doc:`attention` for :class:`MultiHeadAttention` and
   :class:`AttentionType`.

Protocols
---------

.. autoclass:: attnax.kernels.AttentionFn
   :members:

.. autoclass:: attnax.kernels.ScoreMod
   :members:

.. autoclass:: attnax.kernels.MaskMod
   :members:

Built-in kernels
----------------

.. autofunction:: attnax.kernels.standard_attention

.. autofunction:: attnax.kernels.memory_efficient_attention

.. autofunction:: attnax.kernels.flash_attention

.. autofunction:: attnax.kernels.pallas_flash_attention

.. autofunction:: attnax.kernels.ring_attention

.. autofunction:: attnax.kernels.ring_attention_reference

.. autofunction:: attnax.kernels.linear_attention

.. autofunction:: attnax.kernels.paged_attention

Prebuilt score-mods
-------------------

.. autofunction:: attnax.kernels.causal_mod

.. autofunction:: attnax.kernels.sliding_window_mod

.. autofunction:: attnax.kernels.alibi_mod

.. autofunction:: attnax.kernels.alibi_slopes

.. autofunction:: attnax.kernels.prefix_lm_mod

.. autofunction:: attnax.kernels.document_mask_mod

.. autofunction:: attnax.kernels.additive_bias_mod

Composition helpers
-------------------

.. autofunction:: attnax.kernels.compose_score_mods

.. autofunction:: attnax.kernels.mask_mod_to_score_mod

.. autofunction:: attnax.kernels.mask_mod_to_boolean_mask

Examples
--------

Pass any :data:`AttentionFn` to :class:`MultiHeadAttention`:

.. code-block:: python

   from attnax import MultiHeadAttention
   from attnax.kernels import standard_attention
   from attnax.kernels.score_mods import (
       alibi_mod, sliding_window_mod, compose_score_mods,
   )

   attn = MultiHeadAttention(
       rngs,
       num_heads=8,
       in_features=512,
       attention_fn=standard_attention,
       score_mod=compose_score_mods(
           alibi_mod(num_heads=8),
           sliding_window_mod(window_size=4096),
       ),
   )

Per-call score-mods compose with the constructor-supplied mod:

.. code-block:: python

   out = attn(x, score_mod=document_mask_mod(doc_ids), deterministic=True)

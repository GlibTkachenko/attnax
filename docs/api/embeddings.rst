Embeddings
==========

.. currentmodule:: attnax

.. autosummary::

   TokenEmbedding
   PositionalEncoding
   RotaryEmbedding
   apply_rope
   rope_cos_sin_table
   rope_cos_sin_from_positions
   rope_inv_freq
   PatchEmbedding
   LearnedPositionalEmbedding

Token and absolute positions
----------------------------

.. autoclass:: TokenEmbedding

.. autoclass:: PositionalEncoding

Rotary positional embeddings (RoPE)
-----------------------------------

.. autoclass:: RotaryEmbedding

.. autofunction:: apply_rope

.. autofunction:: rope_cos_sin_table

.. autofunction:: rope_cos_sin_from_positions

.. autofunction:: rope_inv_freq

Vision embeddings
-----------------

.. autoclass:: PatchEmbedding

.. autoclass:: LearnedPositionalEmbedding

Attention
=========

.. currentmodule:: attnax

.. autosummary::

   MultiHeadAttention
   AttentionType

Multi-head attention
--------------------

.. autoclass:: MultiHeadAttention

Attention backend
-----------------

.. autoclass:: AttentionType

.. seealso::

   :doc:`kernels` for the attention kernels (``standard_attention``,
   ``memory_efficient_attention``, ``flash_attention``,
   ``pallas_flash_attention``, ``linear_attention``, ``ring_attention``,
   ``paged_attention``, ``lite_attention``), the :data:`AttentionFn`
   protocol, and the prebuilt :data:`ScoreMod` / :data:`MaskMod`
   constructors that are passed to :class:`MultiHeadAttention` via
   ``attention_fn=`` and ``score_mod=``.

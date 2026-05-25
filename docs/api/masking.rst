Masking utilities
=================

.. currentmodule:: attnax

Boolean masks broadcastable to ``(batch, num_heads, seq_q, seq_kv)``.
``True`` means attend.

.. autosummary::

   make_padding_mask
   make_causal_mask
   make_sliding_window_mask
   make_document_mask
   combine_masks

.. autofunction:: make_padding_mask

.. autofunction:: make_causal_mask

.. autofunction:: make_sliding_window_mask

.. autofunction:: make_document_mask

.. autofunction:: combine_masks

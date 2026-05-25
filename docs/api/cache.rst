KV cache
========

.. currentmodule:: attnax

Key/value caches for autoregressive self-attention. Buffers are
post-RoPE and in KV-head layout. Cross-attention caching is not
supported.

.. autosummary::

   KVLayerCache
   init_kv_layer_cache
   update_kv_layer_cache
   init_decoder_kv_caches
   init_decoder_kv_caches_from_config

Per-layer cache
---------------

.. autoclass:: KVLayerCache

.. autofunction:: init_kv_layer_cache

.. autofunction:: update_kv_layer_cache

Whole-model caches
------------------

.. autofunction:: init_decoder_kv_caches

.. autofunction:: init_decoder_kv_caches_from_config

Paged KV cache
--------------

.. autosummary::

   PagedKVCache
   init_paged_kv_cache
   allocate_blocks
   append_kv
   gather_kv

.. autoclass:: PagedKVCache

.. autofunction:: init_paged_kv_cache

.. autofunction:: allocate_blocks

.. autofunction:: append_kv

.. autofunction:: gather_kv

Pair :class:`PagedKVCache` with :func:`attnax.paged_attention` to
attend against a sequence stored in the cache.

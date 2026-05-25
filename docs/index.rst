:github_url: https://github.com/glibtkachenko/attnax/tree/main/docs
:html_theme.sidebar_secondary.remove: 

.. role:: raw-html(raw)
   :format: html

.. raw:: html

   <div class="attnax-landing"></div>

Attnax
======

.. currentmodule:: attnax

Attnax (*Attention for JAX*) is a library of attention kernels and
transformer components for `JAX <https://jax.readthedocs.io/>`_ and
`Flax <https://flax.readthedocs.io/>`_. It ships the layers and
kernels — attention, KV caches, feed-forward, Mixture-of-Experts,
normalisation, positional encodings, and a Vision Transformer — that
you assemble into your own attention-based model.

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: Pluggable kernels

      Standard, memory-efficient, FlashAttention, Pallas, ring, linear,
      and paged attention. Every backend is a pure JAX function with
      the same call signature.

   .. grid-item-card:: Composable biases

      ALiBi, sliding window, prefix-LM, and document masks expressed
      as score-mod callables and composed in a single line.

   .. grid-item-card:: KV caching

      Contiguous and paged KV caches for autoregressive inference
      and batched serving.

   .. grid-item-card:: Vision Transformer

      ViT encoder reusing the same kernels and blocks as the text
      stack.

   .. grid-item-card:: Minimal dependencies

      Depends only on JAX and Flax.

Installation
------------

.. code-block:: bash

   pip install attnax

Or from source:

.. code-block:: bash

   git clone https://github.com/glibtkachenko/attnax.git
   cd attnax
   pip install -e .

Requires Python 3.10+, JAX 0.10.0+, and Flax 0.12.7+.

.. toctree::
   :hidden:
   :maxdepth: 1

   getting_started
   development

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API reference

   api/config
   api/attention
   api/kernels
   api/blocks
   api/transformer
   api/vision
   api/embeddings
   api/feedforward
   api/norms
   api/cache
   api/masking



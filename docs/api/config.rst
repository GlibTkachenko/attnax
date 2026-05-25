Configuration
=============

.. currentmodule:: attnax

.. autosummary::

   TransformerConfig

.. autoclass:: TransformerConfig

String literal aliases
----------------------

.. py:data:: PosEmbType
   :type: Literal["sinusoidal", "rope", "alibi", "none"]

   Positional embedding variant.

   * ``"sinusoidal"``: fixed sinusoidal absolute positions.
   * ``"rope"``: rotary positional embeddings on Q and K.
   * ``"alibi"``: per-head ALiBi additive bias via
     :func:`~attnax.alibi_mod`.
   * ``"none"``: no positional information.

.. py:data:: NormKind
   :type: Literal["layer", "rms"]

   Normalisation variant.

   * ``"layer"``: :class:`flax.nnx.LayerNorm`.
   * ``"rms"``: :class:`RMSNorm`.

.. py:data:: FfActivation
   :type: Literal["gelu", "relu", "swiglu", "geglu"]

   Feed-forward activation.

   * ``"gelu"``, ``"relu"``: two-layer MLP.
   * ``"swiglu"``: ``SiLU(gate(x)) * up(x)`` then ``down``.
   * ``"geglu"``: ``GELU(gate(x)) * up(x)`` then ``down``.

.. py:data:: Pool
   :type: Literal["cls", "mean"]

   :class:`VisionTransformer` pooling strategy.

   * ``"cls"``: take the ``[CLS]`` token.
   * ``"mean"``: mean over all tokens.

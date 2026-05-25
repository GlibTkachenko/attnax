# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration shared by the Attnax test suite.

Two session-wide JAX overrides:

* ``jax_default_matmul_precision = "highest"`` so correctness tests are
  deterministic across CPU, GPU, and TPU. Nvidia >= Ampere defaults to
  TF32 (10-bit mantissa) for ``dot_general``, which is too coarse for
  the 1e-4 tolerances used here.
* A persistent XLA compilation cache under ``.jax_cache/`` so successive
  ``pytest`` invocations reuse already-compiled kernels instead of
  recompiling from scratch. Most of the wall-clock cost of this suite is
  XLA compilation, especially on GPU.

JAX, Flax, and Optax apply the same overrides in their own test suites.
"""

from __future__ import annotations

import pathlib

import jax

jax.config.update("jax_default_matmul_precision", "highest")

_CACHE_DIR = pathlib.Path(__file__).resolve().parent.parent / ".jax_cache"
_CACHE_DIR.mkdir(exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(_CACHE_DIR))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

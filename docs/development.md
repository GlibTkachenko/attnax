# Development

Contributions are welcome: bug fixes, tests, documentation, and features
that stay within the library's scope (attention and transformer
building blocks on JAX / Flax).

If you are unsure whether something fits, open an issue first.

## How to contribute

Use GitHub issues for bugs and design questions, and pull requests for
code or doc changes. The usual workflow is: fork, branch, change, run
the tests and doc build locally, open a PR.

What is expected of a PR:

- New behavior comes with tests under `tests/`.
- If the public API changes, the relevant docstrings and any affected
  pages in `docs/` are updated in the same PR.

The project is distributed under the
[Apache License 2.0](https://github.com/glibtkachenko/attnax/blob/main/LICENSE).
There is no separate contributor agreement: if you open a PR that is
merged, your contribution is included under the same license as the
rest of the repository.

## Local install

From a clone of the repository:

```bash
cd attnax
pip install -e ".[test]"
```

That installs Attnax in editable mode plus `pytest` and `optax` (used
by the training tests and the getting-started notebook).

## Tests

```bash
python -m pytest tests/
```

## Writing docstrings

`attnax` uses Google-style docstrings rendered by `sphinx.ext.napoleon`.
Inline markup *inside* a docstring is reStructuredText, not Markdown
(page-level docs — `getting_started.ipynb`, this page, the README — are
the other way around: Markdown / MyST, never RST).

A typical docstring for a new public function or class:

```python
def my_function(x: jnp.ndarray, ...) -> jnp.ndarray:
  """One-line description ending with a period.

  Longer description if helpful, including mathematical context for
  algorithms.

  Args:
    x: Description of ``x``.
    ...

  Returns:
    Description of the return value.

  Raises:
    ValueError: When the inputs do not satisfy ...

  Example:
    ::

      import jax.numpy as jnp
      from attnax import my_function

      y = my_function(jnp.ones((2, 4)))
  """
```

Existing classes such as
[`TransformerConfig`](https://github.com/glibtkachenko/attnax/blob/main/attnax/config.py)
and
[`MultiHeadAttention`](https://github.com/glibtkachenko/attnax/blob/main/attnax/attention.py)
are good references when in doubt.

## Doctests

Code blocks inside `Example:` sections can be executed as doctests:

```bash
python -m doctest -v attnax/<module>.py
```

`docs/conf.py` configures `doctest_global_setup` so `jax`,
`jax.numpy` as `jnp`, `flax.nnx` as `nnx`, and `attnax` are already
imported in every doctest block — examples should reuse those names
rather than re-importing.

## Building the documentation

Install the docs extra (Sphinx, MyST-NB, the theme, etc.):

```bash
pip install -e ".[docs]"
cd docs
make html
```

Open `docs/_build/html/index.html`. The getting-started guide is a
notebook (`getting_started.ipynb`); the Sphinx build renders it without
re-executing cells (`nb_execution_mode = "off"` in `conf.py`).

Read the Docs uses `docs/requirements.txt` in addition to installing the
package; keep that file aligned with the `[docs]` dependencies in
`pyproject.toml` if you add doc-only packages.

## Code style

The codebase follows the
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
`ruff` enforces the rules wired in `pyproject.toml`.

## AI-generated contributions

AI coding tools are fine to use; the bar for accepting a contribution
is the same as for human-written code. In practice that means a
contribution should not cost the reviewer more time than the
contributor put into preparing it — read what you submit, run the
tests, and check that the docstring template and code style match the
rest of the package before opening the PR.

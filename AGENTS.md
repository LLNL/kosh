# Repository Guidelines

## Project Structure & Module Organization
`kosh/` contains the Python package and CLI entry points. `tests/` holds the pytest suite, including `tests/parallel/` for MPI or resource-heavy cases and `tests/baselines/` for fixture data. Documentation lives under `docs/` and `docs/source/`. Example notebooks are in `examples/` and are used as user-facing reference material.

## Build, Test, and Development Commands
- `pip install .` installs the package in a local environment.
- `pytest tests/test_kosh_*.py` runs the main serial test suite.
- `pytest -s --cov=kosh tests/non_parallel/test_kosh_*.py` runs non-parallel tests with coverage.
- `pytest tests/parallel/test_kosh_*.py` runs parallel tests; use the site scheduler or MPI launcher when required.
- `make -C docs html` builds the Sphinx docs into `docs/build/html`.
- `make -C docs linkcheck` checks documentation links.

On LC machines, activate Python with `source ${HOME}/${SYS_TYPE}_venv/bin/activate` before running commands.

## Coding Style & Naming Conventions
Use Python 3.8+ code, 4-space indentation, and keep functions and modules aligned with existing `kosh` naming patterns. Follow the repository’s flake8 expectations; `tests/test_kosh_flake8.py` reflects the style baseline. Prefer descriptive test names like `test_kosh_store.py` and keep notebook filenames consistent with the existing `Example_*.ipynb` pattern. Avoid splitting logic into very small functions that are only a couple of lines long unless that improves clarity or reuse. Document functions and public APIs with Sphinx-style docstrings.

## Testing Guidelines
Add a test for every code change. Place new unit tests alongside the relevant module in `tests/`, and add or update fixtures under `tests/baselines/` only when needed. If a change affects user workflows, update the docs and extend an existing notebook or add a new one when that is the clearest way to demonstrate the behavior.
If the change adds or alters Kosh skill behavior, update `kosh/skills/kosh/SKILL.md` in the same change and add or update a Codex skill test such as `tests/test_kosh_skill_codex_smoke.py`.

## Commit & Pull Request Guidelines
Recent commits use short, imperative subjects, often in lowercase, such as `fix dataset record type` or `setuptools-scm and helper`. Keep commits focused and readable. Pull requests should target the `develop` branch, include the relevant test results, and note any docs or notebook updates.
Do not commit or push changes unless the user explicitly asks; leave version control actions to the user.

## Contributor Notes
Avoid changing generated artifacts unless they are part of the requested fix. If a feature touches CLI behavior, data loading, or workflow examples, verify the corresponding documentation and example notebook paths under `docs/source/` and `examples/`.

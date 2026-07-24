# Codex Skill: Kosh

Kosh ships a Codex skill folder inside the Python package. It helps Codex use
Kosh to store results, attach files, query them later, and build loaders,
transformers, and operators.

## Locate the skill in your Python install

```bash
python -c "import pathlib,kosh; print(pathlib.Path(kosh.__file__).parent / 'skills' / 'kosh')"
```

## Install the skill into Codex

Set `CODEX_HOME` if needed. Many setups use `~/.codex`.

```bash
export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
SKILL_SRC="$(python -c 'import pathlib,kosh; print(pathlib.Path(kosh.__file__).parent / "skills" / "kosh")')"
mkdir -p "$CODEX_HOME/skills"
DEST="$CODEX_HOME/skills/kosh"

if [ -z "$SKILL_SRC" ] || [ ! -d "$SKILL_SRC" ]; then
  echo "ERROR: SKILL_SRC is not a directory: '$SKILL_SRC'" 1>&2
  exit 2
fi

rm -rf "$DEST"
ln -sfn "$SKILL_SRC" "$DEST"
```

The installed skill will be at `"$CODEX_HOME/skills/kosh"`.

If you install Kosh with `pip install .` or `pip install -e .`, the symlink
will continue to point at the packaged skill in your environment.

## Optional smoke test

To run the live Codex smoke test from this repository:

```bash
pytest tests/test_kosh_skill_codex_smoke.py
```

The test stages a temporary `HOME`, copies in `~/.codex/config.toml`, and
looks for a LLameMe key in `./.llamame-api-key.txt` or
`~/.llamame-api-key.txt`. If the key is missing, the test exits early with a
skip message.

It defaults to this Codex command unless `KOSH_CODEX_SMOKE_CMD` is set:

```bash
/usr/workspace/sduser/mapp/sandbox/codex-gemma-4 exec --skip-git-repo-check --ephemeral --json -
```

The smoke task now runs in three phases: store and ensemble creation, file
association plus `alias_feature` checks, and then a loader/operator pass that
verifies the final math. Each phase is checked before the next one runs.

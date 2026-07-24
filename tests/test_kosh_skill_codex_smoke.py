import json
from pathlib import Path
import os
import shutil
import subprocess
import textwrap
from shutil import copy2

import h5py
import numpy as np
import pytest

import kosh


SKILL_NAME = "kosh"
DEFAULT_SMOKE_CMD = (
    "/usr/workspace/sduser/mapp/sandbox/codex-gemma-4 exec "
    "--skip-git-repo-check --ephemeral --json -"
)
STORE_NAME = "kosh_smoke_store.db"
CSV_NAME = "alpha.csv"
HDF5_NAME = "beta.h5"
PHASE1_REPORT = "phase1_report.json"
PHASE2_REPORT = "phase2_report.json"
PHASE3_REPORT = "phase3_report.json"
CSV_HEADERS = ["COL_A", "COL_B", "COL_C"]
HDF5_FEATURES = ["col_a", "col_b", "col_c"]
CSV_VALUES = np.array([[1, 10, 100], [2, 20, 200]])
HDF5_VALUES = np.array([[3, 30, 300], [4, 40, 400]])
CSV_FIRST_COLUMN = CSV_VALUES[:, 0].tolist()
HDF5_FIRST_COLUMN = HDF5_VALUES[:, 0].tolist()
EXPECTED_FIRST_FEATURE_SUM = [4, 6]


def _skill_source() -> Path:
    return Path(kosh.__file__).resolve().parent / "skills" / SKILL_NAME


def _install_skill(codex_home: Path) -> Path:
    skill_src = _skill_source()
    skill_dst = codex_home / "skills" / SKILL_NAME
    skill_dst.parent.mkdir(parents=True, exist_ok=True)

    if skill_dst.exists() or skill_dst.is_symlink():
        if skill_dst.resolve() != skill_src.resolve():
            if skill_dst.is_dir() and not skill_dst.is_symlink():
                shutil.rmtree(skill_dst)
            else:
                skill_dst.unlink()

    if not skill_dst.exists():
        skill_dst.symlink_to(skill_src, target_is_directory=True)

    return skill_dst


def _prepare_wrapper_home(home: Path) -> None:
    config_src = Path.home() / ".codex" / "config.toml"
    key_src = Path.cwd() / ".llamame-api-key.txt"
    if not key_src.is_file():
        key_src = Path.home() / ".llamame-api-key.txt"
    if not config_src.is_file():
        pytest.skip(f"Missing Codex config file: {config_src}")
    if not key_src.is_file():
        pytest.skip(f"Missing LLameMe key file: {key_src}")

    config_dst = home / ".codex" / "config.toml"
    key_dst = home / ".llamame-api-key.txt"
    config_dst.parent.mkdir(parents=True, exist_ok=True)
    copy2(config_src, config_dst)
    copy2(key_src, key_dst)


def _write_fixtures(workdir: Path) -> tuple[Path, Path]:
    csv_path = workdir / CSV_NAME
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(CSV_HEADERS) + "\n")
        for row in CSV_VALUES.tolist():
            handle.write(",".join(str(value) for value in row) + "\n")
    hdf5_path = workdir / HDF5_NAME
    with h5py.File(hdf5_path, "w") as handle:
        for index, feature in enumerate(HDF5_FEATURES):
            handle.create_dataset(feature, data=HDF5_VALUES[:, index])
    return csv_path, hdf5_path


def _read_report(report_path: Path) -> dict:
    return json.loads(report_path.read_text(encoding="utf-8"))


def _run_codex_step(smoke_cmd: str, env: dict, workdir: Path, prompt_name: str, prompt_text: str):
    prompt_file = workdir / prompt_name
    prompt_file.write_text(prompt_text, encoding="utf-8")
    result = subprocess.run(
        smoke_cmd,
        shell=True,
        env=env,
        cwd=workdir,
        capture_output=True,
        text=True,
        input=prompt_text,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    return result


def test_codex_skill_smoke(tmp_path):
    smoke_cmd = os.environ.get("KOSH_CODEX_SMOKE_CMD") or DEFAULT_SMOKE_CMD

    wrapper_home = Path(os.environ.get("KOSH_CODEX_HOME") or str(tmp_path / "codex-home"))
    if wrapper_home.exists() and not wrapper_home.is_dir():
        pytest.skip(f"KOSH_CODEX_HOME must be a directory, got: {wrapper_home}")
    _prepare_wrapper_home(wrapper_home)
    codex_home = wrapper_home / ".codex"
    skill_dst = _install_skill(codex_home)

    assert (skill_dst / "SKILL.md").is_file()
    assert (skill_dst / "agents" / "openai.yaml").is_file()

    workdir = tmp_path / "codex-work"
    workdir.mkdir()
    csv_path, hdf5_path = _write_fixtures(workdir)
    assert csv_path.is_file()
    assert hdf5_path.is_file()
    store_path = workdir / STORE_NAME

    env = os.environ.copy()
    env["HOME"] = str(wrapper_home)
    env["KOSH_CODEX_SKILL_DIR"] = str(skill_dst)
    env["LLAMAME_API_KEY"] = (wrapper_home / ".llamame-api-key.txt").read_text(encoding="utf-8").strip()

    phase1_prompt = textwrap.dedent(
        f"""\
        Using the installed Kosh skill, do only this step and stop:
        - Create or open a store named {STORE_NAME} in the current directory.
        - Create three datasets named alpha, beta, and gamma.
        - Put alpha and beta in an ensemble named paired.
        - Find beta from the store and alpha from the ensemble.
        - Write {PHASE1_REPORT} with keys store, datasets, ensemble,
          found_from_store, and found_from_ensemble.
        - Print one short completion line.
        """
    )
    _run_codex_step(smoke_cmd, env, workdir, "phase1_prompt.txt", phase1_prompt)
    phase1_report = _read_report(workdir / PHASE1_REPORT)
    assert phase1_report["store"] == STORE_NAME
    assert phase1_report["datasets"] == ["alpha", "beta", "gamma"]
    assert phase1_report["ensemble"] == "paired"
    assert phase1_report["found_from_store"] == "beta"
    assert phase1_report["found_from_ensemble"] == "alpha"

    assert store_path.is_file(), f"Missing store file: {store_path}"
    store = kosh.connect(str(store_path), read_only=True)
    try:
        assert [ds.name for ds in store.find(name="alpha")] == ["alpha"]
        assert [ds.name for ds in store.find(name="beta")] == ["beta"]
        assert [ds.name for ds in store.find(name="gamma")] == ["gamma"]
        ensembles = list(store.find_ensembles(name="paired"))
        assert len(ensembles) == 1
        assert ensembles[0].name == "paired"
        assert sorted(ds.name for ds in ensembles[0].get_members()) == ["alpha", "beta"]
    finally:
        store.close()

    phase2_prompt = textwrap.dedent(
        f"""\
        Using the existing store and datasets, do only this step and stop:
        - Create {CSV_NAME} with headers {CSV_HEADERS} and rows {CSV_VALUES.tolist()}.
        - Create {HDF5_NAME} with feature names {HDF5_FEATURES} and values
          {HDF5_VALUES.tolist()}.
        - Associate {CSV_NAME} with alpha and {HDF5_NAME} with beta.
        - Set alias_feature on alpha so lowercase names map to the uppercase CSV
          headers, and set alias_feature on beta so uppercase names map to the
          lowercase HDF5 features.
        - Verify alpha features and beta features after association.
        - Read alpha["col_a"] and beta["COL_A"].
        - Write {PHASE2_REPORT} with keys csv, hdf5, alpha_features,
          beta_features, alpha_alias_lookup, and beta_alias_lookup.
        - Print one short completion line.
        """
    )
    _run_codex_step(smoke_cmd, env, workdir, "phase2_prompt.txt", phase2_prompt)
    phase2_report = _read_report(workdir / PHASE2_REPORT)
    assert phase2_report["csv"] == CSV_NAME
    assert phase2_report["hdf5"] == HDF5_NAME
    assert phase2_report["alpha_features"] == CSV_HEADERS
    assert phase2_report["beta_features"] == HDF5_FEATURES
    assert phase2_report["alpha_alias_lookup"] == CSV_FIRST_COLUMN
    assert phase2_report["beta_alias_lookup"] == HDF5_FIRST_COLUMN

    store = kosh.connect(str(store_path), read_only=True)
    try:
        alpha = next(store.find(name="alpha"))
        beta = next(store.find(name="beta"))
        assert alpha.list_features() == CSV_HEADERS
        assert beta.list_features() == HDF5_FEATURES
        np.testing.assert_array_equal(alpha["col_a"][:], CSV_FIRST_COLUMN)
        np.testing.assert_array_equal(beta["COL_A"][:], HDF5_FIRST_COLUMN)
    finally:
        store.close()

    phase3_prompt = textwrap.dedent(
        f"""\
        Using the existing store, datasets, and associated files, do only this
        step and stop:
        - Define a DualFormatLoader that can read both CSV and HDF5 sources and
          return feature data as either a numpy array or a Python list.
        - Define an AddOperator that adds the first-column values from alpha and
          beta with the right numeric result.
        - Use the loader and operator to add alpha["col_a"] and beta["COL_A"].
        - Write {PHASE3_REPORT} with keys loader, operator, sum_numpy, and
          sum_list.
        - Make sum_numpy and sum_list equal to {EXPECTED_FIRST_FEATURE_SUM}.
        - Print one short completion line.
        """
    )
    _run_codex_step(smoke_cmd, env, workdir, "phase3_prompt.txt", phase3_prompt)
    phase3_report = _read_report(workdir / PHASE3_REPORT)
    assert phase3_report["loader"] == "DualFormatLoader"
    assert phase3_report["operator"] == "AddOperator"
    assert phase3_report["sum_numpy"] == EXPECTED_FIRST_FEATURE_SUM
    assert phase3_report["sum_list"] == EXPECTED_FIRST_FEATURE_SUM

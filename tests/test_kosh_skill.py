from pathlib import Path

import kosh


def test_kosh_skill_is_packaged():
    skill_dir = Path(kosh.__file__).resolve().parent / "skills" / "kosh"
    assert (skill_dir / "SKILL.md").is_file()
    assert (skill_dir / "agents" / "openai.yaml").is_file()


def test_codex_skill_doc_exists():
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "docs" / "codex_skill.md").is_file()


def test_kosh_skill_mentions_ensemble_tags():
    skill_dir = Path(kosh.__file__).resolve().parent / "skills" / "kosh"
    skill_text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    assert "ensemble_tags" in skill_text
    assert "Keep a given attribute in one place" in skill_text


def test_kosh_skill_mentions_query_paths():
    skill_dir = Path(kosh.__file__).resolve().parent / "skills" / "kosh"
    skill_text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    assert "Query later with" in skill_text
    assert "store.find(...)" in skill_text
    assert "store.find_ensembles(...)" in skill_text
    assert "dataset.find(...)" in skill_text
    assert "ensemble.find_datasets(...)" in skill_text


def test_kosh_skill_mentions_workflow_tracking():
    skill_dir = Path(kosh.__file__).resolve().parent / "skills" / "kosh"
    skill_text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    assert "Workflow Tracking" in skill_text
    assert "kosh_workflow" in skill_text
    assert "StepRequest" in skill_text
    assert "apply_step" in skill_text

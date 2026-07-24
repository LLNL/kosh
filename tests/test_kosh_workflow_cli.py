from __future__ import print_function

import os
import sys
import time
from subprocess import run

from koshbase import KoshTest


def _run_workflow(args, env=None):
    cmd = [sys.executable, "-m", "kosh.workflow", *args]
    proc = run(cmd, capture_output=True, text=True, env=env)
    return proc.returncode, proc.stdout, proc.stderr


class KoshTestWorkflowCli(KoshTest):
    def setUp(self):
        super(KoshTestWorkflowCli, self).setUp()
        self._env = dict(os.environ)
        # Some environments inject a sitecustomize that expects PATH to exist.
        self._env.setdefault("PATH", "/usr/bin:/bin")

    def test_check_no_match_exit_1_and_no_output(self):
        store, uri = self.connect()
        store.close()

        ensemble = "ens"
        rc, out, err = _run_workflow(
            ["--store", uri, "--ensemble", ensemble, "--step", "init", "--param", "p0=1.0"],
            env=self._env,
        )
        self.assertEqual(rc, 0, msg=err)

        rc, out, err = _run_workflow(
            ["--store", uri, "--ensemble", ensemble, "--check", "--param", "p0=2.0"],
            env=self._env,
        )
        self.assertEqual(rc, 1, msg=err)
        self.assertEqual(out.strip(), "")

        os.remove(uri)

    def test_check_prints_step_and_timestamp(self):
        store, uri = self.connect()
        store.close()

        ensemble = "ens"
        rc, out, err = _run_workflow(
            ["--store", uri, "--ensemble", ensemble, "--step", "init", "--param", "p0=1.0"],
            env=self._env,
        )
        self.assertEqual(rc, 0, msg=err)

        rc, out, err = _run_workflow(
            ["--store", uri, "--ensemble", ensemble, "--check", "--param", "p0=1.0"],
            env=self._env,
        )
        self.assertEqual(rc, 0, msg=err)
        line = out.strip().splitlines()[0]
        parts = line.split("\t")
        self.assertEqual(len(parts), 4, msg=line)
        ds_id, step, ts_str, ts_epoch = parts
        self.assertTrue(ds_id)
        self.assertEqual(step, "init")
        self.assertTrue(ts_str)
        float(ts_epoch)

        os.remove(uri)

    def test_check_time_format_changes_string_column(self):
        store, uri = self.connect()
        store.close()

        ensemble = "ens"
        rc, out, err = _run_workflow(
            ["--store", uri, "--ensemble", ensemble, "--step", "init", "--param", "p0=1.0"],
            env=self._env,
        )
        self.assertEqual(rc, 0, msg=err)

        rc, out, err = _run_workflow(
            [
                "--store",
                uri,
                "--ensemble",
                ensemble,
                "--check",
                "--check-time-format",
                "%Y-%m-%d %H:%M:%S",
                "--param",
                "p0=1.0",
            ],
            env=self._env,
        )
        self.assertEqual(rc, 0, msg=err)
        ts_str = out.strip().splitlines()[0].split("\t")[2]
        self.assertIn(" ", ts_str)
        self.assertNotIn("T", ts_str)
        self.assertGreaterEqual(len(ts_str), 19)

        os.remove(uri)

    def test_step_update_changes_step(self):
        store, uri = self.connect()
        store.close()

        ensemble = "ens"
        rc, out, err = _run_workflow(
            ["--store", uri, "--ensemble", ensemble, "--step", "init", "--param", "p0=1.0"],
            env=self._env,
        )
        self.assertEqual(rc, 0, msg=err)

        time.sleep(0.05)
        rc, out, err = _run_workflow(
            ["--store", uri, "--ensemble", ensemble, "--step", "running", "--param", "p0=1.0"],
            env=self._env,
        )
        self.assertEqual(rc, 0, msg=err)

        rc, out, err = _run_workflow(
            ["--store", uri, "--ensemble", ensemble, "--check", "--param", "p0=1.0"],
            env=self._env,
        )
        self.assertEqual(rc, 0, msg=err)
        step = out.strip().splitlines()[0].split("\t")[1]
        self.assertEqual(step, "running")

        os.remove(uri)

    def test_strict_match_ignores_tolerance(self):
        store, uri = self.connect()
        store.close()

        ensemble = "ens"
        rc, out, err = _run_workflow(
            ["--store", uri, "--ensemble", ensemble, "--step", "init", "--param", "p0=1.0"],
            env=self._env,
        )
        self.assertEqual(rc, 0, msg=err)

        # Within default tolerances => match without strict matching.
        rc, out, err = _run_workflow(
            ["--store", uri, "--ensemble", ensemble, "--check", "--param", "p0=1.0000000005"],
            env=self._env,
        )
        self.assertEqual(rc, 0, msg=err)

        # Exact matching => no match.
        rc, out, err = _run_workflow(
            ["--store", uri, "--ensemble", ensemble, "--check", "--strict-match", "--param", "p0=1.0000000005"],
            env=self._env,
        )
        self.assertEqual(rc, 1, msg=err)

        os.remove(uri)

    def test_strict_match_mutually_exclusive_with_rtol_atol(self):
        store, uri = self.connect()
        store.close()

        ensemble = "ens"
        rc, out, err = _run_workflow(
            ["--store", uri, "--ensemble", ensemble, "--step", "init", "--param", "p0=1.0"],
            env=self._env,
        )
        self.assertEqual(rc, 0, msg=err)

        rc, out, err = _run_workflow(
            [
                "--store",
                uri,
                "--ensemble",
                ensemble,
                "--check",
                "--strict-match",
                "--rtol",
                "1e-3",
                "--param",
                "p0=1.0",
            ],
            env=self._env,
        )
        self.assertEqual(rc, 2)
        self.assertIn("mutually exclusive", err)

        os.remove(uri)

    def test_check_without_ensemble_searches_store(self):
        store, uri = self.connect()
        store.close()

        rc, out, err = _run_workflow(
            ["--store", uri, "--ensemble", "ens", "--step", "init", "--param", "p0=1.0"],
            env=self._env,
        )
        self.assertEqual(rc, 0, msg=err)

        # Omitting --ensemble should search the whole store and still match.
        rc, out, err = _run_workflow(
            ["--store", uri, "--check", "--param", "p0=1.0"],
            env=self._env,
        )
        self.assertEqual(rc, 0, msg=err)
        self.assertTrue(out.strip())

        os.remove(uri)

    def test_check_without_dataset_record_type_searches_store(self):
        store, uri = self.connect(dataset_record_type="some_new_type")
        store.create(metadata={"p0": 1.0, "workflow_step": "init"})
        store.close()

        rc, out, err = _run_workflow(
            ["--store", uri, "--check", "--param", "p0=1.0"],
            env=self._env,
        )
        self.assertEqual(rc, 0, msg=err)
        self.assertTrue(out.strip())

        os.remove(uri)

    def test_implicit_param_flag_syntax(self):
        store, uri = self.connect()
        store.close()

        ensemble = "ens"
        rc, out, err = _run_workflow(
            ["--store", uri, "--ensemble", ensemble, "--step", "init", "--p0=1.0"],
            env=self._env,
        )
        self.assertEqual(rc, 0, msg=err)

        rc, out, err = _run_workflow(
            ["--store", uri, "--ensemble", ensemble, "--check", "--p0=1.0"],
            env=self._env,
        )
        self.assertEqual(rc, 0, msg=err)
        self.assertTrue(out.strip())

        os.remove(uri)

    def test_mixed_implicit_and_explicit_param_syntax(self):
        store, uri = self.connect()
        store.close()

        ensemble = "ens"
        rc, out, err = _run_workflow(
            [
                "--store",
                uri,
                "--ensemble",
                ensemble,
                "--step",
                "init",
                "--param",
                "p 1=2.0",
                "--p0=1.0",
            ],
            env=self._env,
        )
        self.assertEqual(rc, 0, msg=err)

        rc, out, err = _run_workflow(
            [
                "--store",
                uri,
                "--ensemble",
                ensemble,
                "--check",
                "--param",
                "p 1=2.0",
                "--p0=1.0",
            ],
            env=self._env,
        )
        self.assertEqual(rc, 0, msg=err)
        self.assertTrue(out.strip())

        os.remove(uri)

    def test_typos_suggest_and_can_be_disabled(self):
        store, uri = self.connect()
        store.close()

        rc, out, err = _run_workflow(
            ["--store", uri, "--ensembl", "ens", "--check", "--param", "p0=1.0"],
            env=self._env,
        )
        self.assertEqual(rc, 2)
        self.assertIn("Did you mean --ensemble?", err)
        self.assertIn("--no-typo-check", err)

        rc, out, err = _run_workflow(
            ["--store", uri, "--check", "--rtlo", "1e-3", "--param", "p0=1.0"],
            env=self._env,
        )
        self.assertEqual(rc, 2)
        self.assertIn("Did you mean --rtol?", err)

        rc, out, err = _run_workflow(
            [
                "--store",
                uri,
                "--no-typo-check",
                "--step",
                "init",
                "--param",
                "p0=1.0",
                "--param",
                "ensembl=ens",
            ],
            env=self._env,
        )
        self.assertEqual(rc, 0, msg=err)

        rc, out, err = _run_workflow(
            ["--store", uri, "--no-typo-check", "--check", "--ensembl", "ens", "--param", "p0=1.0"],
            env=self._env,
        )
        self.assertEqual(rc, 0, msg=err)
        self.assertTrue(out.strip())

        os.remove(uri)

    def test_step_update_without_ensemble_searches_store(self):
        store, uri = self.connect()
        store.close()

        ensemble = "ens"
        rc, out, err = _run_workflow(
            ["--store", uri, "--ensemble", ensemble, "--step", "init", "--param", "p0=1.0"],
            env=self._env,
        )
        self.assertEqual(rc, 0, msg=err)

        time.sleep(0.05)
        # Omitting --ensemble should update the matching dataset found in the store.
        rc, out, err = _run_workflow(
            ["--store", uri, "--step", "running", "--param", "p0=1.0"],
            env=self._env,
        )
        self.assertEqual(rc, 0, msg=err)

        rc, out, err = _run_workflow(
            ["--store", uri, "--ensemble", ensemble, "--check", "--param", "p0=1.0"],
            env=self._env,
        )
        self.assertEqual(rc, 0, msg=err)
        step = out.strip().splitlines()[0].split("\t")[1]
        self.assertEqual(step, "running")

        os.remove(uri)

    def test_size_flag_prints_ensemble_count(self):
        store, uri = self.connect()
        store.close()

        ensemble = "ens"
        rc, out, err = _run_workflow(
            ["--store", uri, "--ensemble", ensemble, "--step", "init", "--param", "p0=1.0"],
            env=self._env,
        )
        self.assertEqual(rc, 0, msg=err)

        rc, out, err = _run_workflow(
            ["--store", uri, "-d", "dataset", "--ensemble", ensemble, "--size"],
            env=self._env,
        )
        self.assertEqual(rc, 0, msg=err)
        self.assertEqual(out.strip(), "1")

        os.remove(uri)

from __future__ import print_function
import os
from koshbase import KoshTest
import shutil
import shlex
from subprocess import Popen, PIPE
import numpy
import kosh
import six
import random


def run_cmd(cmd, verbose=False):
    if verbose:
        print(cmd)
    cmd = shlex.split(cmd)
    p = Popen(cmd, stdout=PIPE, stderr=PIPE)
    o, e = p.communicate()
    if p.returncode != 0:
        print("OOOOPSY:", o.decode())
        print("OOOOPSY:", e.decode())
    if verbose:
        print("OUT:", o)
        print("ERR:", e)
    assert(p.returncode == 0)
    return o.decode(
        "utf-8").strip().split("\n"), e.decode("utf-8").strip().split("\n")


class KoshTestDataset(KoshTest):

    def _tar(self, tar_command):
        store, kosh_db = self.connect(dataset_record_type="dataset")
        store2, kosh_db_2 = self.connect(dataset_record_type="dataset")

        d1 = store.create("one")
        d1.associate("setup.py", "python")
        seed = random.randint(0, 100000)
        o, e = run_cmd(
            "kosh {} --store={} -c -v -f my_setup_{}.tar setup.py".format(tar_command, kosh_db, seed), verbose=True)
        if tar_command != "htar":
            self.assertTrue(os.path.exists("my_setup_{}.tar".format(seed)))

        self.assertEqual(len(list(store2.search())), 0)
        o, e = run_cmd(
            "kosh {} --store={} -x -v -f my_setup_{}.tar".format(tar_command, kosh_db_2, seed), verbose=True)

        self.assertEqual(len(list(store2.search())), 1)
        os.remove(kosh_db)
        os.remove(kosh_db_2)
        if tar_command != "htar":
            os.remove("my_setup_{}.tar".format(seed))

    def test_htar(self):
        if os.environ.get("SOURCE_ZONE", "CZ") != "CZ":
            self._tar("htar")

    def test_tar(self):
        self._tar("tar")

    def test_create_dataset(self):
        store, kosh_db = self.connect(dataset_record_type="dataset")
        # Empty store
        datasets = list(store.search())
        self.assertEqual(len(datasets), 0)
        o, e = run_cmd(
            "kosh create --store={} paramint=2 paramfloat 2.4 paramstr \"'45'\"".format(kosh_db), verbose=True)

        datasets = list(store.search())
        # Created a new dataset
        self.assertEqual(len(datasets), 1)
        ds = datasets[0]
        self.assertEqual(ds.list_attributes(), ["creator", "id", "name", "paramfloat", "paramint", "paramstr"])

        self.assertEqual(ds.paramint, 2)
        self.assertIsInstance(ds.paramint, int)

        self.assertEqual(ds.paramfloat, 2.4)
        self.assertIsInstance(ds.paramfloat, float)

        self.assertEqual(ds.paramstr, "45")
        self.assertIsInstance(ds.paramstr, six.text_type)
        os.remove(kosh_db)

    def test_create_store(self):
        name = "kosh_command_open_new.sql"
        if os.path.exists(name):
            os.remove(name)

        o, e = run_cmd("kosh create_new_db -u {}".format(name))

        self.assertTrue(os.path.exists(name))

        store = kosh.KoshStore(name)

        self.assertEqual(len(list(store.search())), 0)
        store.create()
        self.assertEqual(len(list(store.search())), 1)
        o, e = run_cmd("kosh create_new_db -u {}".format(name))
        store = kosh.KoshStore(name)
        self.assertEqual(len(list(store.search())), 0)
        os.remove(name)

    def test_kosh_command(self):
        shutil.copy("tests/baselines/sina/data.sqlite", "cmd_line.sql")
        store, kosh_db = self.connect(
            db_uri="cmd_line.sql", dataset_record_type="obs")
        # Search the all store
        o, e = run_cmd("kosh search -s 'cmd_line.sql' -d obs")
        self.assertEqual(len(o), 27)
        o, e = run_cmd("kosh search -s 'cmd_line.sql' -d obs PARAM1=143.557")
        self.assertEqual(len(o), 1)
        o, e = run_cmd("kosh search -s 'cmd_line.sql' -d obs PARAM1>241.289")
        self.assertEqual(len(o), 3)
        o, e = run_cmd("kosh search -s 'cmd_line.sql' -d obs PARAM1>=241.289")
        self.assertEqual(len(o), 4)
        o, e = run_cmd("kosh search -s 'cmd_line.sql' -d obs PARAM1<241.289")
        self.assertEqual(len(o), 6)
        o, e = run_cmd("kosh search -s 'cmd_line.sql' -d obs PARAM1<=241.289")
        self.assertEqual(len(o), 7)
        o, e = run_cmd(
            "kosh search -s 'cmd_line.sql' -d obs PARAM1=DataRange(140,445)")
        self.assertEqual(len(o), 8)
        o, e = run_cmd(
            "kosh add -s 'cmd_line.sql' -d obs -i '2020-03-11-13-45-23' PARAM1=156 PARAM2=.2 PARAM3=something")
        ds = store.open('2020-03-11-13-45-23')
        self.assertEqual(ds.PARAM1, 156)
        self.assertEqual(ds.PARAM2, .2)
        self.assertEqual(ds.PARAM3, "something")
        o, e = run_cmd(
            "kosh remove -s 'cmd_line.sql' -d obs -i '2020-03-11-13-45-23' -f")
        with self.assertRaises(Exception):
            ds = store.open('2020-03-11-13-45-23')
        o, e = run_cmd(
            "kosh associate -s 'cmd_line.sql' -d obs -i '2019-04-05-02-11-29'"
            " -u tests/baselines/node_extracts2/node_extracts2.hdf5 -m hdf5")
        ds = store.open('2019-04-05-02-11-29')
        self.assertEqual(len(ds._associated_data_), 1)
        o, e = run_cmd(
            "kosh dissociate -s 'cmd_line.sql' -d obs -i '2019-04-05-02-11-29'"
            " -u tests/baselines/node_extracts2/node_extracts2.hdf5")
        self.assertEqual(len(ds._associated_data_), 0)
        o, e = run_cmd(
            "kosh associate -s 'cmd_line.sql' -d obs -i '2019-04-05-02-11-29'"
            " -u tests/baselines/node_extracts2/node_extracts2.hdf5 -m hdf5")
        o, e = run_cmd(
            "kosh features -s 'cmd_line.sql' -d obs -i '2019-04-05-02-11-29'")
        self.assertTrue("node/metrics_4" in o[1])
        npyfile = "test_kosh_cmd.npy"
        if os.path.exists(npyfile):
            os.remove(npyfile)
        o, e = run_cmd(
            "kosh extract -s 'cmd_line.sql' -d obs -i '2019-04-05-02-11-29'"
            " -f node/metrics_4 zone/metrics_4 --dump {npyfile}".format(npyfile=npyfile))
        self.assertTrue(os.path.exists(npyfile))
        n4, z4 = numpy.load(npyfile)
        self.assertEqual(n4.shape, (18,))
        self.assertEqual(z4.shape, (18,))
        os.remove(kosh_db)
        os.remove(npyfile)

    def test_dissociate_dead_files(self):
        store, kosh_db = self.connect()
        ds = store.create()
        ds.associate("setup.py", "py")  # real one
        ds.associate("blablabla.py", "py")  # dead one
        ds.associate("blablbla.hdf5", "hdf5")  # dead one
        # real one
        ds.associate(
            "tests/baselines/node_extracts2/node_extracts2.hdf5", "hdf5")
        ds.associate("README.md", "md")  # real
        ds.associate("REEEEDME.mmmmdddd", "md")  # fake one
        verbose = False
        self.assertEqual(len(list(ds.search())), 6)
        # first test cleanup python files only
        self.assertEqual(len(list(ds.search(mime_type="py"))), 2)
        # Dry run first
        cmd = "kosh cleanup_files -s '{}' -d blah --dry-run mime_type=py".format(
            kosh_db)
        o, e = run_cmd(cmd, verbose=verbose)
        # Let's make sure it's still all here
        self.assertEqual(len(list(ds.search())), 6)
        self.assertEqual(len(list(ds.search(mime_type="py"))), 2)
        cmd = "kosh cleanup_files -s '{}' -d blah mime_type=py".format(kosh_db)
        o, e = run_cmd(cmd, verbose=verbose)
        # Let's make sure only one py file was removed
        self.assertEqual(len(list(ds.search())), 5)
        self.assertEqual(len(list(ds.search(mime_type="py"))), 1)
        # Let's clean it all
        cmd = "kosh cleanup_files -s '{}' -d blah ".format(kosh_db)
        o, e = run_cmd(cmd, verbose=verbose)
        self.assertEqual(len(list(ds.search())), 3)
        self.assertEqual(len(list(ds.search(mime_type="py"))), 1)
        self.assertEqual(len(list(ds.search(mime_type="md"))), 1)
        self.assertEqual(len(list(ds.search(mime_type="hdf5"))), 1)
        os.remove(kosh_db)


if __name__ == "__main__":
    A = KoshTestDataset()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

from __future__ import print_function
from koshbase import KoshTest
import os
from subprocess import Popen, PIPE
import shlex


def create_file(filename):
    with open(filename, "w") as f:
        print("whatever", file=f)


def run_rm(sources, store_sources, verbose=False):
    cmd = "python scripts/kosh_command.py rm --dataset_record_type=blah "
    for store in store_sources:
        cmd += " --store {}".format(store)
    cmd += " --sources {} ".format(" ".join(sources))

    if verbose:
        print("TESTING:", cmd)
    p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
    o, e = p.communicate()
    print(o.decode())
    print(e.decode())


class KoshTestRm(KoshTest):
    def test_rm_single_file(self):
        store, db_uri = self.connect()
        ds = store.create()
        filename = os.path.abspath("a_test_file_for_rm.txt")
        create_file(filename)
        ds.associate(filename, "text")
        ds.associate("fake_one.text", "text")
        associated = list(ds.find(mime_type="text"))
        self.assertEqual(len(associated), 2)
        run_rm([filename, ], [db_uri, ])

        associated = list(ds.find(mime_type="text"))
        self.assertEqual(len(associated), 1)
        self.assertEqual(associated[0].uri, "fake_one.text")
        self.assertFalse(os.path.exists(filename))

        os.remove(db_uri)

    def test_rm_multi_files(self):
        store, db_uri = self.connect()
        ds = store.create()
        filenames = []
        for f in ["rm_a", "rm_b"]:
            filenames.append(os.path.abspath(f))
            create_file(f)
        ds.associate(filenames, "text")
        ds.associate("fake_one.text", "text")
        run_rm(filenames, [db_uri, ])

        associated = list(ds.find(mime_type="text"))
        self.assertEqual(len(associated), 1)
        self.assertEqual(associated[0].uri, "fake_one.text")
        for filename in filenames:
            self.assertFalse(os.path.exists(filename))

        os.remove(db_uri)

    def test_rm_dir(self):
        store, db_uri = self.connect()
        ds = store.create()
        filenames = []
        try:
            os.removedirs("rm_from_dir")
        except BaseException:
            pass
        try:
            os.makedirs("rm_from_dir")
        except BaseException:
            pass
        for f in ["rm_a", "rm_b"]:
            filename = os.path.join("rm_from_dir", f)
            filenames.append(os.path.abspath(filename))
            create_file(filename)
        ds.associate(filenames, "text")
        ds.associate("fake_one.text", "text")
        run_rm(["rm_from_dir", ], [db_uri, ])

        associated = list(ds.find(mime_type="text"))
        self.assertEqual(len(associated), 1)
        self.assertEqual(associated[0].uri, "fake_one.text")
        for filename in filenames:
            self.assertFalse(os.path.exists(filename))

        os.remove(db_uri)

    def test_rm_mix(self):
        store, db_uri = self.connect()
        ds = store.create()
        filenames = []
        try:
            os.removedirs("rm_from_dir_mix")
        except BaseException:
            pass
        try:
            os.makedirs("rm_from_dir_mix")
        except BaseException:
            pass
        for f in ["rm_a", "rm_b"]:
            filename = os.path.join("rm_from_dir_mix", f)
            filenames.append(os.path.abspath(filename))
            create_file(filename)
        filenames.append(os.path.abspath("rm_mixed_file.text"))
        create_file(filenames[-1])
        ds.associate(filenames, "text")
        ds.associate("fake_one.text", "text")
        run_rm(["rm_from_dir_mix", filenames[-1]], [db_uri, ])

        associated = list(ds.find(mime_type="text"))
        self.assertEqual(len(associated), 1)
        self.assertEqual(associated[0].uri, "fake_one.text")
        for filename in filenames:
            self.assertFalse(os.path.exists(filename))

        os.remove(db_uri)


if __name__ == "__main__":
    A = KoshTestRm()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

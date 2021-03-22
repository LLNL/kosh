from __future__ import print_function
from koshbase import KoshTest
import os
import kosh
import random
from subprocess import Popen, PIPE
import shlex
import sys
import shutil


def create_file(filename):
    with open(filename, "w") as f:
        print("whatever", file=f)
        print(random.randint(0, 1000000), file=f)


def move_file(old, new):
    os.rename(old, new)
    return os.path.abspath(new), os.path.abspath(old)


def run_reassociate(store_sources, new_uris, original_uris=[]):
    cmd = "{}/bin/python scripts/kosh_command.py reassociate --dataset_record_type=blah ".format(
        sys.prefix)
    for store in store_sources:
        cmd += " --store {}".format(store)
    cmd += " --new_uris {}".format(" ".join(new_uris))
    if original_uris != []:
        cmd += " --original_uris {}".format(" ".join(original_uris))

    print("TESTING :", cmd)
    p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
    o, e = p.communicate()
    out = o, e
    return p, out


class KoshTestReassociate(KoshTest):
    def test_dataset_reassociate(self):
        store, db_uri = self.connect()
        ds = store.create()

        filename = "reassociate.py"
        create_file(filename)

        ds.associate("reassociate.py", "py", long_sha=True)

        # Ok let's move this file w//o kosh
        filename, old_name = move_file(filename, "new_name.py")
        # Use old uri
        ds.reassociate(filename, old_name)
        self.assertEqual(len(ds.search(uri=filename)), 1)

        # Ok let's move this file w//o kosh
        filename, old_name = move_file(filename, "another_new_name.py")
        # Use short sha
        ds.reassociate(filename, kosh.utils.compute_fast_sha(filename))
        self.assertEqual(len(ds.search(uri=filename)), 1)

        # Ok let's move this file w//o kosh
        filename, old_name = move_file(filename, "a_new_name.py")
        # Use long sha
        ds.reassociate(filename, kosh.utils.compute_long_sha(filename))
        self.assertEqual(len(ds.search(uri=filename)), 1)

        # Ok let's move this file w//o kosh
        filename, old_name = move_file(filename, "final_name.py")
        # Use no sources
        ds.reassociate(filename)
        self.assertEqual(len(ds.search(uri=filename)), 1)

        os.remove(filename)
        os.remove(db_uri)

    def test_store_reassociate(self):
        store, db_uri = self.connect()
        ds = store.create()

        filename = "store_reassociate.py"
        create_file(filename)

        ds.associate("store_reassociate.py", "py", long_sha=True)

        # Ok let's move this file w//o kosh
        filename, old_name = move_file(filename, "store_new_name.py")
        # Use old uri
        store.reassociate(filename, old_name)
        self.assertEqual(len(ds.search(uri=filename)), 1)

        # Ok let's move this file w//o kosh
        filename, old_name = move_file(filename, "store_another_new_name.py")
        # Use short sha
        store.reassociate(filename, kosh.utils.compute_fast_sha(filename))
        self.assertEqual(len(ds.search(uri=filename)), 1)

        # Ok let's move this file w//o kosh
        filename, old_name = move_file(filename, "store_a_new_name.py")
        # Use long sha
        store.reassociate(filename, kosh.utils.compute_long_sha(filename))
        self.assertEqual(len(ds.search(uri=filename)), 1)

        # Ok let's move this file w//o kosh
        filename, old_name = move_file(filename, "store_final_name.py")
        # Use no sources
        store.reassociate(filename)
        self.assertEqual(len(ds.search(uri=filename)), 1)

        os.remove(filename)
        os.remove(db_uri)

    def test_command_line_one_file(self):
        store, db_uri = self.connect()
        ds = store.create()

        rand = str(random.randint(0, 100000000))
        filename = rand + "_reassociate.py"
        create_file(filename)

        ds.associate(filename, "py", long_sha=True)

        # Ok let's move this file w//o kosh
        filename, old_name = move_file(filename, rand + "_new_name.py")
        # Use old uri
        run_reassociate([db_uri, ], [filename, ], [old_name, ])
        self.assertEqual(len(ds.search(uri=filename)), 1)

        # Ok let's move this file w//o kosh
        filename, old_name = move_file(filename, rand + "_another_new_name.py")
        # Use short sha
        run_reassociate([db_uri, ], [filename, ], [
                        kosh.utils.compute_fast_sha(filename), ])
        self.assertEqual(len(ds.search(uri=filename)), 1)

        # Ok let's move this file w//o kosh
        filename, old_name = move_file(filename, rand + "_a_new_name.py")
        # Use long sha
        run_reassociate([db_uri, ], [filename, ], [
                        kosh.utils.compute_long_sha(filename), ])
        self.assertEqual(len(ds.search(uri=filename)), 1)

        # Ok let's move this file w//o kosh
        filename, old_name = move_file(filename, rand + "_final_name.py")
        # Use no sources
        run_reassociate([db_uri, ], [filename, ])
        self.assertEqual(len(ds.search(uri=filename)), 1)

        os.remove(filename)
        os.remove(db_uri)

    def test_command_line_multi_files(self):
        store, db_uri = self.connect()
        ds = store.create()

        rand = str(random.randint(0, 100000000))
        filename1 = rand + "_reassociate_1.py"
        filenames = [filename1, ]
        create_file(filename1)
        ds.associate(filename1, "py", long_sha=True)
        filename2 = rand + "_reassociate_2.py"
        create_file(filename2)
        filenames.append(filename2)
        ds.associate(filenames, "py", long_sha=True)

        # Ok let's move this file w//o kosh
        filename1, old_name1 = move_file(filename1, rand + "_new_name1.py")
        filename2, old_name2 = move_file(filename2, rand + "_new_name2.py")
        print("FILENAME! NOW:", filename1)
        # Use old uri
        run_reassociate([db_uri, ], [filename1, filename2],
                        [old_name1, old_name2])
        self.assertEqual(len(ds.search(uri=filename1)), 1)
        self.assertEqual(len(ds.search(uri=filename2)), 1)

        # Ok let's move this file w//o kosh
        filename1, old_name1 = move_file(filename1, rand + "_a_new_name1.py")
        filename2, old_name2 = move_file(filename2, rand + "_a_new_name2.py")
        # Use no source
        run_reassociate([db_uri, ], [filename1, filename2])
        self.assertEqual(len(ds.search(uri=filename1)), 1)
        self.assertEqual(len(ds.search(uri=filename2)), 1)

        # Ok let's move this file w//o kosh
        filename1, old_name1 = move_file(filename1, rand + "_some_name1.py")
        filename2, old_name2 = move_file(filename2, rand + "_some_name2.py")
        # Use pattern
        run_reassociate([db_uri, ], [rand + "_some_name*.py", ])
        self.assertEqual(len(ds.search(uri=filename1)), 1)
        self.assertEqual(len(ds.search(uri=filename2)), 1)

        try:
            shutil.rmtree(rand)
        except BaseException:
            pass
        os.makedirs(rand)
        filename1, old_name1 = move_file(
            filename1, rand + "/" + rand + "_a_new_name1.py")
        filename2, old_name2 = move_file(
            filename2, rand + "/" + rand + "_a_new_name2.py")
        # Use pattern
        run_reassociate([db_uri, ], [rand, ])
        self.assertEqual(len(ds.search(uri=filename1)), 1)
        self.assertEqual(len(ds.search(uri=filename2)), 1)

        filename1, old_name1 = move_file(
            filename1, rand + "_another_new_name1.py")
        filename2, old_name2 = move_file(
            filename2, rand + "_another_new_name2.py")
        # Use wrong args (check failed)
        p, out = run_reassociate(
            [db_uri, ], [filename1, filename2], [old_name1, ])
        self.assertNotEqual(p.returncode, 0)

        os.remove(filename1)
        os.remove(filename2)
        os.remove(db_uri)


if __name__ == "__main__":
    A = KoshTestReassociate()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

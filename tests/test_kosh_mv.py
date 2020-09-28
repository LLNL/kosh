from __future__ import print_function
from koshbase import KoshTest
from subprocess import Popen, PIPE
import shlex
import os
import random
import shutil


def create_file(filename):
    with open(filename, "w") as f:
        print("whatever", file=f)


def run_mv(sources, dest, store_sources, store_destinations=None):
    cmd = "python scripts/kosh_command.py mv --dataset_record_type=blah "
    for store in store_sources:
        cmd += " --store {}".format(store)
    if store_destinations is not None:
        for store in store_destinations:
            cmd += " --destination-store {}".format(store)
    cmd += " --sources {} --destination {}".format(" ".join(sources), dest)

    p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
    o, e = p.communicate()
    out = o, e
    print("CMD:", cmd)
    return p, out


class KoshTestMv(KoshTest):
    def file_exist(self, name):
        if "@" not in name:
            # regular local file
            return os.path.exists(name)
        else:
            is_file_cmd = "if [ -f {} ]; then echo -e 1 ; else echo -e 0 ;  fi ;"
            filename = ":".join(name.split(":")[1:])  # split over :
            is_file_cmd = is_file_cmd.format(filename)
            cmd = "ssh {}@{} '{}'".format(self.user,
                                          self.hostname, is_file_cmd)
            is_file_proc = Popen(
                "/usr/bin/bash",
                stdin=PIPE,
                stdout=PIPE,
                stderr=PIPE)
            o, e = is_file_proc.communicate(cmd.encode())
            o = o.decode().split("\n")
            return int(o[0])

    def test_file_to_file(self):
        # kosh mv --stores store1.sql store2.sql --source file1 --destination
        # file2
        rand = str(random.randint(0, 1000000))
        store1, db1 = self.connect()
        store2, db2 = self.connect()

        file_src_orig = os.path.abspath(rand + "_file_to_file.py")
        create_file(file_src_orig)
        ds1 = store1.create(name="test")
        ds1.associate(file_src_orig, mime_type="py")
        ds2 = store2.create(name="test")
        ds2.associate(file_src_orig, mime_type="py")

        dest_name_orig = os.path.abspath("file_dest.py")
        run_mv([file_src_orig, ], dest_name_orig, [db1, db2])
        for ds in [ds1, ds2]:
            associated = ds.search(mime_type="py")[0]
            self.assertEqual(associated.uri, dest_name_orig)
        self.assertFalse(self.file_exist(file_src_orig))
        self.assertTrue(self.file_exist(dest_name_orig))

        # cleanup file
        os.remove(dest_name_orig)

        # cleanup stores
        for db in [db1, db2]:
            os.remove(db)

    def test_move_files_to_new_directory(self):
        # kosh mv --stores_store1.sql store2.sql --source dir1 --destination
        # dir2
        rand = str(random.randint(0, 1000000))
        store1, db1 = self.connect()
        store2, db2 = self.connect()

        file_src_orig = [rand + "_f2d/1.py", rand + "_f2d/sub/file2.py"]
        file_src_orig_associate = [os.path.abspath(x) for x in file_src_orig]

        try:
            os.removedirs(os.path.dirname(file_src_orig[0]))
        except BaseException:
            pass
        try:
            os.makedirs(os.path.dirname(file_src_orig[1]))
        except BaseException:
            pass
        for src in file_src_orig:
            create_file(src)
        ds1 = store1.create()
        ds1.associate(file_src_orig_associate, mime_type="py")

        ds2 = store2.create()
        ds2.associate(file_src_orig_associate, mime_type="py")

        dest_name_orig = rand + "_f2d_dest"
        try:
            os.removedirs(dest_name_orig)
        except BaseException:
            pass
        try:
            os.makedirs(dest_name_orig)
        except BaseException:
            pass
        run_mv(file_src_orig, dest_name_orig, [db1, db2])
        # First let's check files are moved
        new_paths = []
        for file_src in file_src_orig:
            self.assertFalse(os.path.exists(file_src))
            new_paths.append(
                os.path.abspath(
                    os.path.join(
                        dest_name_orig,
                        os.path.basename(file_src))))
            self.assertTrue(os.path.exists(new_paths[-1]))

        for ds in [ds1, ds2]:
            associated_uris = ds.search(mime_type="py")
            for associated in associated_uris:
                self.assertTrue(associated.uri in new_paths)

        # Cleanup files
        shutil.rmtree(rand + "_f2d")
        shutil.rmtree(rand + "_f2d_dest")

        # cleanup stores
        for db in [db1, db2]:
            os.remove(db)

    def test_move_directory(self):
        # kosh mv --stores_store1.sql store2.sql --source dir1 --destination
        # dir2
        rand = str(random.randint(0, 1000000))
        store1, db1 = self.connect()
        store2, db2 = self.connect()

        file_src_orig = [rand + "_d2d/1.py", rand + "_d2d/sub/file2.py"]
        file_src_orig_associate = [os.path.abspath(x) for x in file_src_orig]
        try:
            os.removedirs(os.path.dirname(file_src_orig[0]))
        except BaseException:
            pass
        try:
            os.makedirs(os.path.dirname(file_src_orig[1]))
        except BaseException:
            pass
        for src in file_src_orig:
            create_file(src)
        ds1 = store1.create()
        ds1.associate(file_src_orig_associate, mime_type="py")

        ds2 = store2.create()
        ds2.associate(file_src_orig_associate[1:-1], mime_type="py")

        dest_name_orig = rand + "_d2d_dest"
        try:
            os.removedirs(dest_name_orig)
        except BaseException:
            pass
        try:
            os.makedirs(dest_name_orig)
        except BaseException:
            pass
        orig_dir = os.path.dirname(os.path.abspath(file_src_orig[0]))
        run_mv([orig_dir, ], dest_name_orig, [db1, db2])
        # First let's check files are moved
        new_paths = []
        for file_src in file_src_orig:
            self.assertFalse(os.path.exists(file_src))
            new_paths.append(
                os.path.abspath(
                    os.path.join(
                        dest_name_orig,
                        file_src)))
            self.assertTrue(os.path.exists(new_paths[-1]))

        for ds in [ds1, ds2]:
            associated_uris = ds.search(mime_type="py")
            for associated in associated_uris:
                self.assertTrue(associated.uri in new_paths)

        # Cleanup files
        shutil.rmtree(rand + "_d2d")
        shutil.rmtree(rand + "_d2d_dest")

        # cleanup stores
        for db in [db1, db2]:
            os.remove(db)

    def test_move_files_pattern_to_new_directory_locally(self):
        # kosh mv --stores store1.sql store2.sql --source *.testme --source
        # dir1/testing_it_*.testme --destination dir2
        rand = str(random.randint(0, 1000000))
        store1, db1 = self.connect()
        store2, db2 = self.connect()
        file_src_orig = [
            rand + "_1.testme",
            rand + "_file2.testme",
            "dir1/testing_it_1.testme",
            "dir1/testing_it_2.testme",
            "dir1/i_dont_move.testme"]
        file_src_orig_associate = [os.path.abspath(x) for x in file_src_orig]
        try:
            os.makedirs(os.path.dirname(file_src_orig[2]))
        except BaseException:
            pass
        for src in file_src_orig:
            create_file(src)
        ds1 = store1.create()
        ds1.associate(file_src_orig_associate, mime_type="testme")
        ds2 = store2.create()
        ds2.associate(file_src_orig_associate[1:-1], mime_type="testme")

        dest_name_orig = rand + "_pattern_dest"

        try:
            os.removedirs(dest_name_orig)
        except BaseException:
            pass
        try:
            os.makedirs(dest_name_orig)
        except BaseException:
            pass
        run_mv(["*.testme", "dir1/testing_it*testme"],
               dest_name_orig, [db1, db2])

        for file_src in file_src_orig_associate[:-1]:
            # Test files are moved
            self.assertFalse(os.path.exists(file_src))
            dest = os.path.abspath(
                os.path.join(
                    dest_name_orig,
                    os.path.basename(file_src)))
            self.assertTrue(os.path.exists(dest))
            # Test datasets are updated
            for ds in [ds1, ds2]:
                associated_uris = ds.search(mime_type="testme")
                for associated in associated_uris:
                    if os.path.basename(
                            associated.uri) == os.path.basename(file_src):
                        self.assertEqual(associated.uri, dest)
        # Test that file that was not moved still is there
        self.assertTrue(os.path.exists(file_src_orig[-1]))
        self.assertGreater(len(ds1.search(uri=file_src_orig_associate[-1])), 0)

        # Cleanup files
        shutil.rmtree("dir1")
        shutil.rmtree(rand + "_pattern_dest")

        # cleanup stores
        for db in [db1, db2]:
            os.remove(db)

    def test_move_file_to_dir(self):
        rand = str(random.randint(0, 1000000))
        store1, db1 = self.connect()

        file_src_orig = "file_to_dir.py"
        file_src_orig_associate = os.path.abspath(file_src_orig)
        create_file(file_src_orig)
        ds1 = store1.create()
        ds1.associate(file_src_orig_associate, mime_type="py")
        dest_name_orig = rand + "_new_dir"
        try:
            os.removedirs(dest_name_orig)
        except BaseException:
            pass
        os.makedirs(dest_name_orig)
        run_mv([file_src_orig_associate, ], dest_name_orig, [db1, ])

        # Test file moved
        self.assertFalse(os.path.exists(file_src_orig_associate))
        dest_path = os.path.abspath(
            os.path.join(
                dest_name_orig,
                file_src_orig))
        self.assertTrue(os.path.exists(dest_path))

        associated = ds1.search(mime_type="py")[0]
        self.assertEqual(associated.uri, dest_path)

        # Cleanup
        shutil.rmtree(dest_name_orig)
        # cleanup stores
        os.remove(db1)

    def test_move_to_no_exist(self):
        rand = str(random.randint(0, 1000000))
        store1, db1 = self.connect()

        file_src_orig = ["file_to_dir_1.py", "file_to_dir_2.py"]
        file_src_orig_associate = [os.path.abspath(x) for x in file_src_orig]
        for x in file_src_orig:
            create_file(x)
        ds1 = store1.create()
        ds1.associate(file_src_orig_associate, mime_type="py")
        dest_name_orig = rand + "_new_dir"
        # Make sure dest dir does not exists
        try:
            os.removedirs(dest_name_orig)
        except BaseException:
            pass

        p, _ = run_mv(file_src_orig_associate, dest_name_orig, [db1, ])

        # Make sure it failed
        self.assertNotEqual(p.returncode, 0)
        # Test file not moved
        for x in file_src_orig:
            self.assertTrue(os.path.exists(x))
            dest_path = os.path.abspath(os.path.join(dest_name_orig, x))
            self.assertFalse(os.path.exists(dest_path))

        associated = ds1.search(mime_type="py")
        for a in associated:
            self.assertTrue(a.uri in file_src_orig_associate)

        # Cleanup
        for f in file_src_orig_associate:
            os.remove(f)
        # cleanup stores
        os.remove(db1)

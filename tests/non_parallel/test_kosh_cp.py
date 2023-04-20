from __future__ import print_function
import kosh
import shutil
import getpass
import socket
import random
import os
import shlex
from subprocess import Popen, PIPE
import sys
sys.path.insert(0, "tests")
from koshbase import KoshTest  # noqa
user = getpass.getuser()
hostname = socket.gethostname()


def create_file(filename):
    with open(filename, "w") as f:
        print("whatever", file=f)


def run_cp(sources, dest, store_sources, store_destinations=None):
    cmd = "python scripts/kosh_command.py cp --dataset_record_type=blah "
    for store in store_sources:
        cmd += " --store {}".format(store)
    if store_destinations is not None:
        for store in store_destinations:
            cmd += " --destination-store {}".format(store)
    cmd += " --sources {} --destination {}".format(" ".join(sources), dest)

    if not sys.platform.startswith("win"):
        cmd = shlex.split(cmd)
    p = Popen(cmd, stdout=PIPE, stderr=PIPE)
    o, e = p.communicate()
    out = o, e
    print("OUT:", o.decode())
    print("ERR:", e.decode())
    return out


class KoshTestCp(KoshTest):
    source_prefix = ""
    dest_prefix = ""

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
        # kosh cv --stores store1.sql --destination_stores store3.sql --source
        # file1 --destination file2
        if sys.platform.startswith("win"):
            print("Skipping test, we are on windows")
            return
        rand = str(random.randint(0, 1000000))
        store1, db1 = self.connect()
        store2, db2 = self.connect()

        file_src_orig = os.path.abspath(rand + "_file_to_file.py")
        create_file(file_src_orig)
        file_src = self.source_prefix + file_src_orig
        ds1 = store1.create(name="test")
        ds1.associate(file_src_orig, mime_type="py")
        store1.close()
        store2.close()

        dest_name_orig = os.path.abspath(rand + "_file_dest.py")
        dest_name = self.dest_prefix + dest_name_orig
        run_cp([file_src, ], dest_name, [self.source_prefix + db1, ],
               [self.dest_prefix + db2, ])
        # is it in dest store with correct url?
        # in case the store were remote we need to reopen them
        store1 = kosh.KoshStore(db_uri=db1, dataset_record_type="blah")
        store2 = kosh.KoshStore(db_uri=db2, dataset_record_type="blah")
        ds_store1 = list(store1.find(name="test"))
        self.assertEqual(len(ds_store1), 1)
        ds1 = next(store1.find(name="test"))
        ds_store2 = list(store2.find(name="test"))
        self.assertEqual(len(ds_store2), 1)
        associated = next(ds_store2[0].find(mime_type="py"))
        self.assertEqual(associated.uri, dest_name_orig)
        associated = next(ds1.find(mime_type="py"))
        self.assertEqual(associated.uri, file_src_orig)

        # cleanup file(s)
        os.remove(dest_name_orig)
        os.remove(file_src_orig)

        # cleanup stores
        for db in [db1, db2]:
            os.remove(db)

    def test_files_to_new_directory(self):
        # kosh mv --stores_store1.sql store2.sql--destination_stores store3.sql
        # --source dir1 --destination dir2
        if sys.platform.startswith("win"):
            print("Skipping test, we are on windows")
            return
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
        file_src = [self.source_prefix + f for f in file_src_orig_associate]
        ds1 = store1.create(name="test")
        ds1.associate(file_src_orig_associate, mime_type="py")

        dest_name_orig = rand + "_f2d_dest"
        try:
            os.removedirs(dest_name_orig)
        except BaseException:
            pass
        try:
            os.makedirs(dest_name_orig)
        except BaseException:
            pass
        dest_name = self.dest_prefix + os.path.abspath(dest_name_orig)
        run_cp(file_src, dest_name, [
               self.source_prefix + db1, ], [self.dest_prefix + db2, ])

        # First let's check files are moved
        new_paths = []
        for file_src in file_src_orig_associate:
            self.assertTrue(os.path.exists(file_src))
            new_paths.append(
                os.path.abspath(
                    os.path.join(
                        dest_name_orig,
                        os.path.basename(file_src))))
            self.assertTrue(os.path.exists(new_paths[-1]))

        store1 = kosh.KoshStore(db_uri=db1, dataset_record_type="blah")
        ds_store1 = list(store1.find(name="test"))
        self.assertEqual(len(ds_store1), 1)
        ds1 = ds_store1[0]
        associated_uris = ds1.find(mime_type="py")
        for associated in associated_uris:
            self.assertTrue(associated.uri in file_src_orig_associate)

        store2 = kosh.KoshStore(db_uri=db2, dataset_record_type="blah")
        ds_store2 = list(store2.find(name="test"))
        self.assertEqual(len(ds_store2), 1)
        ds2 = ds_store2[0]
        associated_uris = ds2.find(mime_type="py")
        for associated in associated_uris:
            self.assertTrue(associated.uri in new_paths)

        # Cleanup files
        shutil.rmtree(rand + "_f2d")
        shutil.rmtree(rand + "_f2d_dest")

        # cleanup stores
        for db in [db1, db2]:
            os.remove(db)

    def test_copy_directory_not_existing(self):
        # kosh cp --stores_store1.sql store2.sql --destination_stores
        # store3.sql --source dir1 --destination dir2
        if sys.platform.startswith("win"):
            print("Skipping test, we are on windows")
            return
        rand = str(random.randint(0, 1000000))
        store1, db1 = self.connect()
        store2, db2 = self.connect()

        file_src_orig = [rand + "_d2d/1.py", rand + "_d2d/sub/file2.py"]
        file_src_orig_associate = [os.path.abspath(x) for x in file_src_orig]
        orig_dir = os.path.dirname(file_src_orig_associate[0])
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
        ds1 = store1.create(name="test")
        ds1.associate(file_src_orig_associate, mime_type="py")
        print(ds1)

        dest_name_orig = os.path.abspath(rand + "_d2d_dest")
        try:
            os.removedirs(dest_name_orig)
        except BaseException:
            pass
        # try:
        #    os.makedirs(dest_name_orig)
        # except:
        #   pass
        dest_name = self.dest_prefix + dest_name_orig
        run_cp([self.source_prefix + orig_dir, ], dest_name,
               [self.source_prefix + db1, ], [self.dest_prefix + db2, ])
        # First let's check files are moved
        for file_src in file_src_orig_associate:
            self.assertTrue(os.path.exists(file_src))

        new_paths = []
        new_paths.append(os.path.join(dest_name_orig, rand + "_d2d", "1.py"))
        self.assertTrue(os.path.exists(new_paths[-1]))
        new_paths.append(
            os.path.join(
                dest_name_orig,
                rand + "_d2d",
                "sub",
                "file2.py"))
        self.assertTrue(os.path.exists(new_paths[-1]))

        store1 = kosh.KoshStore(db_uri=db1, dataset_record_type="blah")
        ds_store1 = list(store1.find(name="test"))
        self.assertEqual(len(ds_store1), 1)
        ds1 = ds_store1[0]
        associated_uris = ds1.find(mime_type="py")
        for associated in associated_uris:
            self.assertTrue(associated.uri in file_src_orig_associate)

        store2 = kosh.KoshStore(db_uri=db2, dataset_record_type="blah")
        ds_store2 = list(store2.find(name="test"))
        self.assertEqual(len(ds_store2), 1)
        ds2 = ds_store2[0]
        associated_uris = ds2.find(mime_type="py")
        for associated in associated_uris:
            self.assertTrue(associated.uri in new_paths)

        # Cleanup files
        shutil.rmtree(rand + "_d2d")
        shutil.rmtree(rand + "_d2d_dest")

        # cleanup stores
        for db in [db1, db2]:
            os.remove(db)

    def test_move_files_pattern_to_new_directory(self):
        # kosh mv --stores store1.sql store2.sql --source *.testme --source
        # dir1/testing_it_*.testme --destination dir2
        if sys.platform.startswith("win"):
            print("Skipping test, we are on windows")
            return
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
        file_src = [self.source_prefix + f for f in file_src_orig]
        ds1 = store1.create(name="test")
        ds1.associate(file_src_orig_associate, mime_type="testme")
        ds2 = store2.create(name="test")
        ds2.associate(file_src_orig_associate[1:-1], mime_type="testme")

        dest_name_orig = os.path.abspath(rand + "_pattern_dest")

        try:
            os.removedirs(dest_name_orig)
        except BaseException:
            pass
        try:
            os.makedirs(dest_name_orig)
        except BaseException:
            pass
        dest_name = self.dest_prefix + dest_name_orig
        apath = os.path.dirname(file_src_orig_associate[0])
        run_cp([self.source_prefix + os.path.join(apath,
                                                  "*.testme"),
                self.source_prefix + os.path.join(apath,
                                                  "dir1/testing_it*testme")],
               dest_name,
               [self.source_prefix + db1,
                ],
               [self.dest_prefix + db2,
                ])

        new_paths = []
        for file_src in file_src_orig_associate[:-1]:
            # Test files are moved
            self.assertTrue(os.path.exists(file_src))
            dest = os.path.abspath(
                os.path.join(
                    dest_name_orig,
                    os.path.basename(file_src)))
            new_paths.append(dest)
            self.assertTrue(os.path.exists(dest))
            # Test datasets are updated

        store1 = kosh.KoshStore(db_uri=db1, dataset_record_type="blah")
        ds_store1 = list(store1.find(name="test"))
        self.assertEqual(len(ds_store1), 1)
        ds1 = ds_store1[0]
        associated_uris = ds1.find(mime_type="testme")
        for associated in associated_uris:
            self.assertTrue(associated.uri in file_src_orig_associate)

        store2 = kosh.KoshStore(db_uri=db2, dataset_record_type="blah")
        ds_store2 = list(store2.find(name="test"))
        self.assertEqual(len(ds_store2), 1)
        ds2 = ds_store2[0]
        associated_uris = ds2.find(mime_type="py")
        for associated in associated_uris:
            self.assertTrue(associated.uri in new_paths)

        # Cleanup files
        shutil.rmtree("dir1")
        shutil.rmtree(rand + "_pattern_dest")

        # cleanup stores
        for db in [db1, db2]:
            os.remove(db)

    # this should be documented n a notebook and not implemented here

    def test_copy_from_a_store_to_another_in_new_dest_dir(self):
        # kosh cp --stores source  --destination_stores dest --source file1
        # file2, ... --destination dir [--dataset]
        pass

        pass

    # this sholud be done by cping the files
    # Maybe doc via a pipe of search command
    def test_copy_dataset_from_store_to_another_remote(self):
        # kosh cp --stores source  --destination_stores
        # user@machine:/path/to/kosh/store.sql --datasets dataset1_id
        # dataset2_id --destination user@machine:/path/to/destination_directory
        pass

    # This is done in mv
    def test_reassociate_files(self):
        # kosh reassociate --store store1 --source file --destination
        # new_file_path
        pass

    def test_dir_and_files_to_dir(self):
        if sys.platform.startswith("win"):
            print("Skipping test, we are on windows")
            return
        rand = str(random.randint(0, 1000000))
        store1, db1 = self.connect()
        store2, db2 = self.connect()

        file_src_orig = [
            rand + "_df2d/1.py",
            rand + "_df2d/file2.py",
            rand + "_f1.py",
            rand + "_f2.py"]
        try:
            os.removedirs(os.path.dirname(file_src_orig[0]))
        except BaseException:
            pass
        try:
            os.makedirs(os.path.dirname(file_src_orig[0]))
        except BaseException:
            pass
        for src in file_src_orig:
            create_file(src)
        file_src_absolute = [os.path.abspath(f) for f in file_src_orig]
        file_src = [self.source_prefix + f for f in file_src_absolute]
        ds1 = store1.create(name="test")
        ds1.associate(file_src_absolute, mime_type="py")

        dest_name_orig = os.path.abspath(rand + "_df2d_dest")
        try:
            os.removedirs(dest_name_orig)
        except BaseException:
            pass
        try:
            os.makedirs(dest_name_orig)
        except BaseException:
            pass
        dest_name = self.dest_prefix + dest_name_orig
        run_cp([self.source_prefix + os.path.abspath(rand + "_df2d"),
                ] + [self.source_prefix + f for f in file_src_absolute[2:]],
               dest_name,
               [self.source_prefix + db1,
                ],
               [self.dest_prefix + db2,
                ])

        new_paths = []
        for i, file_src in enumerate(file_src_absolute):
            # Test files are moved
            self.assertTrue(os.path.exists(file_src))
            dest = os.path.abspath(
                os.path.join(
                    dest_name_orig,
                    file_src_orig[i]))
            new_paths.append(dest)
            self.assertTrue(os.path.exists(dest))
            # Test datasets are updated

        store1 = kosh.KoshStore(db_uri=db1, dataset_record_type="blah")
        ds_store1 = list(store1.find(name="test"))
        self.assertEqual(len(ds_store1), 1)
        ds1 = ds_store1[0]
        associated_uris = ds1.find(mime_type="testme")
        for associated in associated_uris:
            self.assertTrue(associated.uri in file_src_absolute)

        store2 = kosh.KoshStore(db_uri=db2, dataset_record_type="blah")
        ds_store2 = list(store2.find(name="test"))
        self.assertEqual(len(ds_store2), 1)
        ds2 = ds_store2[0]
        associated_uris = ds2.find(mime_type="py")
        for associated in associated_uris:
            self.assertTrue(associated.uri in new_paths)

        # Cleanup files
        shutil.rmtree(rand + "_df2d")
        shutil.rmtree(rand + "_df2d_dest")

        # cleanup stores
        for db in [db1, db2]:
            os.remove(db)

    def test_file_in_double_dir_to_file(self):
        if sys.platform.startswith("win"):
            print("Skipping test, we are on windows")
            return
        rand = str(random.randint(0, 1000000))
        store1, db1 = self.connect()
        store2, db2 = self.connect()

        file_src_orig = os.path.abspath(
            rand + "_a_dir/another_dir/file_to_file.py")
        try:
            os.remove(file_src_orig)
        except BaseException:
            pass
        try:
            os.removedirs(os.path.dirname(file_src_orig))
            os.removedirs(os.path.dirname(os.path.dirname(file_src_orig)))
        except BaseException:
            pass
        try:
            os.makedirs(os.path.dirname(file_src_orig))
        except BaseException:
            pass
        create_file(file_src_orig)
        file_src = self.source_prefix + file_src_orig
        ds1 = store1.create(name="test")
        ds1.associate(file_src_orig, mime_type="py")

        dest_name_orig = os.path.abspath("file_dest.py")
        dest_name = self.dest_prefix + dest_name_orig
        run_cp([file_src, ], dest_name, [self.source_prefix + db1, ],
               [self.dest_prefix + db2, ])

        # Test files are moved
        self.assertTrue(os.path.exists(file_src_orig))
        self.assertTrue(os.path.exists(dest_name_orig))
        # Test datasets are updated

        store1 = kosh.KoshStore(db_uri=db1, dataset_record_type="blah")
        ds_store1 = list(store1.find(name="test"))
        self.assertEqual(len(ds_store1), 1)
        ds1 = ds_store1[0]
        associated_uris = ds1.find(mime_type="testme")
        for associated in associated_uris:
            self.assertEqual(associated.uri, file_src_orig)

        store2 = kosh.KoshStore(db_uri=db2, dataset_record_type="blah")
        ds_store2 = list(store2.find(name="test"))
        self.assertEqual(len(ds_store2), 1)
        ds2 = ds_store2[0]
        associated_uris = ds2.find(mime_type="py")
        for associated in associated_uris:
            self.assertEqual(associated.uri, dest_name_orig)

        # Cleanup files
        os.remove(file_src_orig)
        os.remove(dest_name_orig)

        # cleanup stores
        for db in [db1, db2]:
            os.remove(db)


class KoshTestCpRemoteDest(KoshTestCp):
    user = user
    hostname = hostname
    source_prefix = ""
    dest_prefix = "{}@{}:".format(user, hostname)


class KoshTestCpRemoteSource(KoshTestCp):
    user = user
    hostname = hostname
    source_prefix = "{}@{}:".format(user, hostname)
    dest_prefix = ""

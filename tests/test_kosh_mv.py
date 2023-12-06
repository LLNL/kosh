from __future__ import print_function
from koshbase import KoshTest
from subprocess import Popen, PIPE
import shlex
import os
import random
import shutil
import sys
import numpy


def create_file(filename):
    with open(filename, "w") as f:
        print("whatever", file=f)


def run_mv(sources, dest, store_sources, store_destinations=None, verbose=False, cmd_extra=''):
    cmd = "python kosh/kosh_command.py mv --dataset_record_type=blah "
    for store in store_sources:
        cmd += " --store {}".format(store)
    if store_destinations is not None:
        for store in store_destinations:
            cmd += " --destination-store {}".format(store)
    cmd += " --sources {} --destination {}".format(" ".join(sources), dest)
    cmd += cmd_extra
    if not sys.platform.startswith("win"):
        cmd = shlex.split(cmd)
    if verbose:
        print("CMD:", cmd)
    p = Popen(cmd, stdout=PIPE, stderr=PIPE)
    o, e = p.communicate()
    out = o, e
    if verbose:
        print("CMD:", cmd)
        print("OUT:", o.decode())
        print("ERR:", e.decode())
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
        if sys.platform.startswith("win"):
            # no mv on win
            return
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
        self.assertFalse(self.file_exist(file_src_orig))
        self.assertTrue(self.file_exist(dest_name_orig))
        for ds in [ds1, ds2]:
            associated = next(ds.find(mime_type="py"))
            self.assertEqual(associated.uri, dest_name_orig)
            # Now dissociate files
            self.assertEqual(1, len(tuple(ds.find(mime_type="py"))))
            ds.dissociate(dest_name_orig)
            self.assertEqual(0, len(tuple(ds.find(mime_type="py"))))

        # cleanup file
        store1.close()
        store2.close()
        os.remove(dest_name_orig)

        # cleanup stores
        for db in [db1, db2]:
            os.remove(db)

    def test_move_files_to_new_directory(self):
        # kosh mv --stores_store1.sql store2.sql --source dir1 --destination
        # dir2
        if sys.platform.startswith("win"):
            # no mv on win
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
            associated_uris = ds.find(mime_type="py")
            for associated in associated_uris:
                self.assertTrue(associated.uri in new_paths)

        # Cleanup files
        shutil.rmtree(rand + "_f2d")
        shutil.rmtree(rand + "_f2d_dest")

        # cleanup stores
        store1.close()
        store2.close()
        for db in [db1, db2]:
            os.remove(db)

    def test_move_directory(self):
        # kosh mv --stores_store1.sql store2.sql --source dir1 --destination
        # dir2
        if sys.platform.startswith("win"):
            # no mv on win
            return
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
            associated_uris = ds.find(mime_type="py")
            for associated in associated_uris:
                self.assertTrue(associated.uri in new_paths)

        # Cleanup files
        shutil.rmtree(rand + "_d2d")
        shutil.rmtree(rand + "_d2d_dest")

        # cleanup stores
        store1.close()
        store2.close()
        for db in [db1, db2]:
            os.remove(db)

    def test_move_files_pattern_to_new_directory_locally(self):
        # kosh mv --stores store1.sql store2.sql --source *.testme --source
        # dir1/testing_it_*.testme --destination dir2
        if sys.platform.startswith("win"):
            # no mv on win
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
                associated_uris = ds.find(mime_type="testme")
                for associated in associated_uris:
                    if os.path.basename(
                            associated.uri) == os.path.basename(file_src):
                        self.assertEqual(associated.uri, dest)
        # Test that file that was not moved still is there
        self.assertTrue(os.path.exists(file_src_orig[-1]))
        self.assertGreater(len(list(ds1.find(uri=file_src_orig_associate[-1]))), 0)

        # Cleanup files
        shutil.rmtree("dir1")
        shutil.rmtree(rand + "_pattern_dest")

        # cleanup stores
        store1.close()
        store2.close()
        for db in [db1, db2]:
            os.remove(db)

    def test_move_file_to_dir(self):
        if sys.platform.startswith("win"):
            # no mv on win
            return
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

        associated = next(ds1.find(mime_type="py"))
        self.assertEqual(associated.uri, dest_path)

        # Cleanup
        shutil.rmtree(dest_name_orig)
        # cleanup stores
        store1.close()
        os.remove(db1)

    def test_move_to_no_exist(self):
        if sys.platform.startswith("win"):
            # no mv on win
            return
        rand = str(random.randint(0, 1000000))
        store1, db1 = self.connect()

        file_src_orig = ["file_to_dir_1a.py", "file_to_dir_2a.py"]
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

        associated = ds1.find(mime_type="py")
        for a in associated:
            self.assertTrue(a.uri in file_src_orig_associate)

        # Cleanup
        for f in file_src_orig_associate:
            os.remove(f)
        # cleanup stores
        store1.close()
        os.remove(db1)

    def test_move_to_no_exist_with_mkdirs(self):
        if sys.platform.startswith("win"):
            # no mv on win
            return
        rand = str(random.randint(0, 1000000))
        store1, db1 = self.connect()

        file_src_orig = ["file_to_dir_1b.py", "file_to_dir_2b.py"]
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

        p, _ = run_mv(file_src_orig_associate, dest_name_orig, [db1, ], verbose=True, cmd_extra=' --mk_dirs')

        # Make sure it passed
        self.assertEqual(p.returncode, 0)

        # cleanup stores
        store1.close()
        os.remove(db1)

    def testWinFail(self):
        if not sys.platform.startswith("win"):
            return
        rand = str(random.randint(0, 1000000))
        store1, db1 = self.connect()

        file_src_orig = ["file_to_dir_1c.py", "file_to_dir_2c.py"]
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
        self.assertEqual(p.returncode, 1)

    def test_triple_subdir(self):
        store, uri = self.connect()
        os.makedirs('a/b/c', exist_ok=True)
        os.makedirs('a/b/c2', exist_ok=True)
        os.makedirs('new_a/new_b/new_c', exist_ok=True)

        numpy.savetxt('a/b/c/a.txt', [1, 2, 3], )
        numpy.savetxt('a/b/c/b.txt', [10, 20, 30], )
        numpy.savetxt('new_a/new_b/new_c/c.txt', [100, 200, 300], )
        numpy.savetxt('new_a/new_b/d.txt', [1000, 2000, 3000], )
        numpy.savetxt('new_a/e.txt', [10000, 20000, 30000], )

        dataset = store.create('foo', metadata={'date': 123, 'dir': 'foodir'})
        dataset.associate('a/b/c/a.txt', mime_type='numpy/txt',
                          metadata={'date': 123.123, 'my_type': 'a', })
        dataset.associate('a/b/c/b.txt', mime_type='numpy/txt',
                          metadata={'date': 123.546, 'my_type': 'b'})
        dataset.associate('new_a/new_b/new_c/c.txt', mime_type='numpy/txt',
                          metadata={'date': 123.123, 'my_type': 'c', })
        dataset.associate('new_a/new_b/d.txt', mime_type='numpy/txt',
                          metadata={'date': 123.546, 'my_type': 'd'})
        dataset.associate('new_a/e.txt', mime_type='numpy/txt',
                          metadata={'date': 123.546, 'my_type': 'e'})

        def check_dataset_and_files(store):
            lst = list(store.find())
            self.assertEqual(len(lst), 1)
            ds = lst[0]
            lst = list(ds.find())
            self.assertEqual(len(lst), 5)
            for associated in lst:
                print(associated.uri)
                self.assertTrue(os.path.exists, associated.uri)
                self.assertEqual(associated.mime_type, "numpy/txt")
                data = ds.get("features", Id=associated.id)
                if os.path.basename(associated.uri) == "a.txt":
                    self.assertTrue(numpy.allclose(data, [1, 2, 3]))
                    self.assertEqual(associated.my_type, "a")
                elif os.path.basename(associated.uri) == "b.txt":
                    self.assertTrue(numpy.allclose(data, [10, 20, 30]))
                    self.assertEqual(associated.my_type, "b")
                elif os.path.basename(associated.uri) == "c.txt":
                    self.assertTrue(numpy.allclose(data, [100, 200, 300]))
                    self.assertEqual(associated.my_type, "c")
                elif os.path.basename(associated.uri) == "d.txt":
                    self.assertTrue(numpy.allclose(data, [1000, 2000, 3000]))
                    self.assertEqual(associated.my_type, "d")
                elif os.path.basename(associated.uri) == "e.txt":
                    self.assertTrue(numpy.allclose(data, [10000, 20000, 30000]))
                    self.assertEqual(associated.my_type, "e")

        check_dataset_and_files(store)

        # manually move a file
        shutil.move("a/b/c/a.txt", "a/b/c2/a.txt")

        # reassociate it
        dataset.reassociate("a/b/c2/a.txt")
        check_dataset_and_files(store)

        # use kosh to mv the file back
        store.mv("a/b/c2/a.txt", "a/b/c/a.txt")
        check_dataset_and_files(store)
        store.mv("a/b/c/", "archive_a/b/", mk_dirs=True)
        # run_mv(["a/b/c/",], "archive_a/b/", [store.db_uri,], verbose=True, cmd_extra=' --mk_dirs')
        check_dataset_and_files(store)

        # mk_dirs with only a file
        store.mv("archive_a/b/c/a.txt", "archive_a2/b2/c2/a.txt", mk_dirs=True)
        check_dataset_and_files(store)

        # mk_dirs keeping directory structure
        store.mv("new_a/", "new_archive/", mk_dirs=True)
        check_dataset_and_files(store)

    def test_linked_folders(self):
        import sina
        store, uri = self.connect()
        dest_dir = 'fifthdir'
        files_to_move = []
        sina_jsons = ['tests/baselines/sina/firstdir/seconddir/thirddir/sina_rec_1_sina.json',
                      'tests/baselines/sina/firstdir/seconddir/thirddir/fourthdir/sina_rec_2_sina.json']

        open('tests/baselines/sina/firstdir_link/seconddir/thirddir/summon_me_as_well.txt', 'a').close()
        open('tests/baselines/sina/firstdir_link/seconddir/thirddir/fourthdir/summon1.txt', 'a').close()

        for sina_json in sina_jsons:
            files_to_move.append(sina_json)
            recs, rels = sina.utils.load_document(sina_json)
            dataset = store.import_dataset(sina_json, match_attributes=["name", "id"])
            for rec in recs:
                for key in rec.files.keys():
                    files_to_move.append(key)

        # CP Files to Dir
        store.cp(files_to_move, dest_dir, mk_dirs=True)

        for file in files_to_move:
            updated_file = os.path.join(dest_dir, os.path.basename(file))
            self.assertTrue(os.path.exists(updated_file))

        # Confirm dataset['files'] has been updated
        files_to_move = []
        for dataset in store.find(load_type='dictionary'):
            self.assertEqual(os.path.abspath(
                os.path.join(
                    dest_dir,
                    os.path.basename(list(dataset['files'].keys())[0]))),
                             list(dataset['files'].keys())[1])
            files_to_move.extend(dataset['files'].keys())

        # MV Files to Dir
        for f in files_to_move:
            if 'fifthdir' in f:
                files_to_move.remove(f)

        dest_dir2 = 'sixth_dir'
        store.mv(files_to_move, dest_dir2, mk_dirs=True)

        for file in files_to_move:
            updated_file = os.path.join(dest_dir2, os.path.basename(file))
            self.assertTrue(os.path.exists(updated_file))

        files_to_move = []
        for dataset in store.find(load_type='dictionary'):
            self.assertEqual(os.path.abspath(
                os.path.join(
                    dest_dir2,
                    os.path.basename(list(dataset['files'].keys())[0]))),
                             list(dataset['files'].keys())[1])
            files_to_move.extend(dataset['files'].keys())

        # MV Dir Link Files to Dir
        for f in files_to_move:
            if 'fifthdir' in f:
                files_to_move.remove(f)

        dest_dir3 = 'sixth_dir_link'
        os.symlink(dest_dir2, dest_dir3)

        dest_dir4 = 'seventh_dir'
        store.mv(dest_dir3+'/*', dest_dir4, mk_dirs=True)

        for file in files_to_move:
            updated_file = os.path.join(dest_dir4, os.path.basename(file))
            self.assertTrue(os.path.exists(updated_file))

        files_to_move = []
        for dataset in store.find(load_type='dictionary'):
            self.assertEqual(os.path.abspath(
                os.path.join(
                    dest_dir4,
                    os.path.basename(list(dataset['files'].keys())[0]))),
                             list(dataset['files'].keys())[1])
            files_to_move.extend(dataset['files'].keys())

        # MV Dir Link to Dir
        for f in files_to_move:
            if 'fifthdir' in f:
                files_to_move.remove(f)

        dest_dir5 = 'seventh_dir_link'
        os.symlink(dest_dir4, dest_dir5)

        dest_dir6 = 'eight_dir'
        store.mv(dest_dir5, dest_dir6, mk_dirs=True)

        for file in files_to_move:
            updated_file = os.path.join(dest_dir6, dest_dir4, os.path.basename(file))
            self.assertTrue(os.path.exists(updated_file))

        files_to_move = []
        for dataset in store.find(load_type='dictionary'):
            self.assertEqual(os.path.abspath(
                os.path.join(
                    dest_dir6, dest_dir4,
                    os.path.basename(list(dataset['files'].keys())[0]))),
                             list(dataset['files'].keys())[1])
            files_to_move.extend(dataset['files'].keys())

        # MV Dir Link to Dir Link
        for f in files_to_move:
            if 'fifthdir' in f:
                files_to_move.remove(f)

        dest_dir7 = 'eigth_dir_link'
        os.symlink(dest_dir6, dest_dir7)

        dest_dir8 = 'ninth_dir'
        dest_dir9 = 'ninth_dir_link'
        os.symlink(dest_dir8, dest_dir9)

        store.mv(dest_dir7, dest_dir9, mk_dirs=True)

        for file in files_to_move:
            updated_file = os.path.join(dest_dir9, dest_dir6, dest_dir4, os.path.basename(file))
            self.assertTrue(os.path.exists(updated_file))

        files_to_move = []
        for dataset in store.find(load_type='dictionary'):
            self.assertEqual(os.path.abspath(
                os.path.join(
                    dest_dir8, dest_dir6, dest_dir4,
                    os.path.basename(list(dataset['files'].keys())[0]))),
                             list(dataset['files'].keys())[1])
            files_to_move.extend(dataset['files'].keys())

        # MV Files Link to Dir
        for i, f in enumerate(files_to_move):
            if 'fifthdir' in f:
                files_to_move.remove(f)

        for i, f in enumerate(files_to_move):
            files_to_move[i] = f.replace('ninth_dir', 'ninth_dir_link')

        dest_dir10 = 'tenth_dir'
        store.mv(files_to_move, dest_dir10, mk_dirs=True)

        for file in files_to_move:
            updated_file = os.path.join(dest_dir10, os.path.basename(file))
            self.assertTrue(os.path.exists(updated_file))

        files_to_move = []
        for dataset in store.find(load_type='dictionary'):
            self.assertEqual(os.path.abspath(
                os.path.join(
                    dest_dir10,
                    os.path.basename(list(dataset['files'].keys())[0]))),
                             list(dataset['files'].keys())[1])
            files_to_move.extend(dataset['files'].keys())

        # CP Dir Link Files to Dir
        for f in files_to_move:
            if 'fifthdir' in f:
                files_to_move.remove(f)

        dest_dir11 = 'tenth_dir_link'
        os.symlink(dest_dir10, dest_dir11)

        dest_dir12 = 'eleventh_dir'
        store.cp(dest_dir11+'/*', dest_dir12, mk_dirs=True)

        for file in files_to_move:
            updated_file = os.path.join(dest_dir12, os.path.basename(file))
            self.assertTrue(os.path.exists(updated_file))

        files_to_move = []
        for dataset in store.find(load_type='dictionary'):
            self.assertEqual(os.path.abspath(
                os.path.join(
                    dest_dir12,
                    os.path.basename(list(dataset['files'].keys())[0]))),
                             list(dataset['files'].keys())[-1])
            files_to_move.extend(dataset['files'].keys())

        # CP Dir Link to Dir
        for f in files_to_move[::-1]:
            if 'fifthdir' in f or 'tenth_dir' in f:
                files_to_move.remove(f)

        dest_dir13 = 'eleventh_dir_link'
        os.symlink(dest_dir12, dest_dir13)

        dest_dir14 = 'twelfth_dir'

        store.cp(dest_dir13, dest_dir14, mk_dirs=True)

        for file in files_to_move:
            updated_file = os.path.join(dest_dir14, dest_dir12, os.path.basename(file))
            self.assertTrue(os.path.exists(updated_file))

        files_to_move = []
        for dataset in store.find(load_type='dictionary'):
            self.assertEqual(os.path.abspath(
                os.path.join(
                    dest_dir14, dest_dir12,
                    os.path.basename(list(dataset['files'].keys())[0]))),
                             list(dataset['files'].keys())[-1])
            files_to_move.extend(dataset['files'].keys())

        # CP Dir Link to Dir Link
        for f in files_to_move[::-1]:
            if 'fifthdir' in f or 'tenth_dir' in f or 'kosh/eleventh_dir' in f:
                files_to_move.remove(f)

        dest_dir15 = 'twelfth_dir_link'
        os.symlink(dest_dir14, dest_dir15)

        dest_dir16 = 'thirteenth_dir'
        dest_dir17 = 'thirteenth_dir_link'
        os.symlink(dest_dir16, dest_dir17)

        store.cp(dest_dir15, dest_dir17, mk_dirs=True)

        for file in files_to_move:
            updated_file = os.path.join(dest_dir17, dest_dir14, dest_dir12, os.path.basename(file))
            self.assertTrue(os.path.exists(updated_file))

        files_to_move = []
        for dataset in store.find(load_type='dictionary'):
            self.assertEqual(os.path.abspath(
                os.path.join(
                    dest_dir16, dest_dir14, dest_dir12,
                    os.path.basename(list(dataset['files'].keys())[0]))),
                             list(dataset['files'].keys())[-1])
            files_to_move.extend(dataset['files'].keys())

        # CP Files Link to Dir
        for f in files_to_move[::-1]:
            if 'fifthdir' in f or 'tenth_dir' in f or 'kosh/eleventh_dir' in f or 'kosh/twelfth_dir' in f:
                files_to_move.remove(f)

        for i, f in enumerate(files_to_move):
            files_to_move[i] = f.replace('thirteenth_dir', 'thirteenth_dir_link')

        dest_dir18 = 'fourteenth_dir'
        store.cp(files_to_move, dest_dir18, mk_dirs=True)

        for file in files_to_move:
            updated_file = os.path.join(dest_dir18, os.path.basename(file))
            self.assertTrue(os.path.exists(updated_file))

        files_to_move = []
        for dataset in store.find(load_type='dictionary'):
            self.assertEqual(os.path.abspath(
                os.path.join(
                    dest_dir18,
                    os.path.basename(list(dataset['files'].keys())[0]))),
                             list(dataset['files'].keys())[-1])
            files_to_move.extend(dataset['files'].keys())

        # cleanup stores
        store.close()
        shutil.rmtree(dest_dir)
        shutil.rmtree(dest_dir2)
        os.remove(dest_dir3)
        shutil.rmtree(dest_dir4)
        os.remove(dest_dir5)
        shutil.rmtree(dest_dir6)
        os.remove(dest_dir7)
        shutil.rmtree(dest_dir8)
        os.remove(dest_dir9)
        shutil.rmtree(dest_dir10)
        os.remove(dest_dir11)
        shutil.rmtree(dest_dir12)
        os.remove(dest_dir13)
        shutil.rmtree(dest_dir14)
        os.remove(dest_dir15)
        shutil.rmtree(dest_dir16)
        os.remove(dest_dir17)
        shutil.rmtree(dest_dir18)


if __name__ == "__main__":
    A = KoshTestMv()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

from __future__ import print_function
import koshbase
import os
import shutil


def create(src, dst):

    for fname in src:
        with open(fname, 'a'):  # Create file if does not exist
            pass

    for dir in dst:
        if os.path.isdir(dir):
            shutil.rmtree(dir)
            os.mkdir(dir)
        else:
            os.mkdir(dir)


class TestKoshMVCPTAR(koshbase.KoshTest):

    def test_store_mv(self):

        src = ['file1_mv_cp_tar.txt', 'file2_mv_cp_tar.txt']
        dst = 'dir1_mv_cp_tar'
        dst2 = 'dir2_mv_cp_tar'
        src2 = [os.path.join(dst, src[0]), os.path.join(dst, src[1])]
        src3 = [os.path.join(dst2, src[0]), os.path.join(dst2, src[1])]

        store1, kosh_db1 = self.connect(sync=False)
        store2, kosh_db2 = self.connect(sync=False)
        store3, kosh_db3 = self.connect(sync=False)
        store4, kosh_db4 = self.connect(sync=False)

        # Simple Source to Destination
        create(src, [dst, dst2])
        store1.mv(src, dst)
        self.assertEqual(os.path.exists(src2[0]), True)
        self.assertEqual(os.path.exists(src2[1]), True)
        store1.mv(src2, dst2)
        self.assertEqual(os.path.exists(src3[0]), True)
        self.assertEqual(os.path.exists(src3[1]), True)
        shutil.rmtree(dst)
        shutil.rmtree(dst2)

        # --stores
        create(src, [dst, dst2])
        store1.mv(src, dst, stores=[store2, store3])
        self.assertEqual(os.path.exists(src2[0]), True)
        self.assertEqual(os.path.exists(src2[1]), True)
        store1.mv(src2, dst2, stores=[store2, store3])
        self.assertEqual(os.path.exists(src3[0]), True)
        self.assertEqual(os.path.exists(src3[1]), True)
        shutil.rmtree(dst)
        shutil.rmtree(dst2)

        # --destination_stores
        create(src, [dst, dst2])
        store1.mv(src, dst, stores=[store2, store3], destination_stores=store4)
        self.assertEqual(os.path.exists(src2[0]), True)
        self.assertEqual(os.path.exists(src2[1]), True)
        store1.mv(src2, dst2, stores=[store2, store3], destination_stores=store4)
        self.assertEqual(os.path.exists(src3[0]), True)
        self.assertEqual(os.path.exists(src3[1]), True)
        shutil.rmtree(dst)
        shutil.rmtree(dst2)

        # --dataset_record_type
        create(src, [dst, dst2])
        store1.mv(src, dst, stores=[store2, store3], destination_stores=store4, dataset_record_type="new_dataset")
        self.assertEqual(os.path.exists(src2[0]), True)
        self.assertEqual(os.path.exists(src2[1]), True)
        store1.mv(src2, dst2, stores=[store2, store3], destination_stores=store4, dataset_record_type="new_dataset")
        self.assertEqual(os.path.exists(src3[0]), True)
        self.assertEqual(os.path.exists(src3[1]), True)
        shutil.rmtree(dst)
        shutil.rmtree(dst2)

        # --version
        create(src, [dst, dst2])
        store1.mv(src, dst, stores=[store2, store3], destination_stores=store4, dataset_record_type="new_dataset",
                  version=True)
        self.assertEqual(os.path.exists(src2[0]), True)
        self.assertEqual(os.path.exists(src2[1]), True)
        store1.mv(src2, dst2, stores=[store2, store3], destination_stores=store4, dataset_record_type="new_dataset",
                  version=True)
        self.assertEqual(os.path.exists(src3[0]), True)
        self.assertEqual(os.path.exists(src3[1]), True)
        shutil.rmtree(dst)
        shutil.rmtree(dst2)

        # --merge_strategy
        create(src, [dst, dst2])
        store1.mv(src, dst, stores=[store2, store3], destination_stores=store4, dataset_record_type="new_dataset",
                  version=True, merge_strategy="preserve")
        self.assertEqual(os.path.exists(src2[0]), True)
        self.assertEqual(os.path.exists(src2[1]), True)
        store1.mv(src2, dst2, stores=[store2, store3], destination_stores=store4, dataset_record_type="new_dataset",
                  version=True, merge_strategy="preserve")
        self.assertEqual(os.path.exists(src3[0]), True)
        self.assertEqual(os.path.exists(src3[1]), True)
        shutil.rmtree(dst)
        shutil.rmtree(dst2)

        store1.close()
        store2.close()
        store3.close()
        store4.close()

        os.remove(kosh_db1)
        os.remove(kosh_db2)
        os.remove(kosh_db3)
        os.remove(kosh_db4)

    def test_store_cp(self):

        src = ['file3_mv_cp_tar.txt', 'file4_mv_cp_tar.txt']
        dst = 'dir3_mv_cp_tar'
        dst2 = 'dir4_mv_cp_tar'
        src2 = [os.path.join(dst, src[0]), os.path.join(dst, src[1])]
        src3 = [os.path.join(dst2, src[0]), os.path.join(dst2, src[1])]

        store1, kosh_db1 = self.connect(sync=False)
        store2, kosh_db2 = self.connect(sync=False)
        store3, kosh_db3 = self.connect(sync=False)
        store4, kosh_db4 = self.connect(sync=False)

        # Simple Source to Destination
        create(src, [dst, dst2])
        store1.cp(src, dst)
        self.assertEqual(os.path.exists(src2[0]), True)
        self.assertEqual(os.path.exists(src2[1]), True)
        store1.cp(src2, dst2)
        self.assertEqual(os.path.exists(src3[0]), True)
        self.assertEqual(os.path.exists(src3[1]), True)
        shutil.rmtree(dst)
        shutil.rmtree(dst2)

        # --stores
        create(src, [dst, dst2])
        store1.cp(src, dst, stores=[store2, store3])
        self.assertEqual(os.path.exists(src2[0]), True)
        self.assertEqual(os.path.exists(src2[1]), True)
        store1.cp(src2, dst2, stores=[store2, store3])
        self.assertEqual(os.path.exists(src3[0]), True)
        self.assertEqual(os.path.exists(src3[1]), True)
        shutil.rmtree(dst)
        shutil.rmtree(dst2)

        # --destination_stores
        create(src, [dst, dst2])
        store1.cp(src, dst, stores=[store2, store3], destination_stores=store4)
        self.assertEqual(os.path.exists(src2[0]), True)
        self.assertEqual(os.path.exists(src2[1]), True)
        store1.cp(src2, dst2, stores=[store2, store3], destination_stores=store4)
        self.assertEqual(os.path.exists(src3[0]), True)
        self.assertEqual(os.path.exists(src3[1]), True)
        shutil.rmtree(dst)
        shutil.rmtree(dst2)

        # --dataset_record_type
        create(src, [dst, dst2])
        store1.cp(src, dst, stores=[store2, store3], destination_stores=store4, dataset_record_type="new_dataset")
        self.assertEqual(os.path.exists(src2[0]), True)
        self.assertEqual(os.path.exists(src2[1]), True)
        store1.cp(src2, dst2, stores=[store2, store3], destination_stores=store4, dataset_record_type="new_dataset")
        self.assertEqual(os.path.exists(src3[0]), True)
        self.assertEqual(os.path.exists(src3[1]), True)
        shutil.rmtree(dst)
        shutil.rmtree(dst2)

        # --version
        create(src, [dst, dst2])
        store1.cp(src, dst, stores=[store2, store3], destination_stores=store4, dataset_record_type="new_dataset",
                  version=True)
        self.assertEqual(os.path.exists(src2[0]), True)
        self.assertEqual(os.path.exists(src2[1]), True)
        store1.cp(src2, dst2, stores=[store2, store3], destination_stores=store4, dataset_record_type="new_dataset",
                  version=True)
        self.assertEqual(os.path.exists(src3[0]), True)
        self.assertEqual(os.path.exists(src3[1]), True)
        shutil.rmtree(dst)
        shutil.rmtree(dst2)

        # --merge_strategy
        create(src, [dst, dst2])
        store1.cp(src, dst, stores=[store2, store3], destination_stores=store4, dataset_record_type="new_dataset",
                  version=True, merge_strategy="preserve")
        self.assertEqual(os.path.exists(src2[0]), True)
        self.assertEqual(os.path.exists(src2[1]), True)
        store1.cp(src2, dst2, stores=[store2, store3], destination_stores=store4, dataset_record_type="new_dataset",
                  version=True, merge_strategy="preserve")
        self.assertEqual(os.path.exists(src3[0]), True)
        self.assertEqual(os.path.exists(src3[1]), True)
        shutil.rmtree(dst)
        shutil.rmtree(dst2)

        store1.close()
        store2.close()
        store3.close()
        store4.close()

        os.remove(kosh_db1)
        os.remove(kosh_db2)
        os.remove(kosh_db3)
        os.remove(kosh_db4)

    def test_store_tar(self):

        src = ['file5_mv_cp_tar.txt', 'file6_mv_cp_tar.txt']
        tar_file = "my_tar_mv_cp_tar.tar"
        create(src, [])

        store1, kosh_db1 = self.connect(sync=False)
        store2, kosh_db2 = self.connect(sync=False)
        store3, kosh_db3 = self.connect(sync=False)

        # Create

        # Simple tar file
        store1.tar(tar_file, "-c", src=src)
        self.assertEqual(os.path.exists(tar_file), True)
        os.remove(tar_file)

        # --stores
        store1.tar(tar_file, "-c", src=src, stores=[store2, store3])
        self.assertEqual(os.path.exists(tar_file), True)
        os.remove(tar_file)

        # --dataset_record_type
        store1.tar(tar_file, "-c", src=src, stores=[store2, store3], dataset_record_type="new_dataset")
        self.assertEqual(os.path.exists(tar_file), True)
        os.remove(tar_file)

        # --no_absolute_path
        store1.tar(tar_file, "-c", src=src, stores=[store2, store3], dataset_record_type="new_dataset",
                   no_absolute_path=True)
        self.assertEqual(os.path.exists(tar_file), True)
        os.remove(tar_file)

        # --merge_strategy
        store1.tar(tar_file, "-c", src=src, stores=[store2, store3], dataset_record_type="new_dataset",
                   no_absolute_path=True, merge_strategy="preserve")
        self.assertEqual(os.path.exists(tar_file), True)

        # Extract

        # Simple tar file
        os.remove(src[0])
        os.remove(src[1])
        store1.tar(tar_file, "-x")
        self.assertEqual(os.path.exists(src[0]), True)
        self.assertEqual(os.path.exists(src[1]), True)

        # --stores
        os.remove(src[0])
        os.remove(src[1])
        store1.tar(tar_file, "-x", stores=[store2, store3])
        self.assertEqual(os.path.exists(src[0]), True)
        self.assertEqual(os.path.exists(src[1]), True)

        # --dataset_record_type
        os.remove(src[0])
        os.remove(src[1])
        store1.tar(tar_file, "-x", stores=[store2, store3], dataset_record_type="new_dataset")
        self.assertEqual(os.path.exists(src[0]), True)
        self.assertEqual(os.path.exists(src[1]), True)

        # --no_absolute_path
        os.remove(src[0])
        os.remove(src[1])
        store1.tar(tar_file, "-x", stores=[store2, store3], dataset_record_type="new_dataset", no_absolute_path=True)
        self.assertEqual(os.path.exists(src[0]), True)
        self.assertEqual(os.path.exists(src[1]), True)

        # --merge_strategy
        os.remove(src[0])
        os.remove(src[1])
        store1.tar(tar_file, "-x", stores=[store2, store3], dataset_record_type="new_dataset", no_absolute_path=True,
                   merge_strategy="preserve")
        self.assertEqual(os.path.exists(src[0]), True)
        self.assertEqual(os.path.exists(src[1]), True)

        store1.close()
        store2.close()
        store3.close()

        os.remove(kosh_db1)
        os.remove(kosh_db2)
        os.remove(kosh_db3)

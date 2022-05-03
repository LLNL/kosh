from __future__ import print_function
from koshbase import KoshTest
import kosh
import os
import numpy


class CustomLoader(kosh.loaders.KoshLoader):
    types = {"xyz": ["numpy"]}

    def extract(self):
        with open(self.obj.uri) as f:
            return numpy.array([float(x) for x in f.read().split(",")])

    def list_features(self):
        return ["data_xyz"]


class TestKoshStoreCustomLoaders(KoshTest):
    def test_store_custom_loader(self):
        store, kosh_db = self.connect()

        ds = store.create(id='123')
        with open("test_kosh_add_custom.xyz", "w") as f:
            print("1, 2, 3, 4", file=f)
        ds.associate("test_kosh_add_custom.xyz", mime_type="xyz")

        feats = ds.list_features()
        self.assertEqual(feats, [])

        store.add_loader(CustomLoader)

        feats = ds.list_features(use_cache=False)
        self.assertEqual(feats, ["data_xyz", ])

        data = ds.get("data_xyz")

        self.assertTrue(numpy.allclose(data, numpy.array([1, 2, 3, 4])))

        # now try to open again and check it was not added to store
        store2 = kosh.KoshStore(db_uri=kosh_db, dataset_record_type="blah")
        ds = store2.open("123")
        feats = ds.list_features()
        self.assertEqual(feats, [])

        # Now add it to the store and store it
        store2.add_loader(CustomLoader, save=True)

        feats = ds.list_features(use_cache=False)
        self.assertEqual(feats, ["data_xyz", ])

        # now loader is in store should know about this type right away
        store3 = kosh.KoshStore(db_uri=kosh_db, dataset_record_type="blah")
        ds = store3.open("123")
        feats = ds.list_features()
        self.assertEqual(feats, ["data_xyz", ])
        store.close()
        store2.close()
        store3.close()
        os.remove(kosh_db)
        os.remove("test_kosh_add_custom.xyz")


if __name__ == "__main__":
    A = TestKoshStoreCustomLoaders()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

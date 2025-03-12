import kosh
import time
from koshbase import KoshTest
import os
from kosh.utils import get_store_info_record_attribute


class FakeLoader(kosh.KoshLoader):
    types = {"fake": [int, ]}

    def extract(self):
        return 2

    def list_features(self, *args, **kargs):
        time.sleep(3)
        return ["fake"]


class FakeLoader2(kosh.KoshLoader):
    types = {"fake": [int, ]}

    def extract(self):
        return 2

    def list_features(self, *args, **kargs):
        print("IN LIST FEATURES*********************", args, kargs)
        time.sleep(3)
        return ["fake"]


class KoshTestList(KoshTest):
    def test_cache_list_features(self):
        store, uri = self.connect()
        store.add_loader(FakeLoader)
        ds = store.create()
        t = time.time()
        ds.associate(
            "tests/baselines/node_extracts2/node_extracts2.hdf5",
            "fake", preload_features=True)
        t = time.time() - t
        # Associate should have called list_features
        self.assertTrue(t > 3)
        t = time.time()
        print(ds["fake"][:])
        t = time.time() - t
        self.assertTrue(t < 3)

        store.close()
        store = kosh.connect(uri)
        print("CACHED FEATURES:", store._cached_features_)
        store.add_loader(FakeLoader)
        ds = next(store.find())  # only one ds in store
        t = time.time()
        print(ds["fake"][:])
        t = time.time() - t
        self.assertTrue(t < 3)
        store.delete_loader(FakeLoader)
        store.add_loader(FakeLoader)
        t = time.time()
        print(ds["fake"][:])
        t = time.time() - t
        self.assertTrue(t > 3)
        recs = store.get_sina_records()
        cached_features = get_store_info_record_attribute(recs, "cached_features")
        self.assertEqual(len(cached_features), 1)
        ds.dissociate("tests/baselines/node_extracts2/node_extracts2.hdf5")
        self.assertEqual(len(store._cached_features_), 0)
        # Now let's pretend it's an old store w/o cached_features in the store rec
        ds.associate(
            "tests/baselines/node_extracts2/node_extracts2.hdf5",
            "fake", preload_features=True)
        self.assertEqual(len(store._cached_features_), 1)
        store._cached_features_ = {}  # reset
        self.assertEqual(len(store._cached_features_), 0)
        store.close()
        store = kosh.connect(uri)
        store.add_loader(FakeLoader)
        ds = next(store.find())  # only one ds in store
        t = time.time()
        ds.associate(
            "tests/baselines/node_extracts2/node_extracts2.hdf5",
            "fake", preload_features=True)
        t = time.time() - t
        self.assertTrue(t > 3)
        t = time.time()
        print(ds["fake"][:])
        t = time.time() - t
        self.assertTrue(t < 3)
        store.close()
        os.remove(uri)

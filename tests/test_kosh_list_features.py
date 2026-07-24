import kosh
from koshbase import KoshTest
import os
from kosh.utils import get_store_info_record_attribute


class FakeLoader(kosh.KoshLoader):
    types = {"fake": [int, ]}
    call_count = 0

    def extract(self):
        return 2

    def list_features(self, *args, **kargs):
        FakeLoader.call_count += 1
        return ["fake"]


class FakeLoader2(kosh.KoshLoader):
    types = {"fake": [int, ]}
    call_count = 0

    def extract(self):
        return 2

    def list_features(self, *args, **kargs):
        FakeLoader2.call_count += 1
        print("IN LIST FEATURES*********************", args, kargs)
        return ["fake"]


class KoshTestList(KoshTest):
    def test_cache_list_features(self):
        FakeLoader.call_count = 0
        store, uri = self.connect()
        store.add_loader(FakeLoader)
        ds = store.create()
        ds.associate(
            "tests/baselines/node_extracts2/node_extracts2.hdf5",
            "fake", preload_features=True)
        self.assertEqual(FakeLoader.call_count, 1)
        print(ds["fake"][:])
        self.assertEqual(FakeLoader.call_count, 1)

        store.close()
        store = kosh.connect(uri)
        print("CACHED FEATURES:", store._cached_features_)
        store.add_loader(FakeLoader)
        ds = next(store.find())  # only one ds in store
        print(ds["fake"][:])
        self.assertEqual(FakeLoader.call_count, 1)
        store.delete_loader(FakeLoader)
        store.add_loader(FakeLoader)
        previous_calls = FakeLoader.call_count
        print(ds["fake"][:])
        self.assertGreater(FakeLoader.call_count, previous_calls)
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
        previous_calls = FakeLoader.call_count
        ds.associate(
            "tests/baselines/node_extracts2/node_extracts2.hdf5",
            "fake", preload_features=True)
        self.assertGreater(FakeLoader.call_count, previous_calls)
        print(ds["fake"][:])
        self.assertEqual(FakeLoader.call_count, previous_calls + 1)
        store.close()
        os.remove(uri)

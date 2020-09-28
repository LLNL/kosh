import os
from koshbase import KoshTest
import time
from sina.utils import DataRange


class KoshTestSync(KoshTest):
    def test_sync_mode_switch(self):
        store, kosh_db = self.connect()
        self.assertTrue(store.is_synchronous())
        self.assertTrue(store.__sync__)
        self.assertFalse(store.synchronous())
        self.assertFalse(store.__sync__)
        self.assertFalse(store.is_synchronous())
        self.assertTrue(store.synchronous())
        self.assertTrue(store.__sync__)
        self.assertTrue(store.is_synchronous())
        self.assertTrue(store.synchronous(True))
        self.assertTrue(store.__sync__)
        self.assertTrue(store.is_synchronous())
        self.assertFalse(store.synchronous(False))
        self.assertFalse(store.__sync__)
        self.assertFalse(store.is_synchronous())
        self.assertFalse(store.synchronous(False))
        self.assertFalse(store.__sync__)
        self.assertTrue(store.synchronous(True))
        self.assertTrue(store.__sync__)
        self.assertTrue(store.is_synchronous())
        os.remove(kosh_db)

    def test_sync_search(self):
        store, kosh_db = self.connect()
        store2, kosh_db = self.connect(db_uri=kosh_db, sync=False)
        # Create many datasets
        ds = store.create(metadata={"key1": 1, "key2": "A"})
        ds2 = store2.create(metadata={"key2": "B", "key3": 3})
        ds3 = store.create()
        store2.create(metadata={"key2": "C", "key3": 4})
        ds.associate("tests/baselines/node_extracts2", "something")
        ds2.associate("tests/baselines/node_extracts2", "something")
        ds3.associate("tests/baselines/node_extracts2", "something")

        s = store.search(key2=DataRange("A"))
        self.assertEqual(len(s), 1)
        s = store2.search(key2=DataRange("A"))
        self.assertEqual(len(s), 3)

        s = store.search(
            key2=DataRange("A"),
            file=os.path.abspath("tests/baselines/node_extracts2"))
        self.assertEqual(len(s), 1)
        s = store2.search(
            key2=DataRange("A"),
            file=os.path.abspath("tests/baselines/node_extracts2"))
        self.assertEqual(len(s), 2)
        store2.sync()
        s = store.search(key2=DataRange("A"))
        self.assertEqual(len(s), 3)
        s = store2.search(key2=DataRange("A"))
        self.assertEqual(len(s), 3)

        s = store.search(
            key2=DataRange("A"),
            file=os.path.abspath("tests/baselines/node_extracts2"))
        self.assertEqual(len(s), 2)
        s = store2.search(
            key2=DataRange("A"),
            file=os.path.abspath("tests/baselines/node_extracts2"))
        self.assertEqual(len(s), 2)

        store2.sync()
        os.remove(kosh_db)

    def test_sync_delete_dataset(self):
        store1, kosh_db = self.connect(sync=True)
        store2, kosh_db = self.connect(db_uri=kosh_db, sync=False)
        # Create dataset on syncing store
        ds1 = store1.create()
        dsid = ds1.__id__
        # Check it exists on store2
        self.assertEqual(len(store1.search()), 1)
        self.assertEqual(len(store2.search()), 1)
        store2.delete(dsid)
        # self.assertEqual(len(store2.search()),0)
        store2.sync()
        self.assertEqual(len(store1.search()), 0)
        store2, kosh_db = self.connect(db_uri=kosh_db)
        with self.assertRaises(Exception):
            store2.open(dsid)
        self.assertEqual(len(store2.search()), 0)

    def test_sync_dataset_attributes(self):
        store1, kosh_db = self.connect(sync=True)
        store2, kosh_db = self.connect(db_uri=kosh_db, sync=False)
        # Create dataset on syncing store
        ds1 = store1.create()
        ds1.test_sync = "Set"
        # Check it exists on store2
        ds2 = store2.open(ds1.__id__)
        # Check they are identical
        self.assertEqual(ds2.__id__, ds1.__id__)
        self.assertEqual(ds2.test_sync, ds1.test_sync)
        # Change in store1, shouldn't change on store2 until synced
        ds1.test_sync = "Changed"
        self.assertEqual(ds1.test_sync, "Changed")
        self.assertNotEqual(ds1.test_sync, ds2.test_sync)
        self.assertEqual(ds2.test_sync, "Set")
        # Sync dataset
        ds2.sync()
        self.assertEqual(ds2.test_sync, ds1.test_sync)
        self.assertEqual(ds1.test_sync, "Changed")
        # Another change
        ds2.test_sync = "Changed from 2nd store"
        self.assertEqual(ds2.test_sync, "Changed from 2nd store")
        self.assertNotEqual(ds1.test_sync, ds2.test_sync)
        self.assertEqual(ds1.test_sync, "Changed")
        ds2.sync()
        self.assertEqual(ds2.test_sync, ds1.test_sync)
        self.assertEqual(ds2.test_sync, "Changed from 2nd store")
        self.assertEqual(ds1.test_sync, "Changed from 2nd store")
        # Now change on store
        ds3 = store2.create()
        ds3.test_sync = "exists"
        # Check it does not exists on store1
        with self.assertRaises(Exception):
            ds3 = store1.open(ds3.__id__)
        ds2.test_sync = "Another change"
        self.assertNotEqual(ds1.test_sync, ds2.test_sync)
        # Sync the store
        store2.sync()
        self.assertEqual(ds2.test_sync, ds1.test_sync)
        self.assertEqual(ds1.test_sync, "Another change")
        ds3 = store1.open(ds3.__id__)
        self.assertEqual(ds3.test_sync, "exists")
        # ok now test it fails if store changed in between
        ds2.test_sync = "I changed it"
        time.sleep(.1)
        ds1.test_sync = "I changed it after you"
        with self.assertRaises(RuntimeError):
            ds2.sync()
        with self.assertRaises(RuntimeError):
            ds2.sync()
        self.assertEqual(ds2.test_sync, "I changed it")
        with self.assertRaises(RuntimeError):
            ds2.sync()
        self.assertEqual(ds1.test_sync, "I changed it after you")
        with self.assertRaises(RuntimeError):
            ds2.sync()
        ds2.test_sync = ds1.test_sync
        ds2.sync()
        self.assertEqual(ds1.test_sync, "I changed it after you")
        self.assertEqual(ds2.test_sync, "I changed it after you")

        # Now testing deletion stuff
        del(ds1.test_sync)
        ds2.test_sync = "Ok let's change you"
        with self.assertRaises(RuntimeError):
            ds2.sync()
        del(ds2.test_sync)
        ds2.sync()

        ds2.associate("ghost", "not_real")
        self.assertNotEqual(ds2._associated_data_, ds1._associated_data_)
        ds1.associate("ghostly", "fake")
        ds2.associate("ghostlier", "not_real_as_well")
        self.assertEqual(len(ds2._associated_data_), 2)
        ds2.sync()
        self.assertEqual(len(ds2._associated_data_), 3)
        self.assertEqual(ds2._associated_data_, ds1._associated_data_)
        ds2.dissociate("ghost")
        self.assertEqual(len(ds2._associated_data_), 2)
        self.assertEqual(len(ds1._associated_data_), 3)
        ds2.sync()
        self.assertEqual(len(ds1._associated_data_), 2)
        self.assertEqual(len(ds2._associated_data_), 2)

        # Ok now let's see if we do conflict
        ds2.associate("conflict", "conf")
        ds1.associate("conflict", "conf2")

        with self.assertRaises(RuntimeError):
            ds2.sync()
        ds2.dissociate("conflict")
        ds2.associate("conflict", "conf2")
        ds2.sync()
        store2.sync()
        os.remove(kosh_db)

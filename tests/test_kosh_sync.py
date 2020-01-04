import os
from koshbase import KoshTest
import kosh
import time

class KoshTestSync(KoshTest):
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
        self.assertEqual(ds2.test_sync, "I changed it")
        self.assertEqual(ds1.test_sync, "I changed it after you")

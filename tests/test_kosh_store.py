from __future__ import print_function
from kosh.store import KoshStore
import os
import kosh
import random
from koshbase import KoshTest


class KoshTestStore(KoshTest):
    def test_connect_base_function(self):
        _, kosh_test_sql_file = self.connect()
        os.remove(kosh_test_sql_file)

    def test_connect(self):
        seed = random.randint(0, 1000000000)
        db_name = "test_connect_{}.sql".format(seed)
        if os.path.exists(db_name):
            os.remove(db_name)

        store = kosh.connect(db_name)

        self.assertTrue(os.path.exists(db_name))

        store.create()
        store.close()

        store = kosh.connect(db_name)
        self.assertEqual(len(list(store.find())), 1)
        os.remove(db_name)

    def test_read_only(self):
        seed = random.randint(0, 1000000000)
        db_name = "test_connect_{}.sql".format(seed)
        if os.path.exists(db_name):
            os.remove(db_name)

        store = kosh.connect(db_name, read_only=True)
        store.create()
        with self.assertRaises(RuntimeError):
            store.sync()

        # Make sure another store can read this
        store2 = kosh.connect(db_name)
        store2.create()

        os.remove(db_name)

    def test_wipe_on_open(self):
        store, db = self.connect()
        store.create()
        self.assertEqual(len(list(store.find())), 1)

        store = kosh.connect(db, delete_all_contents=True)
        self.assertEqual(len(list(store.find())), 0)

        os.remove(db)

    def test_create(self):
        self.assertIsInstance(
            kosh.create_new_db("blah_blah_blah.sql"),
            kosh.KoshStore)
        os.remove("blah_blah_blah.sql")

    def test_associate_stores_error_trap(self):
        store, db = self.connect()
        store_2, db_2 = self.connect()

        store.create(name="main")
        store_2.create(name="associated")

        store.associate(store_2)
        # double association shouldn't matter
        store.associate(store_2)
        self.assertEqual(len(list(store.find(ids_only=True))), 2)

        # Single dissociation should work
        store.dissociate(store_2)
        self.assertEqual(len(list(store.find(ids_only=True))), 1)

        # double dissociation shouldn't matter
        store.dissociate(store_2)
        self.assertEqual(len(list(store.find(ids_only=True))), 1)

        # associate bad store
        with self.assertRaises(TypeError):
            store.associate("setup.py")
        with self.assertRaises(TypeError):
            store.associate(1)

        # good store gone after opening!
        # Looks like Sina can't catch that because it's cached?
        """
        new_store, new_db = self.connect()
        new_store.create()
        store.associate(new_store)
        self.assertEqual(len(list(new_store.find(ids_only=True))), 1)
        os.remove(new_db)
        self.assertFalse(os.path.exists(new_db))
        # self.assertEqual(len(list(store.find(ids_only=True))), 1)
        """
        os.remove(db)

    def test_chained_associate_1(self):
        central_store, db = self.connect()
        sub_store, db_sub = self.connect()
        third_store, db_3 = self.connect()

        central_store.create(name="dataset_in_central_store")
        sub_store.create(name="dataset_in_sub_store")

        self.assertEqual(len(list(central_store.find())), 1)

        central_store.associate(sub_store)
        self.assertEqual(len(list(central_store.find())), 2)
        self.assertEqual(len(list(sub_store.find())), 1)

        central_store.dissociate(sub_store)
        self.assertEqual(len(list(central_store.find())), 1)

        central_store.associate(sub_store, reciprocal=True)
        self.assertEqual(len(list(central_store.find())), 2)
        self.assertEqual(len(list(sub_store.find())), 2)

        third_store.associate(sub_store)
        self.assertEqual(len(list(third_store.find())), 2)

        sub_store.dissociate(central_store, reciprocal=True)
        self.assertEqual(len(list(third_store.find())), 1)

        stores_uris = list(third_store.get_associated_stores())
        self.assertEqual(len(stores_uris), 1)
        print(stores_uris)
        self.assertTrue(isinstance(stores_uris[0], str))

        stores_uris = list(third_store.get_associated_stores(uris=False))
        self.assertEqual(len(stores_uris), 1)
        self.assertTrue(isinstance(stores_uris[0], KoshStore))

        sub_copy = third_store.get_associated_store(sub_store.db_uri)
        self.assertEqual(sub_copy, sub_store)

        os.remove(db)
        os.remove(db_sub)
        os.remove(db_3)

    def test_chained_associate_2(self):
        store, db = self.connect()
        store_2, db_2 = self.connect()
        store_3, db_3 = self.connect()

        store.create(name="main")
        store_2.create(name="associated")

        store.associate(store_2, reciprocal=True)
        self.assertEqual(len(list(store_2.find())), 2)
        asso = list(store_2.get_associated_stores())
        self.assertEqual(len(asso), 1)
        self.assertEqual(asso[0], store.db_uri)
        asso = list(store_2.get_associated_stores(uris=False))
        self.assertEqual(asso[0], store)

        # Chained store should lead to discovery of both datasets
        # even though this store itself is empty
        self.assertEqual(len(list(store_3.find())), 0)
        store_3.associate(store_2)
        store_2_copy = store_3.get_associated_store(store_2.db_uri)
        self.assertEqual(store_2, store_2_copy)
        self.assertEqual(len(list(store_3.find())), 2)

        store_2.dissociate(store.db_uri)
        # Store_2 does not know about store any more -> no chaining
        # note that store still knows about store_2
        self.assertEqual(len(list(store_3.find())), 1)
        self.assertEqual(len(list(store.find())), 2)
        os.remove(db)
        os.remove(db_2)
        os.remove(db_3)

    def test_associate_stores_reciprocal(self):
        store, db = self.connect()
        store_2, db_2 = self.connect()
        store_3, db_3 = self.connect()

        store.create(name="main")
        store_2.create(name="associated")
        store_3.create(name="associated_2")

        # other store should be changed
        store.associate(store_2, reciprocal=True)
        self.assertEqual(len(list(store.find(ids_only=True))), 2)
        self.assertEqual(len(list(store_2.find(ids_only=True))), 2)

        # Associating second store to third should link it
        store_3.associate(store_2, reciprocal=True)
        self.assertEqual(len(list(store_3.find(ids_only=True))), 3)

        # dissociate should go back to normal
        store_2.dissociate(store, reciprocal=True)
        self.assertEqual(len(list(store.find(ids_only=True))), 1)
        self.assertEqual(len(list(store_2.find(ids_only=True))), 2)
        self.assertEqual(len(list(store_3.find(ids_only=True))), 2)

        # reciprocal associate and unilateral dissociate
        store.associate(store_2, reciprocal=True)
        store_2.dissociate(store)
        store_3.dissociate(store_2)
        self.assertEqual(len(list(store.find(ids_only=True))), 3)
        self.assertEqual(len(list(store_2.find(ids_only=True))), 2)
        self.assertEqual(len(list(store_3.find(ids_only=True))), 1)

        os.remove(db)
        os.remove(db_2)
        os.remove(db_3)

    def test_associate_stores(self):
        store, db = self.connect()
        store_2, db_2 = self.connect()
        store_3, db_3 = self.connect()

        store.create(name="main")
        store_2.create(name="associated")
        store_3.create(name="associated_2")

        # Make sure no association are in place (e.g 1 ds per store)
        self.assertEqual(len(list(store.find(ids_only=True))), 1)
        self.assertEqual(len(list(store_2.find(ids_only=True))), 1)
        self.assertEqual(len(list(store_3.find(ids_only=True))), 1)
        # Make sure all datasets are in the correct store
        self.assertEqual(
            len(list(store.find(name="associated", ids_only=True))), 0)
        self.assertEqual(
            len(list(store.find(name="associated_2", ids_only=True))), 0)
        self.assertEqual(
            len(list(store_2.find(name="associated", ids_only=True))), 1)
        self.assertEqual(
            len(list(store_3.find(name="associated_2", ids_only=True))), 1)

        # Now associate a store with the main
        # Main should find datasets of associated store
        # but not other way around
        # other store should be unchanged
        store.associate(store_2)
        self.assertEqual(len(list(store.find(ids_only=True))), 2)
        self.assertEqual(len(list(store_2.find(ids_only=True))), 1)
        self.assertEqual(len(list(store_3.find(ids_only=True))), 1)
        # Make sure the correct new ds is found
        self.assertEqual(
            len(list(store.find(name="associated", ids_only=True))), 1)
        self.assertEqual(
            len(list(store.find(name="associated_2", ids_only=True))), 0)

        # Now associate another store with the first assoc store
        # should be picked up by first store
        store_2.associate(store_3)
        self.assertEqual(len(list(store.find(ids_only=True))), 3)
        self.assertEqual(len(list(store_2.find(ids_only=True))), 2)
        self.assertEqual(len(list(store_3.find(ids_only=True))), 1)
        self.assertEqual(
            len(list(store.find(name="associated", ids_only=True))), 1)
        self.assertEqual(
            len(list(store.find(name="associated_2", ids_only=True))), 1)
        self.assertEqual(
            len(list(store_2.find(name="associated", ids_only=True))), 1)
        self.assertEqual(
            len(list(store_2.find(name="associated_2", ids_only=True))), 1)
        self.assertEqual(
            len(list(store_3.find(name="associated", ids_only=True))), 0)

        # now backward let's dissociate store 2
        store_2.dissociate(store_3)
        self.assertEqual(len(list(store.find(ids_only=True))), 2)
        self.assertEqual(len(list(store_2.find(ids_only=True))), 1)
        self.assertEqual(len(list(store_3.find(ids_only=True))), 1)
        # Make sure the correct new dataset is found
        self.assertEqual(
            len(list(store.find(name="associated", ids_only=True))), 1)
        self.assertEqual(
            len(list(store.find(name="associated_2", ids_only=True))), 0)

        # dissociate everything
        store.dissociate(store_2)
        # Make sure no association are in place (e.g 1 ds per store)
        self.assertEqual(len(list(store.find(ids_only=True))), 1)
        self.assertEqual(len(list(store_2.find(ids_only=True))), 1)
        self.assertEqual(len(list(store_3.find(ids_only=True))), 1)
        # Make sure all datasets are in the correct store
        self.assertEqual(
            len(list(store.find(name="associated", ids_only=True))), 0)
        self.assertEqual(
            len(list(store.find(name="associated_2", ids_only=True))), 0)
        self.assertEqual(
            len(list(store_2.find(name="associated", ids_only=True))), 1)
        self.assertEqual(
            len(list(store_3.find(name="associated_2", ids_only=True))), 1)

        os.remove(db)
        os.remove(db_3)
        os.remove(db_2)


if __name__ == "__main__":
    A = KoshTestStore()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

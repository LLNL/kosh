from __future__ import print_function
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
            kosh.sina.KoshSinaStore)
        os.remove("blah_blah_blah.sql")


if __name__ == "__main__":
    A = KoshTestStore()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

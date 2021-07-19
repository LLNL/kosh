from __future__ import print_function
import os
import kosh
import json
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
        self.assertEqual(len(list(store.search())), 1)
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
        self.assertEqual(len(list(store.search())), 1)

        store = kosh.connect(db, delete_all_contents=True)
        self.assertEqual(len(list(store.search())), 0)

        os.remove(db)

    def test_create(self):
        self.assertIsInstance(
            kosh.create_new_db("blah_blah_blah.sql"),
            kosh.sina.KoshSinaStore)
        os.remove("blah_blah_blah.sql")

    def test_import_export_datsets(self):
        store, kosh_test_sql_file = self.connect()
        store2, kosh_test_sql_file2 = self.connect()

        ds1 = store.create(name="one", metadata={"param1": 5, "param2": 6})
        # import via dataset.export
        store2.import_dataset(ds1.export())
        self.assertEqual(len(list(store2.search(name="one"))), 1)

        ds2 = store.create(name="two", metadata={"param1": 5, "param2": 3})
        # import dataset directly
        store2.import_dataset(ds2)
        self.assertEqual(len(list(store2.search(name="two"))), 1)

        # Import again should work
        store2.import_dataset(ds2)
        d2 = list(store2.search(name="two"))
        self.assertEqual(len(d2), 1)

        # Import again should work even though we added an attribute
        ds2.param3 = "blah"
        store2.import_dataset(ds2)
        d2 = list(store2.search(name="two"))
        self.assertEqual(len(d2), 1)
        self.assertEqual(d2[0].param3, "blah")

        # if we alter it should not work though
        ds2.param2 = 7
        with self.assertRaises(ValueError) as context:
            store2.import_dataset(ds2)
        self.assertTrue(
            "Trying to import dataset with attribute 'param2' value :"
            " 7. But value for this attribute in target is '3'" in str(
                context.exception))

        # now let's create another dataset named 'one'
        # Should prevent re-importing it
        ds1b = store2.create("one", metadata={"p1": 6})
        self.assertEqual(len(list(store2.search(name="one"))), 2)

        with self.assertRaises(ValueError):
            store2.import_dataset(ds1b)

        # but making it more specific when matching should help
        ds1.param1 = 'b'
        # Attribute changed so should reject
        with self.assertRaises(ValueError) as context:
            store2.import_dataset(ds1, match_attributes=["name", "param2"])
        self.assertTrue(
            "Trying to import dataset with attribute 'param1' value : b. "
            "But value for this attribute in target is '5'" in str(
                context.exception))

        # Now using param1 should lead to creation of new dataset since no
        # match in dest store
        store2.import_dataset(ds1, match_attributes=["name", "param1"])
        d1 = list(store2.search(name="one"))
        self.assertEqual(len(d1), 3)
        d1 = list(store2.search(param1='b', name="one"))
        self.assertEqual(len(d1), 1)

        # Let's make sure associated files are transfered
        ds = store.create(name="foo_association")
        ds.associate("setup.py", "py")
        store2.import_dataset(ds)
        ds2 = list(store2.search(name=ds.name))[0]
        self.assertEqual(len(ds2._associated_data_), 1)
        self.assertEqual(
            store2._load(
                ds2._associated_data_[0]).uri,
            os.path.abspath("setup.py"))

        json_name = "tests/kosh_export.json"
        if os.path.exists(json_name):
            os.remove(json_name)
        ds.export(json_name)
        self.assertTrue(os.path.exists(json_name))

        with open(json_name) as f:
            data = json.load(f)

        # len 2 because of  associated data
        self.assertEqual(len(data["records"]), 2)
        ds.export(json_name)

        with open(json_name) as f:
            data = json.load(f)

        # len 2 because of  associated data
        self.assertEqual(len(data["records"]), 2)
        # Now test that we can overwrite exisitng dataset with new one
        ds1 = store.create(metadata={"a": 1, "b": 2, "c": 3, "d": 4})
        ds2 = store2.create(metadata={"a": 1, "b": 2, "c": 4})
        store2.import_dataset(ds1.export(), match_attributes=[
                              "a", "b"], merge_handler="overwrite")
        self.assertEqual(ds2.c, 3)
        self.assertEqual(ds2.d, 4)
        # revert to test preserve
        ds2.c = 4
        store2.import_dataset(ds1.export(), match_attributes=[
                              "a", "b"], merge_handler="preserve")
        self.assertEqual(ds2.c, 4)

        os.remove(kosh_test_sql_file)
        os.remove(kosh_test_sql_file2)
        os.remove(json_name)


if __name__ == "__main__":
    A = KoshTestStore()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

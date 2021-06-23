import os
import kosh
from koshbase import KoshTest


class KoshTestStore(KoshTest):
    def test_connect(self):
        store, kosh_test_sql_file = self.connect()
        os.remove(kosh_test_sql_file)

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
        self.assertEqual(len(store2.search(name="one")), 1)

        ds2 = store.create(name="two", metadata={"param1": 5, "param2": 3})
        # import dataset directly
        store2.import_dataset(ds2)
        self.assertEqual(len(store2.search(name="two")), 1)

        # Import again should work
        store2.import_dataset(ds2)
        d2 = store2.search(name="two")
        self.assertEqual(len(d2), 1)

        # Import again should work even though we added an attribute
        ds2.param3 = "blah"
        store2.import_dataset(ds2)
        d2 = store2.search(name="two")
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
        self.assertEqual(len(store2.search(name="one")), 2)

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
        d1 = store2.search(name="one")
        self.assertEqual(len(d1), 3)
        d1 = store2.search(param1='b', name="one")
        self.assertEqual(len(d1), 1)

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


if __name__ == "__main__":
    A = KoshTestStore()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

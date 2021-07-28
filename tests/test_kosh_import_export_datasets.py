from __future__ import print_function
import os
import json
from koshbase import KoshTest
import sina


class KoshTestImportExport(KoshTest):
    def test_import_from_sina(self):
        store, kosh_test_sql_file = self.connect()

        store.import_dataset("tests/baselines/sina/sina_curve_rec_mimes_and_curves.json")

        datasets = list(store.find())
        self.assertEqual(len(datasets), 1)
        dataset = datasets[0]
        asso = list(dataset.find())
        self.assertEqual(len(asso), 1)
        asso = list(dataset.find(mime_type="hdf5"))
        self.assertEqual(len(asso), 1)
        asso = list(dataset.find(mime_type="foo"))
        self.assertEqual(len(asso), 0)
        asso = list(dataset.find("summary_hdf5"))
        self.assertEqual(len(asso), 1)
        asso = list(dataset.find(summary_hdf5=sina.utils.exists()))
        self.assertEqual(len(asso), 1)
        asso = list(dataset.find(summary_hdf5="some_Value"))
        self.assertEqual(len(asso), 0)
        features = dataset.list_features()
        self.assertEqual(len(features), 28)

        os.remove(kosh_test_sql_file)

    def test_import_export_datsets(self):
        store, kosh_test_sql_file = self.connect()
        store2, kosh_test_sql_file2 = self.connect()

        ds1 = store.create(name="one", metadata={"param1": 5, "param2": 6})
        # import via dataset.export
        store2.import_dataset(ds1.export())
        self.assertEqual(len(list(store2.find(name="one"))), 1)

        ds2 = store.create(name="two", metadata={"param1": 5, "param2": 3})
        # import dataset directly
        store2.import_dataset(ds2)
        self.assertEqual(len(list(store2.find(name="two"))), 1)

        # Import again should work
        store2.import_dataset(ds2)
        d2 = list(store2.find(name="two"))
        self.assertEqual(len(d2), 1)

        # Import again should work even though we added an attribute
        ds2.param3 = "blah"
        store2.import_dataset(ds2)
        d2 = list(store2.find(name="two"))
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
        self.assertEqual(len(list(store2.find(name="one"))), 2)

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
        d1 = list(store2.find(name="one"))
        self.assertEqual(len(d1), 3)
        d1 = list(store2.find(param1='b', name="one"))
        self.assertEqual(len(d1), 1)

        # Let's make sure associated files are transfered
        ds = store.create(name="foo_association")
        ds.associate("setup.py", "py")
        store2.import_dataset(ds)
        ds2 = list(store2.find(name=ds.name))[0]
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
    A = KoshTestImportExport()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

from __future__ import print_function
import os
import json
from koshbase import KoshTest
import sina
import numpy


class KoshTestImportExport(KoshTest):
    def test_import_from_sina(self):
        store, kosh_test_sql_file = self.connect()

        store.import_dataset(
            "tests/baselines/sina/sina_curve_rec_mimes_and_curves.json")

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

        store.close()
        os.remove(kosh_test_sql_file)

    def test_import_export_datasets(self):
        store, kosh_test_sql_file = self.connect()
        store2, kosh_test_sql_file2 = self.connect()
        store3, kosh_test_sql_file3 = self.connect()

        ds1 = store.create(name="one", metadata={"param1": 5, "param2": 6})
        # import via dataset.export
        store2.import_dataset(ds1.export())
        self.assertEqual(len(list(store2.find(name="one"))), 1)

        ds2 = store.create(name="two", metadata={"param1": 5, "param2": 3})
        # import dataset directly
        store2.import_dataset(ds2)
        self.assertEqual(len(list(store2.find(name="two"))), 1)

        # Import again should work
        # but not add a dataset
        store2.import_dataset(ds2)
        d2 = list(store2.find(name="two"))
        self.assertEqual(len(d2), 1)

        # Import again should work even though we added an attribute
        # the dataset in store2 should be updated
        ds2.param3 = "blah"
        store2.import_dataset(ds2)
        d2 = list(store2.find(name="two"))
        self.assertEqual(len(d2), 1)
        self.assertEqual(d2[0].param3, "blah")

        # if we alter it should not work by default though
        ds2.param2 = 7
        with self.assertRaises(ValueError) as context:
            store2.import_dataset(ds2)
        self.assertTrue(
            "Trying to import dataset with attribute 'param2' value :"
            " 7. But value for this attribute in target is '3'" in str(
                context.exception))

        # now let's create another dataset named 'one'
        # Should prevent re-importing it since conflict
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

        # Let's make sure associated files are transferred
        ds = store.create(name="foo_association")
        ds.associate("setup.py", "py")
        self.assertEqual(len(ds2._associated_data_), 0)
        store2.import_dataset(ds)
        ds2 = list(store2.find(name=ds.name))[0]
        print(ds2)
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
        # Now test that we can overwrite existing dataset with a new one
        ds1 = store.create(metadata={"a": 1, "b": 2, "c": 3, "d": 4})
        ds2 = store2.create(metadata={"a": 1, "b": 2, "c": 4})
        # import in overwrite mode 'c' should become 3
        store2.import_dataset(ds1.export(), match_attributes=[
                              "a", "b"], merge_handler="overwrite")
        self.assertEqual(ds2.c, 3)
        self.assertEqual(ds2.d, 4)
        # revert to test preserve
        ds2.c = 4
        # Now import again but in preserve mode 'c' shouldn't change
        store2.import_dataset(ds1.export(), match_attributes=[
                              "a", "b"], merge_handler="preserve")
        self.assertEqual(ds2.c, 4)

        store3.import_dataset([ds, ds2])
        self.assertEqual(len(list(store3.find())), 2)
        store.close()
        store2.close()
        store3.close()
        os.remove(kosh_test_sql_file)
        os.remove(kosh_test_sql_file2)
        os.remove(kosh_test_sql_file3)
        os.remove(json_name)

    def test_import_merge_overwrite_curves(self):
        store, uri = self.connect()
        store.import_dataset(
            "tests/baselines/sina/sina_curve_rec_mimes_and_curves.json",
            match_attributes=[
                "param1",
            ])
        self.assertEqual(len(tuple(store.find())), 1)
        d1 = store.open("obj1")
        self.assertTrue(numpy.allclose(
            d1.get("timeplot_1/feature_b"), [10, 20, 30.3]))
        with self.assertRaises(RuntimeError):
            store.import_dataset(
                "tests/baselines/sina/sina_curve_rec_mimes_and_curves_2.json",
                merge_handler="conservative",
                match_attributes=[
                    "param1",
                ])
        self.assertEqual(len(tuple(store.find())), 1)
        self.assertTrue(numpy.allclose(
            d1.get("timeplot_1/feature_b"), [10, 20, 30.3]))
        store.import_dataset(
            "tests/baselines/sina/sina_curve_rec_mimes_and_curves_2.json",
            merge_handler="preserve",
            match_attributes=[
                "param1",
            ])
        self.assertEqual(len(tuple(store.find())), 1)
        self.assertTrue(numpy.allclose(
            d1.get("timeplot_1/feature_b"), [10, 20, 30.3]))
        store.import_dataset(
            "tests/baselines/sina/sina_curve_rec_mimes_and_curves_2.json",
            merge_handler="overwrite",
            match_attributes=[
                "param1",
            ])
        self.assertEqual(len(tuple(store.find())), 1)
        self.assertTrue(numpy.allclose(
            d1.get("timeplot_1/feature_a"), [1, 2, 3]))
        store.close()
        os.remove(uri)

    def test_custom_handler(self):
        source_store, db_source = self.connect()
        target_store, db_target = self.connect()

        dataset2 = source_store.create(name="example")
        dataset2.bar = "foo"
        dataset2.foo = "bar2"
        dataset2.foosome = "foo1"

        dataset3 = target_store.create(name="example")
        dataset3.bar = "foo"
        dataset3.foo = "bar3"
        dataset3.foosome = "foo2"

        # Now custom merge handler
        target_store.import_dataset(dataset2, match_attributes=["bar", "name"],
                                    merge_handler=my_handler,
                                    merge_handler_kargs={"overwrite_attributes": ["foo", ]})
        target_ds = tuple(target_store.find())
        self.assertEqual(len(target_ds), 1)
        self.assertEqual(target_ds[0].foo, "bar2")
        self.assertEqual(target_ds[0].foosome, "foo2")

        source_store.close()
        target_store.close()
        os.remove(db_source)
        os.remove(db_target)

    def test_skip_section(self):
        store, db_source = self.connect()
        store.import_dataset(
            "tests/baselines/sina/sina_curve_rec_mimes_and_curves_2.json",
            skip_sina_record_sections=["curve_sets", ])
        ds = next(store.find())
        self.assertEqual(ds.list_features(),
                         ['cycles',
                          'direction',
                          'elements',
                          'node',
                          'node/metrics_0',
                          'node/metrics_1',
                          'node/metrics_10',
                          'node/metrics_11',
                          'node/metrics_12',
                          'node/metrics_2',
                          'node/metrics_3',
                          'node/metrics_4',
                          'node/metrics_5',
                          'node/metrics_6',
                          'node/metrics_7',
                          'node/metrics_8',
                          'node/metrics_9',
                          'zone',
                          'zone/metrics_0',
                          'zone/metrics_1',
                          'zone/metrics_2',
                          'zone/metrics_3',
                          'zone/metrics_4'])
        store.close()
        os.remove(db_source)

    def test_associated_import(self):
        source_store, db_source = self.connect()
        target_store, db_target = self.connect()

        # Now custom dataset
        dataset = source_store.create(name="example")
        dataset.bar = "foo"
        dataset.foo = "bar2"
        dataset.associate("setup.py", "py")

        dataset_t = target_store.create(name="example")
        dataset_t.bar = "foo"
        dataset_t.foosome = "foo2"

        target_store.import_dataset(dataset, match_attributes=["bar", "name"])
        self.assertEqual(dataset_t.bar, "foo")
        self.assertEqual(dataset_t.foo, "bar2")
        self.assertEqual(dataset_t.foosome, "foo2")
        self.assertEqual(len(dataset_t._associated_data_), 1)

        source_store.close()
        target_store.close()
        os.remove(db_source)
        os.remove(db_target)


def my_handler(store_dataset, imported_dataset_dict,
               section, overwrite_attributes=[], **kargs):
    # prepare the target dict
    imported_attributes = imported_dataset_dict
    target_attributes = {}
    if section == "data":
        store_attributes = store_dataset.list_attributes(dictionary=True)
        target_attributes.update(imported_attributes)
        target_attributes.update(store_attributes)
        for attribute, value in imported_attributes.items():
            if attribute in store_attributes:
                if attribute in overwrite_attributes:
                    target_attributes[attribute] = value
    return target_attributes


if __name__ == "__main__":
    A = KoshTestImportExport()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

from __future__ import print_function
import os
from koshbase import KoshTest
import kosh
from sina.utils import DataRange
import time


class KoshTestDataset(KoshTest):
    def test_add_dataset(self):
        store, kosh_db = self.connect()
        # Check it's empy
        self.assertEqual(len(store.search()), 0)
        # Create dataset
        ds = store.create()
        # Check it's in db
        all_ds = store.search()
        self.assertEqual(len(all_ds), 1)
        self.assertEqual(ds.listattributes(), ["creator", "name"])
        # check error on non-existing attribute
        with self.assertRaises(AttributeError):
            print(ds.person)
        # Create an attribute
        ds.person = "Charles"
        self.assertEqual(ds.listattributes(), ["creator", "name", "person"])
        self.assertEqual(ds.person, "Charles")
        # modify attribute
        ds.person = "Charles Doutriaux"
        self.assertEqual(ds.listattributes(), ["creator", "name", "person"])
        self.assertEqual(ds.person, "Charles Doutriaux")
        # delete attribute
        del(ds.person)
        self.assertEqual(ds.listattributes(), ["creator", "name"])
        with self.assertRaises(AttributeError):
            print(ds.person)
        # Protected Attributes
        self.assertEqual(ds.__type__, store._dataset_record_type)
        # Make sure you can't change it
        ds.__type__ = "another_type"
        self.assertEqual(ds.__type__, store._dataset_record_type)
        # Make sure you cannot delete it
        del(ds.__type__)
        self.assertEqual(ds.__type__, store._dataset_record_type)
        printTestResults = """\
KOSH DATASET
        id: {id}
        name:Unnamed Dataset
        creator: {creator}

--- Attributes ---
        creator: {creator}
        name: Unnamed Dataset
--- Associated Data (0)---
""".format(id=ds.__id__, creator=ds.creator)
        print(ds)
        self.assertEqual(str(ds).replace("\t", "        "), printTestResults)
        # Set/update many attributes at once
        ds.update({"creator": "a new creator!",
                   "some_new_attribute": "a new one",
                   "some_int_attribute": 5})
        # check they are all here
        self.assertEqual(
            ds.listattributes(), [
                "creator", "name", "some_int_attribute", "some_new_attribute"])
        # Check they are correctly added with correct value
        self.assertEqual(ds.some_new_attribute, "a new one")
        self.assertEqual(ds.some_int_attribute, 5)
        # Check the pre-existing one was updated
        self.assertEqual(ds.creator, "a new creator!")
        os.remove(kosh_db)

    def test_search_datasets_in_store(self):
        store, kosh_db = self.connect()
        # Create many datasets
        store.create(metadata={"key1": 1, "key2": "A"})
        store.create(metadata={"key1": 2, "key2": "B"})
        store.create(metadata={"key1": 3, "key3": "c"})
        store.create(metadata={"key1": 4, "key3": "d", "key2": "D"})
        all_ds = store.search()
        self.assertEqual(len(all_ds), 4)
        with self.assertRaises(NotImplementedError):
            self.assertEqual(len(store.search("key1")), 4)
        with self.assertRaises(NotImplementedError):
            self.assertEqual(len(store.search("key2")), 3)
        with self.assertRaises(NotImplementedError):
            self.assertEqual(len(store.search("key3")), 2)
        # Remove this when above passes outside of exceptions
        from sina.utils import DataRange
        self.assertEqual(len(store.search(key1=DataRange(min=-1.e40))), 4)
        self.assertEqual(len(store.search(key2=DataRange(min=""))), 3)
        self.assertEqual(len(store.search(key3=DataRange(min=""))), 2)
        k1 = store.search(key1=2)
        self.assertEqual(len(k1), 1)
        self.assertEqual(k1[0].key1, 2)
        all_ds = store.search()
        self.assertEqual(len(all_ds), 4)
        os.remove(kosh_db)

    def test_associate(self):
        store, kosh_db = self.connect()
        # Create many datasets
        ds = store.create(metadata={"key1": 1, "key2": "A"})
        self.assertEqual(len(ds.search()), 0)
        ds.associate(
            "tests/baselines/node_extracts2",
            "something",
            absolute_path=False)
        self.assertEqual(len(ds.search()), 1)
        # Make sure associating again will not create additional data
        ds.associate(
            "tests/baselines/node_extracts2",
            "something",
            absolute_path=False)
        self.assertEqual(len(ds.search()), 1)
        # adding again does not create additional entry
        with self.assertRaises(ValueError):
            ds.associate(
                "tests/baselines/node_extracts2",
                "something_else",
                absolute_path=False)
        self.assertEqual(len(ds.search()), 1)
        f = ds.associate(
            "tests/baselines/node_extracts2/node_extracts2.hdf5",
            "hdf5",
            id_only=False)
        self.assertTrue(isinstance(f, kosh.sina.core.KoshSinaObject))
        self.assertEqual(len(ds.search()), 2)
        self.assertEqual(len(ds.search(mime_type="hdf5")), 1)
        self.assertEqual(len(ds.search(mime_type="something")), 1)
        self.assertEqual(len(ds.search(mime_type="somemimetype")), 0)
        ds.dissociate("tests/baselines/node_extracts2", absolute_path=False)
        self.assertEqual(len(ds._associated_data_), 1)
        # Now mutliple datasets at once
        ds = store.create()
        ds.associate([str(i) for i in range(200)],
                     metadata=[{"name": str(i)} for i in range(200)],
                     mime_type=["type_{}".format(i) for i in range(200)])
        self.assertEqual(len(ds._associated_data_), 200)
        self.assertEqual(len(ds.search(mime_type="type_12")), 1)
        self.assertEqual(len(ds.search(name="13")), 1)

        # Ok list completion tests
        ds = store.create()
        ds.associate([str(i) for i in range(200)],
                     metadata=[{"name": str(i)} for i in range(200)],
                     mime_type="a_mime_type")
        self.assertEqual(len(ds._associated_data_), 200)
        self.assertEqual(len(ds.search(mime_type="a_mime_type")), 200)

        ds = store.create()
        ds.associate([str(i) for i in range(200)], metadata={
                     "name": "my name"}, mime_type="stuff")
        self.assertEqual(len(ds._associated_data_), 200)
        self.assertEqual(len(ds.search(name="my name")), 200)

        os.remove(kosh_db)

    def test_search(self):
        store, kosh_db = self.connect()
        # Create many datasets
        ds = store.create(metadata={"key1": 1, "key2": "A"})
        ds2 = store.create(metadata={"key2": "B", "key3": 3})
        ds3 = store.create()
        store.create(metadata={"key2": "C", "key3": 4})
        ds.associate("tests/baselines/node_extracts2", "something")
        ds2.associate("tests/baselines/node_extracts2", "something")
        ds3.associate("tests/baselines/node_extracts2", "something")

        s = store.search(key2=DataRange("A"))
        self.assertEqual(len(s), 3)

        s = store.search(
            key2=DataRange("A"),
            file=os.path.abspath("tests/baselines/node_extracts2"))
        self.assertEqual(len(s), 2)

        self.assertEqual(len(ds._associated_data_), 1)
        ds2.dissociate("tests/baselines/node_extracts2")
        self.assertEqual(len(ds2._associated_data_), 0)
        s = store.search(
            key2=DataRange("A"),
            file=os.path.abspath("tests/baselines/node_extracts2"))
        self.assertEqual(len(s), 1)
        os.remove(kosh_db)

    def test_delete_dataset(self):
        store, kosh_db = self.connect()
        # Create many datasets
        ds = store.create(metadata={"key1": 1, "key2": "A", "project": "test"})
        ds2 = store.create(
            metadata={
                "key2": "B",
                "key3": 2,
                "project": "test"})
        ds3 = store.create(
            metadata={
                "key2": "c",
                "key3": 3,
                "project": "test"})
        ds4 = store.create(
            metadata={
                "key2": "D",
                "key3": 4,
                "project": "test"})
        ds.associate("setup.py", "ascii")
        ds2.associate("tests/baselines/images/LLNLiconWHITE.png", "png")
        ds3.associate(
            "tests/baselines/node_extracts2/node_extracts2.hdf5",
            "hdf5")
        ds4.associate("tests/baselines/node_extracts2", "something")
        ds_associated = ds._associated_data_[0]
        _ = store.open(ds_associated)
        ds.dissociate("setup.py")
        with self.assertRaises(Exception):
            _ = store.open(ds_associated)
        self.assertEqual(len(store.search(project="test")), 4)
        self.assertEqual(len(store.search()), 4)
        store2, kosh_db = self.connect(db_uri=kosh_db)
        self.assertEqual(len(store2.search()), 4)
        store.delete(ds.__id__)
        self.assertEqual(len(store.search(project="test")), 3)
        self.assertEqual(len(store.search()), 3)
        self.assertEqual(len(store2.search()), 3)
        store2, kosh_db = self.connect(db_uri=kosh_db)
        self.assertEqual(len(store2.search()), 3)
        # 04b6d302f33d00a5701a42b333c845832a5e6d65
        # sina 8c1b2cc21dc84ad32a6ff03a742ecef70ab89551
        ds_associated = ds2._associated_data_[0]
        _ = store.open(ds_associated)
        store.delete(ds2.__id__)
        self.assertEqual(len(store.search(project="test")), 2)
        with self.assertRaises(Exception):
            _ = store.open(ds_associated)
        os.remove(kosh_db)

    def test_use_cache(self):
        store, db_uri = self.connect()

        ds = store.create()

        class SomeLoader(kosh.KoshLoader):
            types = {"sometype": ["numpy", ]}

            def list_features(self):
                time.sleep(1)
                return ["a feature", ]

        store.add_loader(SomeLoader)

        ds.associate("fake_file", "sometype", absolute_path=False)

        start = time.time()
        features = ds.list_features()
        self.assertEqual(len(features), 1)
        end = time.time()
        self.assertGreaterEqual(end - start, 1.)

        # Now let's run it again and make sure cache is used
        start = time.time()
        features = ds.list_features()
        end = time.time()
        self.assertLess(end - start, 1.)

        # Ok this time let's use skip cache
        start = time.time()
        features = ds.list_features(use_cache=False)
        end = time.time()
        self.assertGreaterEqual(end - start, 1.)

        # let's associate something to ensure cache is reset
        ds.associate("fake_file_2", "sometype")
        start = time.time()
        features = ds.list_features()
        end = time.time()
        self.assertEqual(len(features), 2)
        self.assertGreaterEqual(end - start, 2.)

        # Now let's run it again and make sure cache is used
        start = time.time()
        features = ds.list_features()
        end = time.time()
        self.assertLess(end - start, 1.)

        # Ok this time let's dissociate to reset cache
        ds.dissociate("fake_file", absolute_path=False)

        start = time.time()
        features = ds.list_features()
        end = time.time()
        self.assertEqual(len(features), 1)
        self.assertGreaterEqual(end - start, 1.)

        os.remove(db_uri)

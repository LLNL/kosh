from __future__ import print_function
import os
from koshbase import KoshTest
import kosh
from sina.utils import DataRange
import time
import sina


class KoshTestDataset(KoshTest):
    def test_getitem_dataset(self):
        store, kosh_db = self.connect()
        ds = store.create()
        ds.associate(
            "tests/baselines/node_extracts2/node_extracts2.hdf5",
            "hdf5")

        ds["cycles"]
        with self.assertRaises(ValueError):
            ds["some_key_not_in_file"]
        os.remove(kosh_db)

    def test_associate_known_mime_no_file(self):
        store, kosh_db = self.connect()
        ds = store.create()
        ds.associate(
            "tests/baselines/I_dont_exists.hdf5",
            "hdf5",
            metadata={
                "bad": True})
        ds.associate(
            "tests/baselines/node_extracts2/node_extracts2.hdf5",
            "hdf5", metadata={"bad": False})
        features = ds.list_features()
        self.assertEqual(features, ['cycles', 'direction', 'elements',
                                    'node', 'node/metrics_0', 'node/metrics_1', 'node/metrics_10',
                                    'node/metrics_11', 'node/metrics_12', 'node/metrics_2',
                                    'node/metrics_3',
                                    'node/metrics_4', 'node/metrics_5',
                                    'node/metrics_6', 'node/metrics_7',
                                    'node/metrics_8', 'node/metrics_9',
                                    'zone', 'zone/metrics_0',
                                    'zone/metrics_1', 'zone/metrics_2',
                                    'zone/metrics_3', 'zone/metrics_4'])
        search = ds.find(bad=True, ids_only=True)
        self.assertEqual(len(list(search)), 1)
        search = ds.find(bad=False, ids_only=True)
        self.assertEqual(len(list(search)), 1)

    def test_add_dataset(self):
        store, kosh_db = self.connect()
        # Check the store is empty
        self.assertEqual(len(list(store.find())), 0)
        # Create a dataset
        ds = store.create()
        # Check datset was created
        all_ds = list(store.find())
        self.assertEqual(len(all_ds), 1)
        self.assertEqual(ds.listattributes(), ["creator", "id", "name"])
        # check error on non-existing attribute
        with self.assertRaises(AttributeError):
            print(ds.person)
        # Create an attribute
        ds.person = "Charles"
        self.assertEqual(
            ds.listattributes(), [
                "creator", "id", "name", "person"])
        self.assertEqual(ds.person, "Charles")
        # modify attribute
        ds.person = "Charles Doutriaux"
        self.assertEqual(
            ds.listattributes(), [
                "creator", "id", "name", "person"])
        self.assertEqual(ds.person, "Charles Doutriaux")
        # delete attribute
        del(ds.person)
        self.assertEqual(ds.listattributes(), ["creator", "id", "name"])
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
        name: Unnamed Dataset
        creator: {creator}

--- Attributes ---
        creator: {creator}
        name: Unnamed Dataset
--- Associated Data (0)---
--- Ensembles (0)---
        []
""".format(id=ds.id, creator=ds.creator)
        print(str(ds).replace("\t", "        "))
        self.assertEqual(str(ds).replace("\t", "        ").strip(), printTestResults.strip())
        # Set/update many attributes at once
        ds.update({"creator": "a new creator!",
                   "some_new_attribute": "a new one",
                   "some_int_attribute": 5})
        # Check they are all here
        self.assertEqual(
            ds.listattributes(), [
                "creator", "id", "name", "some_int_attribute", "some_new_attribute"])
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
        all_ds = list(store.find())
        self.assertEqual(len(all_ds), 4)
        self.assertEqual(len(list(store.find("key1"))), 4)
        self.assertEqual(len(list(store.find("key2"))), 3)
        self.assertEqual(len(list(store.find("key3"))), 2)
        self.assertEqual(len(list(store.find(key1=DataRange(min=-1.e40)))), 4)
        self.assertEqual(len(list(store.find(key2=DataRange(min="")))), 3)
        self.assertEqual(len(list(store.find(key3=DataRange(min="")))), 2)
        k1 = list(store.find(key1=2))
        self.assertEqual(len(k1), 1)
        self.assertEqual(k1[0].key1, 2)
        all_ds = list(store.find())
        self.assertEqual(len(all_ds), 4)
        os.remove(kosh_db)

    def test_associate(self):
        store, kosh_db = self.connect()
        # Create a dataset
        ds = store.create(metadata={"key1": 1, "key2": "A"})
        self.assertEqual(len(list(ds.find())), 0)
        ds.associate(
            "tests/baselines/node_extracts2",
            "something",
            absolute_path=False)
        self.assertEqual(len(list(ds.find())), 1)
        # Make sure associating again will not create additional data
        ds.associate(
            "tests/baselines/node_extracts2",
            "something",
            absolute_path=False)
        self.assertEqual(len(list(ds.find())), 1)
        # Adding again does not create additional entry
        with self.assertRaises(TypeError):
            ds.associate(
                "tests/baselines/node_extracts2",
                "something_else",
                absolute_path=False)
        self.assertEqual(len(list(ds.find())), 1)
        # Associating with another dataset does not create another object in the db
        n_files = len(
            list(
                store.find(
                    types=store._sources_type,
                    ids_only=True)))
        ds_2 = store.create("multi")
        ds_2.associate(
            "tests/baselines/node_extracts2",
            "something",
            absolute_path=False)
        n_files_2 = len(
            list(
                store.find(
                    types=[
                        store._sources_type,
                    ],
                    ids_only=True)))
        self.assertEqual(n_files, n_files_2)

        f = ds.associate(
            "tests/baselines/node_extracts2/node_extracts2.hdf5",
            "hdf5",
            id_only=False)
        self.assertTrue(isinstance(f, kosh.core_sina.KoshSinaObject))
        self.assertEqual(len(list(ds.find())), 2)
        self.assertEqual(len(list(ds.find(mime_type="hdf5"))), 1)
        # test passing a data dictionary (Sina style)
        self.assertEqual(len(list(ds.find(data={'mime_type': "hdf5"}))), 1)
        self.assertEqual(len(list(ds.find(mime_type="something"))), 1)
        self.assertEqual(len(list(ds.find(mime_type="somemimetype"))), 0)
        ds.dissociate("tests/baselines/node_extracts2", absolute_path=False)
        self.assertEqual(len(ds._associated_data_), 1)
        # Now mutliple datasets at once
        ds = store.create()
        ds.associate([str(i) for i in range(200)],
                     metadata=[{"name": str(i)} for i in range(200)],
                     mime_type=["type_{}".format(i) for i in range(200)])
        self.assertEqual(len(ds._associated_data_), 200)
        self.assertEqual(len(list(ds.find(mime_type="type_12"))), 1)
        self.assertEqual(len(list(ds.find(name="13"))), 1)
        self.assertEqual(len(list(ds.find("name"))), 200)

        # List completion tests
        ds = store.create()
        ds.associate([str(i + 300) for i in range(200)],
                     metadata=[{"name": str(i)} for i in range(200)],
                     mime_type="a_mime_type")
        self.assertEqual(len(ds._associated_data_), 200)
        self.assertEqual(len(list(ds.find(mime_type="a_mime_type"))), 200)

        ds = store.create()
        ds.associate([str(i + 600) for i in range(200)], metadata={
                     "name": "my name"}, mime_type="stuff")
        self.assertEqual(len(ds._associated_data_), 200)
        self.assertEqual(len(list(ds.find(name="my name"))), 200)

        # Make sure you cannot assocaate with different types
        ds.associate("some_uri", "some_mime_type")
        with self.assertRaises(TypeError):
            ds.associate("some_uri", "some_other_mime_type")
        with self.assertRaises(TypeError):
            ds_2.associate("some_uri", "some_other_mime_type")

        # Make sure dissociate fully removes dataset from the store
        n_files = len(
            list(
                store.find(
                    types=store._sources_type,
                    ids_only=True)))
        ds.dissociate("some_uri")  # shouldn't be anywhere now
        n_files_2 = len(
            list(
                store.find(
                    types=store._sources_type,
                    ids_only=True)))
        self.assertEqual(n_files - 1, n_files_2)
        os.remove(kosh_db)

    def test_find(self):
        store, kosh_db = self.connect()
        # Create many datasets
        ds = store.create(metadata={"key1": 1, "key2": "A"})
        ds2 = store.create(metadata={"key2": "B", "key3": 3})
        ds3 = store.create()
        store.create(metadata={"key2": "C", "key3": 4})
        ds.associate("tests/baselines/node_extracts2", "something")
        ds2.associate("tests/baselines/node_extracts2", "something")
        ds3.associate("tests/baselines/node_extracts2", "something")

        s = list(store.find(key2=DataRange("A")))
        self.assertEqual(len(s), 3)

        s = list(store.find(
            key2=DataRange("A"),
            file=os.path.abspath("tests/baselines/node_extracts2")))
        self.assertEqual(len(s), 2)

        self.assertEqual(len(ds._associated_data_), 1)
        ds2.dissociate("tests/baselines/node_extracts2")
        self.assertEqual(len(ds2._associated_data_), 0)
        s = list(store.find(
            key2=DataRange("A"),
            file=os.path.abspath("tests/baselines/node_extracts2")))
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
        self.assertEqual(len(list(store.find(project="test"))), 4)
        self.assertEqual(len(list(store.find())), 4)
        store2, kosh_db = self.connect(db_uri=kosh_db)
        self.assertEqual(len(list(store2.find())), 4)
        store.delete(ds.id)
        self.assertEqual(len(list(store.find(project="test"))), 3)
        self.assertEqual(len(list(store.find())), 3)
        self.assertEqual(len(list(store2.find())), 3)
        store2, kosh_db = self.connect(db_uri=kosh_db)
        self.assertEqual(len(list(store2.find())), 3)
        ds_associated = ds2._associated_data_[0]
        _ = store.open(ds_associated)
        store.delete(ds2.id)
        self.assertEqual(len(list(store.find(project="test"))), 2)
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

        os.remove(db_uri)

    def test_list_features(self):
        store, db_uri = self.connect()
        ds = store.create()
        ds.associate(
            "tests/baselines/node_extracts2/node_extracts2.hdf5", "hdf5")
        ds.associate(
            "tests/baselines/images/LLNLiconWHITE.png", "png")

        self.assertEqual(sorted(ds.list_features()), ['cycles',
                                                      'direction',
                                                      'elements',
                                                      'image',
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

        self.assertEqual(sorted(ds.list_features(next(ds.find(mime_type="hdf5", ids_only=True)))),
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
        self.assertEqual(sorted(ds.list_features(next(ds.find(mime_type="png", ids_only=True)))),
                         ["image", ])
        os.remove(db_uri)

    def test_get_sina_objects(self):
        store, db_uri = self.connect()
        self.assertIsInstance(store.get_sina_store(), sina.datastore.DataStore)
        self.assertIsInstance(
            store.get_sina_records(),
            sina.datastore.DataStore.RecordOperations)
        ds = store.create()
        self.assertIsInstance(ds.get_sina_store(), sina.datastore.DataStore)
        self.assertIsInstance(
            ds.get_sina_records(),
            sina.datastore.DataStore.RecordOperations)

        os.remove(db_uri)


if __name__ == "__main__":
    A = KoshTestDataset()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

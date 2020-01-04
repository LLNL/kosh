import os
from koshbase import KoshTest
import kosh


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
        with self.assertRaises(AttributeError) as err:
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
        with self.assertRaises(AttributeError) as err:
            print(ds.person)
        # Protected Attributes
        self.assertEqual(ds.__type__, "dataset")
        # Make sure you can't change it
        ds.__type__ = "another_type"
        self.assertEqual(ds.__type__, "dataset")
        # Make sure you cannot delete it
        del(ds.__type__)
        self.assertEqual(ds.__type__, "dataset")
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
        self.assertEqual(str(ds).replace("\t","        "), printTestResults)
        store.sync()
        del(store)
        os.remove(kosh_db)

    def test_search_datasets_in_store(self):
        store, kosh_db = self.connect()
        # Create many datasets
        ds = store.create(metadata={"key1": 1, "key2": "A"})
        ds = store.create(metadata={"key1": 2, "key2": "B"})
        ds = store.create(metadata={"key1": 3, "key3": "c"})
        ds = store.create(metadata={"key1": 4, "key3": "d", "key2": "D"})
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
        store.sync()
        del(store)
        os.remove(kosh_db)

    def test_add_file(self):
        store, kosh_db = self.connect()
        # Create many datasets
        ds = store.create(metadata={"key1": 1, "key2": "A"})
        self.assertEqual(len(ds.search()), 0)
        ds.add_file("tests/baselines/mash/node_extracts2", "mash")
        self.assertEqual(len(ds.search()), 1)
        # adding again does not create additional entry
        with self.assertRaises(ValueError):
            ds.add_file("tests/baselines/mash/node_extracts2", "mash")
        self.assertEqual(len(ds.search()), 1)
        f = ds.add_file("tests/baselines/mash/node_extracts2/node_extracts2.hdf5", "hdf5")
        self.assertTrue(isinstance(f, kosh.sina.core.KoshSinaObject))
        self.assertEqual(len(ds.search()), 2)
        self.assertEqual(len(ds.search(mime_type="hdf5")), 1)
        self.assertEqual(len(ds.search(mime_type="mash")), 1)
        self.assertEqual(len(ds.search(mime_type="somemimetype")), 0)
        store.sync()
        del(store)
        os.remove(kosh_db)

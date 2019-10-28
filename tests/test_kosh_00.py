import unittest
import os
import sys
import shlex
from subprocess import PIPE, Popen
from kosh import KoshStore
import uuid
import logging


## Turn off sina logging
for name in ["sina.datastores.sql", "sina.model", "sina.utils"]:
    logger = logging.getLogger(name)
    logger.setLevel(logging.ERROR)


class KoshTest(unittest.TestCase):
    def init_db(self, engine=None):
        if engine is None:
            engine = os.environ.get("KOSH_ENGINE", "sina")
        if engine == "sina":
            # Make sure local file is new sql file
            kosh_test_sql_file = "kosh_test_{}.sql".format(uuid.uuid1().hex)
            if os.path.exists(kosh_test_sql_file):
                os.remove(kosh_test_sql_file)

            cmd = "{}/bin/python scripts/init_sina.py --sina_db={}".format(
                sys.prefix, kosh_test_sql_file)
            p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
            o, e = p.communicate()
            return kosh_test_sql_file

    def connect(self, engine=None):
        if engine is None:
            engine = os.environ.get("KOSH_ENGINE", "sina")
        kosh_db = self.init_db(engine)
        if engine == "sina":
            # os.getlogin does not work on my WSL
            store = KoshStore(engine="sina", username=os.environ["USER"], sql='sql',
                      db_path=kosh_db)
        return store, kosh_db
    def test_connect(self):
        store, kosh_test_sql_file = self.connect()
        os.remove(kosh_test_sql_file)

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
        os.remove(kosh_db)

    def test_search_datasets(self):
        store, kosh_db = self.connect()
        # Create many datasets
        ds = store.create(metadata={"key1": 1, "key2": "A"})
        ds = store.create(metadata={"key1": 2, "key2": "B"})
        ds = store.create(metadata={"key1": 3, "key3": "c"})
        ds = store.create(metadata={"key1": 4, "key3": "d", "key2": "D"})
        """
        all_ds = store.search()
        self.assertEqual(len(all_ds), 4)
        with self.assertRaises(NotImplementedError):
            self.assertEqual(len(store.search("key1")), 4)
        with self.assertRaises(NotImplementedError):
            self.assertEqual(len(store.search("key2")), 3)
        with self.assertRaises(NotImplementedError):
            self.assertEqual(len(store.search("key3")), 2)
        # Remove this when above passes
        """
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

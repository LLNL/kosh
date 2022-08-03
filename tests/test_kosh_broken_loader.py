import kosh
from koshbase import KoshTest
import os


class MyLoader(kosh.KoshLoader):
    types = {"my": "numpy"}

    def extract(self):
        return 1

    def list_features(self):
        import some_module_not_in_our_python  # noqa
        return ["blah", ]


class MyLoader2(kosh.KoshLoader):
    types = {"my": "numpy"}

    def extract(self):
        return 1

    def list_features(self):
        return ["blah", ]


class MyLoader3(kosh.KoshLoader):
    types = {"my": "numpy"}

    def extract(self):
        import some_module_not_in_our_python  # noqa
        return 1

    def list_features(self):
        return ["blah", ]


class BrokenLoader(KoshTest):
    def test_broken_in_list_features(self):
        store, db_uri = self.connect()
        store.add_loader(MyLoader)
        ds = store.create()
        ds.associate("setup.py", "my")

        self.assertEqual(ds.list_features(), [])
        with self.assertRaises(ValueError):
            print(ds["blah"][:])
        store.close()
        os.remove(db_uri)

    def test_fallback(self):
        store, db_uri = self.connect()
        store.add_loader(MyLoader)
        store.add_loader(MyLoader2)
        ds = store.create()
        ds.associate("setup.py", "my")

        self.assertEqual(ds.list_features(), ["blah", ])
        self.assertEqual(ds["blah"][:], 1)
        store.close()
        os.remove(db_uri)

    def test_broken_in_get(self):
        store, db_uri = self.connect()
        store.add_loader(MyLoader3)
        ds = store.create()
        ds.associate("setup.py", "my")

        self.assertEqual(ds.list_features(), ["blah", ])
        with self.assertRaises(ImportError):
            ds["blah"][:]
        store.close()
        os.remove(db_uri)

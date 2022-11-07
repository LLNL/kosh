import kosh
import types
import numpy
from koshbase import KoshTest


@kosh.operators.numpy_operator
def my_operator(*data, **kargs):
    return numpy.concatenate(*data)


@kosh.transformers.numpy_transformer
def my_transformer(data):
    return data


class FakeLoader(kosh.KoshLoader):
    types = {"fake": [int, ]}

    def extract(self):
        return 2

    def list_features(self, *args, **kargs):
        return ["fake"]


class KoshTestDescribes(KoshTest):
    def test_describe_entry_features(self):
        store, _ = self.connect()
        ds = store.create()
        ds.associate(
            "tests/baselines/node_extracts2/node_extracts2.hdf5",
            "hdf5")
        c = ds["cycles"]
        entries = c.describe_entries()
        self.assertTrue(isinstance(entries, types.GeneratorType))
        entries = list(entries)
        self.assertEqual(len(entries), 1)
        info = entries[0]
        self.assertTrue(isinstance(info, dict))
        self.cleanup_store(store)

    def test_describe_entry_features_transformer(self):
        store, _ = self.connect()
        ds = store.create()
        ds.associate(
            "tests/baselines/node_extracts2/node_extracts2.hdf5",
            "hdf5")
        c = ds.get_execution_graph("cycles", transformers=my_transformer)
        entries = c.describe_entries()
        self.assertTrue(isinstance(entries, types.GeneratorType))
        entries = list(entries)
        self.assertEqual(len(entries), 1)
        info = entries[0]
        self.assertTrue(isinstance(info, dict))
        self.cleanup_store(store)

    def test_describe_entry_features_operator(self):
        store, _ = self.connect()
        ds = store.create()
        ds.associate(
            "tests/baselines/node_extracts2/node_extracts2.hdf5",
            "hdf5")
        double = my_operator(ds["cycles"], ds["cycles"])
        entries = double.describe_entries()
        self.assertTrue(isinstance(entries, types.GeneratorType))
        entries = list(entries)
        self.assertEqual(len(entries), 2)
        for info in entries:
            self.assertTrue(isinstance(info, dict))
        self.cleanup_store(store)

    def test_describe_entry_features_operator_and_transformer(self):
        store, _ = self.connect()
        ds = store.create()
        ds.associate(
            "tests/baselines/node_extracts2/node_extracts2.hdf5",
            "hdf5")
        c = ds.get_execution_graph("cycles", transformers=my_transformer)
        double = my_operator(c, c)
        entries = double.describe_entries()
        self.assertTrue(isinstance(entries, types.GeneratorType))
        entries = list(entries)
        self.assertEqual(len(entries), 2)
        for info in entries:
            self.assertTrue(isinstance(info, dict))
        self.cleanup_store(store)

    def test_describe_entry_features_operators_and_transformers(self):
        store, _ = self.connect()
        ds = store.create()
        ds.associate(
            "tests/baselines/node_extracts2/node_extracts2.hdf5",
            "hdf5")
        c = ds.get_execution_graph("cycles", transformers=my_transformer)
        double = my_operator(c, c)
        triple = my_operator(c, c, c)
        combine = my_operator(double, triple)
        mixed = my_operator(combine, combine)
        entries = mixed.describe_entries()
        self.assertTrue(isinstance(entries, types.GeneratorType))
        entries = list(entries)
        self.assertEqual(len(entries), 10)
        for info in entries:
            self.assertTrue(isinstance(info, dict))
        self.cleanup_store(store)

    def test_no_describe_function(self):
        store, _ = self.connect()
        store.add_loader(FakeLoader)
        ds = store.create()
        ds.associate("fake", "fake")
        entries = ds["fake"].describe_entries()
        self.cleanup_store(store)
        entries = list(entries)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0], {})

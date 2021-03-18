import os
import numpy
import kosh
from koshbase import KoshTest
import collections


class StringsLoader(kosh.loaders.KoshLoader):
    types = {"ascii": ["numlist", "some_format", "Another_format"]}

    def extract(self):
        return [1, 2, 3, 4, 5, 6]

    def list_features(self):
        return ["numbers", ]


class MyT(kosh.transformers.KoshTransformer):
    types = collections.OrderedDict(
        [("numlist", ["numpy", ]), ("some_format", ["pandas", ])])

    def transform(self, input, format):
        return numpy.array(input)


class ADD(kosh.operators.KoshOperator):
    types = collections.OrderedDict(
        [("numpy", ["numpy", "pandas"]), ("pandas", ["numpy", "pandas"])])

    def operate(self, *inputs, **kargs):
        out = inputs[0]
        for input_ in inputs[1:]:
            out += input_
        return out


class KoshTestOperators(KoshTest):
    def test_no_good_output(self):
        store, db_uri = self.connect()
        store.add_loader(StringsLoader)

        ds = store.create()
        ds.associate("some_file.nb", mime_type="ascii")

        nb = ds["numbers"]

        with self.assertRaises(Exception):
            ADD(nb, nb)
        os.remove(db_uri)

    def test_simple_add(self):
        store, db_uri = self.connect()
        store.add_loader(StringsLoader)

        ds = store.create()
        ds.associate("some_file.nb", mime_type="ascii")

        nb = ds.get_execution_graph("numbers", transformers=[MyT(), ])

        # Now with the transformer we should be good
        A = ADD(nb, nb)

        print(A[:])
        self.assertEqual(numpy.allclose(
            A[:], numpy.array([2, 4, 6, 8, 10, 12])), 1)
        os.remove(db_uri)

    def test_nested_graphs(self):
        store, db_uri = self.connect()
        store.add_loader(StringsLoader)

        ds = store.create()
        ds.associate("some_file.nb", mime_type="ascii")

        nb = ds.get_execution_graph("numbers", transformers=[MyT(), ])

        # Now with the transformer we should be good
        A = ADD(nb, nb)
        A2 = ADD(A, nb)

        self.assertEqual(numpy.allclose(
            A2[:], numpy.array([3, 6, 9, 12, 15, 18])), 1)
        os.remove(db_uri)


if __name__ == "__main__":
    A = KoshTestOperators()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

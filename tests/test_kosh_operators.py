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

    def __init__(self, *args, **kargs):
        super(MyT, self).__init__(*args, **kargs)
        self.log_file = kargs.pop("log_file", "")
        if self._verbose:
            with open(self.log_file, "w") as f:
                f.write("Log from MyT\n")

    def transform(self, input, format):
        return numpy.array(input)

    def _print(self, message):
        with open(self.log_file, "r+") as f:
            f.seek(0, os.SEEK_END)
            f.write(message)


class ADD(kosh.operators.KoshOperator):
    types = collections.OrderedDict(
        [("numpy", ["numpy", "pandas"]), ("pandas", ["numpy", "pandas"])])

    def __init__(self, *args, **kargs):
        super(ADD, self).__init__(*args, **kargs)
        self.log_file = kargs.pop("log_file", "")
        if self._verbose and not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                f.write("Log from ADD\n")

    def operate(self, *inputs, **kargs):
        out = inputs[0]
        for input_ in inputs[1:]:
            out += input_
        return out

    def _print(self, message):
        with open(self.log_file, "r+") as f:
            f.seek(0, os.SEEK_END)
            f.write(message)


class KoshTestOperators(KoshTest):
    def test_no_good_output(self):
        store, db_uri = self.connect()
        store.add_loader(StringsLoader)

        ds = store.create()
        ds.associate("some_file.nb", mime_type="ascii")

        nb = ds["numbers"]

        with self.assertRaises(Exception):
            ADD(nb, nb)
        store.close()
        os.remove(db_uri)

    def test_simple_add(self):
        store, db_uri = self.connect()
        store.add_loader(StringsLoader)

        ds = store.create()
        ds.associate("some_file.nb", mime_type="ascii")

        nb = ds.get_execution_graph("numbers", transformers=[MyT(), ])

        # Now with the transformer we should be good
        # string are transformed to int
        A = ADD(nb, nb)

        self.assertEqual(numpy.allclose(
            A[:], numpy.array([2, 4, 6, 8, 10, 12])), 1)
        store.close()
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
        store.close()
        os.remove(db_uri)

    def test_operator_get_inputs(self):
        store, db_uri = self.connect()
        store.add_loader(StringsLoader)

        ds = store.create()
        ds.associate("some_file.nb", mime_type="ascii")
        ds2 = store.create()
        ds2.associate("some_file.nb", mime_type="ascii")

        nb = ds.get_execution_graph("numbers", transformers=[MyT(), ])
        nb2 = ds2.get_execution_graph("numbers", transformers=[MyT(), ])

        # Now with the transformer we should be good
        A = ADD(nb2, nb)
        A2 = ADD(A, nb, nb2)

        req = A2.get_input_datasets()
        self.assertEqual([x.id for x in req], [ds2.id, ds.id, ds.id, ds2.id])
        req = A2.get_input_loaders()
        self.assertEqual([x.feature for x in req], ["numbers", ]*4)
        store.close()
        os.remove(db_uri)

    def test_operator_and_transformers_verbose(self):
        store, db_uri = self.connect()
        store.add_loader(StringsLoader)

        ds = store.create()
        ds.associate("some_file.nb", mime_type="ascii")

        my_t_log = "myt_{}.txt".format(numpy.random.randint(10000000))
        nb = ds.get_execution_graph("numbers", transformers=[MyT(
            log_file=my_t_log, cache=True, verbose=True), ])

        # Now with the transformer we should be good
        add_log = "add_{}.txt".format(numpy.random.randint(10000000))
        A = ADD(nb, nb, verbose=True, log_file=add_log, cache=True)

        _ = A[:]
        self.assertEqual(numpy.allclose(
            A[:], numpy.array([2, 4, 6, 8, 10, 12])), 1)

        self.assertTrue(os.path.exists(my_t_log))
        self.assertTrue(os.path.exists(add_log))

        for filename in [my_t_log, add_log]:
            with open(filename) as f:
                self.assertTrue("Loaded results from cache file" in f.read())
            os.remove(filename)
        store.close()
        os.remove(db_uri)


if __name__ == "__main__":
    A = KoshTestOperators()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

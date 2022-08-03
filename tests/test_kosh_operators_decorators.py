from kosh import numpy_operator, typed_operator
from kosh.operators.core import typed_operator_with_kwargs
import koshbase
import kosh
import numpy
import os


class FakeLoader(kosh.loaders.KoshLoader):
    types = {"fake": ["numpy", ], "fakestr": ["str", ]}

    def list_features(self):
        return ["data", ]

    def extract(self):
        out = numpy.array([1, 2, 3, 2, 1])
        if self.format == "str":
            return repr(out)
        else:
            return out


@numpy_operator
def Add(*inputs, **kwargs):  # noqa
    out = inputs[0][:]
    for input_ in inputs[1:]:
        out += input_[:]
    return out


@typed_operator_with_kwargs({"str": ["str", "numpy"], "numpy": ["numpy", ]})
def Add_kw(*inputs, **kargs):  # noqa
    from numpy import array  # noqa
    out = eval(inputs[0])
    for input_ in inputs[1:]:
        out += eval(input_)
    if kargs["format"] == "str":
        return str(out)
    else:
        return out


@typed_operator({"str": ["str", ]})
def Add_str(*inputs):  # noqa
    from numpy import array  # noqa
    out = eval(inputs[0])
    for input_ in inputs[1:]:
        out += eval(input_)
    return str(out)


@typed_operator()
def Add_str_bad(*inputs):  # noqa
    from numpy import array  # noqa
    out = eval(inputs[0])
    for input_ in inputs[1:]:
        out += eval(input_)
    return str(out)


class TestKoshOperatorDecorators(koshbase.KoshTest):
    def testNumpyOperatorDecorator(self):
        store, db_uri = self.connect()

        store.add_loader(FakeLoader)
        ds = store.create(name="testit")
        ds.associate("fake.fake", "fake")
        data = ds["data"]
        self.assertTrue(
            numpy.allclose(Add(data, data)[:],
                           numpy.array([2, 4, 6., 4, 2])))
        store.close()
        os.remove(db_uri)

    def testTypedOperatorDecorator(self):
        store, db_uri = self.connect()

        store.add_loader(FakeLoader)
        ds = store.create(name="testit")
        ds.associate("fake.fake", "fakestr")
        data = ds["data"]
        self.assertEqual(Add_str(data, data)[:], str(
            numpy.array([2, 4, 6, 4, 2])))
        with self.assertRaises(RuntimeError):
            Add_str_bad(data, data)[:]
        store.close()
        os.remove(db_uri)

    def testTypedKwargOperatorDecorator(self):
        store, db_uri = self.connect()

        store.add_loader(FakeLoader)
        ds = store.create(name="testit")
        ds.associate("fake.fake", "fakestr")
        data = ds["data"]
        self.assertEqual(Add_kw(data, data)(format="str")[:], str(
            numpy.array([2, 4, 6, 4, 2])))
        self.assertTrue(numpy.allclose(Add_kw(data, data)(format="numpy")[:],
                                       numpy.array([2, 4, 6, 4, 2])))
        store.close()
        os.remove(db_uri)


if __name__ == "__main__":
    A = TestKoshOperatorDecorators()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

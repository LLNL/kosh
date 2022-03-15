from kosh import numpy_transformer, typed_transformer
from kosh.transformers.core import typed_transformer_with_format
import koshbase
import kosh
import numpy


class FakeLoader(kosh.loaders.KoshLoader):
    types = {"fake": ["numpy", ], "fakestr": ["str", ]}

    def list_features(self):
        return ["data", ]

    def extract(self):
        out = numpy.array([1, 2, 3, 2, 1], dtype=numpy.float)
        if self.format == "str":
            return repr(out)
        else:
            return out


@numpy_transformer
def normalize(input_):
    return (input_ - input_.min()) / (input_.max() - input_.min())


@typed_transformer({"str": ["str", ]})
def normalize_str(input_):
    from numpy import array  # noqa
    input_ = eval(input_)
    return str((input_ - input_.min()) / (input_.max() - input_.min()))


@typed_transformer_with_format({"str": ["str", "numpy"]})
def normalize_str_or_numpy(input_, format):
    from numpy import array  # noqa
    input_ = eval(input_)
    out = (input_ - input_.min()) / (input_.max() - input_.min())
    if format == "str":
        return str(out)
    else:
        return out


class TestKoshTransformers(koshbase.KoshTest):
    def testNumpyTransformerDecorator(self):
        store, db_uri = self.connect()

        store.add_loader(FakeLoader)
        ds = store.create(name="testit")
        ds.associate("fake.fake", "fake")
        self.assertTrue(
            numpy.allclose(
                ds.get(
                    "data", transformers=[
                        normalize, ]), [
                    0, 0.5, 1., .5, 0]))

    def testTypedTransformerDecorator(self):
        store, db_uri = self.connect()

        store.add_loader(FakeLoader)
        ds = store.create(name="testit")
        ds.associate("fake.fake", "fakestr")
        self.assertEqual(ds.get("data", transformers=[normalize_str, ]), str(
            numpy.array([0, 0.5, 1., .5, 0])))

    def testTypedTransformerWithFormatDecorator(self):
        store, db_uri = self.connect()

        store.add_loader(FakeLoader)
        ds = store.create(name="testit")
        ds.associate("fake.fake", "fakestr")
        data = ds.get("data", transformers=[normalize_str_or_numpy, ], format="str")
        self.assertEqual(data, str(numpy.array([0, 0.5, 1., .5, 0])))
        data = ds.get("data", transformers=[normalize_str_or_numpy, ], format="numpy")
        self.assertTrue(numpy.allclose(data, numpy.array([0, 0.5, 1., .5, 0])))


if __name__ == "__main__":
    A = TestKoshTransformers()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

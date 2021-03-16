from koshbase import KoshTest
import kosh
import os


class MyTrsf(kosh.transformers.KoshTransformer):
    types = {"numpy": ["numpy", ]}

    def transform(self, input, format):
        print("T1")
        return input


class MyTrsf2(kosh.transformers.KoshTransformer):
    types = {"numpy": ["numpy", ]}

    def transform(self, input, format):
        print("T2")
        return input


class MyTrsf3(kosh.transformers.KoshTransformer):
    types = {"numpy": ["test_stuff", ]}

    def transform(self, input, format):
        print("T3")
        top = self.parent
        print("TOP:", top)
        while hasattr(top, "parent"):
            top = top.parent
        return top._user_passed_parameters, top.feature, top.format, format


class KoshTestTransformerParent(KoshTest):
    def test_parent(self):
        store, uri = self.connect()
        ds = store.create()
        ds.associate(
            "/g/g19/cdoutrix/git/kosh/tests/baselines/node_extracts2/node_extracts2.hdf5",
            "hdf5")

        features = sorted(ds.list_features())

        data = ds.get(
            features[0],
            transformers=[
                MyTrsf(),
                MyTrsf2(),
                MyTrsf3()])
        print("DATA:", data)
        self.assertEqual(data[0], (None, {}))
        self.assertEqual(data[1], "cycles")
        self.assertEqual(data[2], "numpy")
        self.assertEqual(data[3], "numpy")
        os.remove(uri)


if __name__ == "__main__":
    A = KoshTestTransformerParent()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

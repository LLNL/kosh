from koshbase import KoshTest
import os
import kosh


class FakeLoader(kosh.loaders.KoshLoader):
    types = {"fake": ["numpy"]}

    def list_features(self):
        return ["fake"]

    def extract(self):
        args, kargs = self._user_passed_parameters
        return self.feature + "___" + self.obj.uri


class KoshTestManyFeaturesSameName(KoshTest):
    def test_many_features_same_name(self):
        store, kosh_db = self.connect()
        store.add_loader(FakeLoader)
        ds = store.create()

        for i in range(6):
            ds.associate(chr(i + 65), mime_type="fake", absolute_path=False)

        # Not necessarily coming back ordered
        feats = sorted(ds.list_features())
        self.assertEqual(len(feats), 6)

        for i, feat in enumerate(feats):
            self.assertEqual(feat, "fake_@_" + chr(i + 65))
            self.assertEqual(ds.get(feat), "fake___" + chr(i + 65))

        os.remove(kosh_db)

    def test_many_hdf5s(self):
        store, kosh_db = self.connect()
        store.add_loader(FakeLoader)
        ds = store.create()
        ds.associate(
            "examples/sample_files/run_001.hdf5",
            mime_type="hdf5",
            absolute_path=False)
        ds.associate(
            "examples/sample_files/run_002.hdf5",
            mime_type="hdf5",
            absolute_path=False)

        ds.get('cycles_@_examples/sample_files/run_001.hdf5')
        os.remove(kosh_db)


if __name__ == "__main__":
    A = KoshTestManyFeaturesSameName()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

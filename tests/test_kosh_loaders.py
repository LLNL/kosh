import os
from koshbase import KoshTest
import kosh
import h5py
import json
import random
import numpy
import sys


def unix_to_win_compatible(pth):
    """converts unix path to windows"""
    if sys.platform.startswith("win"):
        return pth.replace("/", "\\")
    else:
        return pth


class SecondHDF5Loader(kosh.loaders.HDF5Loader):
    types = {"hdf5": ["numpy", ]}

    def extract(self):
        if not isinstance(self.feature, list):
            features = [self.feature, ]
        else:
            features = self.feature

        out = []
        h5 = h5py.File(self.obj.uri, "r")
        for feature in features:
            out.append(h5[feature][:] * 2.)
        if isinstance(self.feature, str):
            out = out[0]
        h5.close()
        return out


class KoshTestLoaders(KoshTest):
    def test_load_jsons(self):
        store, kosh_db = self.connect()
        ds = store.create()
        name = "kosh_random_json_{}".format(random.randint(0, 23434434))
        with open(name, "w") as f:
            json.dump([1, 2, 3, 4], f)

        ds.associate(name, "json")
        lst = ds.get("content")
        self.assertEqual(lst, [1, 2, 3, 4])
        with open(name, "w") as f:
            json.dump("testme", f)
        st = ds.get("content")
        self.assertEqual(st, "testme")
        with open(name, "w") as f:
            json.dump({"A": "a", "B": "b", "C": "c"}, f)
        self.assertEqual(
            ds.list_features(
                use_cache=False), [
                "A", "B", "C", "content"])
        ct = ds.get("content")
        self.assertEqual(ct, {"A": "a", "B": "b", "C": "c"})
        ct = ds.get("A")
        self.assertEqual(ct, "a")
        ct = ds.get(["B", "A"])
        self.assertEqual(ct, ["b", "a"])
        ct = ds.get(["B", "A"], format="dict", group=True)
        self.assertEqual(ct, {"B": "b", "A": "a"})
        store.close()
        os.remove(name)
        os.remove(kosh_db)

    def test_loader(self):
        store, kosh_db = self.connect()
        ds = store.create(metadata={"key1": 1, "key2": "A"})
        ds.associate(
            "tests/baselines/node_extracts2/node_extracts2.hdf5", "hdf5")
        ld, _ = store._find_loader(ds._associated_data_[0])
        self.assertEqual(sorted(ld.known_types()), ["hdf5"])
        self.assertEqual(ld.known_load_formats("file"), [])
        store.close()
        os.remove(kosh_db)

    def test_generic_loader(self):
        store, kosh_db = self.connect()
        ds = store.create(metadata={"key1": 1, "key2": "A"})
        ds.associate("setup.py", "ascii")
        ld, _ = store._find_loader(ds._associated_data_[0])
        self.assertIsInstance(ld, kosh.loaders.core.KoshFileLoader)
        self.assertEqual(sorted(ld.known_types()), sorted(
            set(["file", store._sources_type])))
        self.assertEqual(ld.known_load_formats("file"), [])
        self.assertIsInstance(ds.get(None), list)
        store.close()
        os.remove(kosh_db)

    def test_images(self):
        store, kosh_db = self.connect()
        ds = store.create(metadata={"key1": 1, "key2": "A"})
        ds.associate(
            "tests/baselines/images/LLNLiconWHITE.png", "png")
        features = sorted(ds.list_features())
        self.assertEqual(features, ["image", ])
        ds.get("image")
        # Duplicate features names URI should be added
        ds.associate(
            "share/icons/png/Kosh_Logo_Blue.png", "png")
        features = sorted(ds.list_features(use_cache=False))[::-1]
        self.assertEqual(features, [unix_to_win_compatible(
            "image_@_{}/tests/baselines/images/LLNLiconWHITE.png".format(os.getcwd())),
            unix_to_win_compatible(
            "image_@_{}/share/icons/png/Kosh_Logo_Blue.png".format(os.getcwd()))])

        ds = store.create(metadata={"key1": 1, "key2": "A"})
        ds.associate("tests/baselines/images/buffalo.pgm", "pgm")
        img = ds.get("image")
        self.assertEqual(img.shape, (321, 481))

        ds = store.create(metadata={"key1": 1, "key2": "A"})
        ds.associate("tests/baselines/images/brain_398.ascii.pgm", "pgm")
        img = ds.get("image")
        self.assertEqual(img.shape, (486, 720))

        info = ds.describe_feature("image")

        self.assertEqual(sorted(info.keys()), ["format", "max_value", "size"])
        self.assertEqual(info["format"], "pgm (P2)")
        self.assertEqual(info["max_value"], 255)
        ds.associate("share/icons/png/Kosh_Logo_Blue.png", mime_type="png")
        self.assertEqual(sorted(ds.list_features(use_cache=False))[::-1],
                         [unix_to_win_compatible(
                             'image_@_{}/tests/baselines/images/brain_398.ascii.pgm'.format(os.getcwd())),
                          unix_to_win_compatible(
                              'image_@_{}/share/icons/png/Kosh_Logo_Blue.png'.format(
                                  os.getcwd()))])  # URI is now added to feature to disambiguate them
        info = ds.describe_feature(
            unix_to_win_compatible("image_@_{}/share/icons/png/Kosh_Logo_Blue.png".format(os.getcwd())))
        self.assertEqual(info["size"], (1035, 403))
        data = ds.get(
            unix_to_win_compatible("image_@_{}/share/icons/png/Kosh_Logo_Blue.png".format(os.getcwd())))
        self.assertEqual(data.shape[:-1], info["size"][::-1])
        store.close()
        os.remove(kosh_db)

    def test_force_loader(self):
        store, kosh_db = self.connect()
        ds = store.create(metadata={"key1": 1, "key2": "A"})
        ds.associate(
            "tests/baselines/node_extracts2/node_extracts2.hdf5", "hdf5")

        # get data via regular loader
        original = ds.get("node/metrics_1")[:]

        # now let's register the new loader
        store.add_loader(SecondHDF5Loader)
        new = ds.get("node/metrics_1", loader=SecondHDF5Loader)

        diff = new - original * 2.

        self.assertEqual(diff.max(), 0.)
        store.close()
        os.remove(kosh_db)

    def test_hdf5(self):
        store, kosh_db = self.connect()
        ds = store.create(metadata={"key1": 1, "key2": "A"})
        kosh_id = ds.associate(
            "tests/baselines/node_extracts2/node_extracts2.hdf5", "hdf5")
        features = sorted(ds.list_features())
        self.assertEqual(features,
                         ['cycles', 'direction', 'elements', 'node', 'node/metrics_0',
                          'node/metrics_1', 'node/metrics_10', 'node/metrics_11',
                          'node/metrics_12', 'node/metrics_2', 'node/metrics_3',
                          'node/metrics_4', 'node/metrics_5', 'node/metrics_6',
                             'node/metrics_7', 'node/metrics_8', 'node/metrics_9',
                             'zone', 'zone/metrics_0', 'zone/metrics_1', 'zone/metrics_2',
                          'zone/metrics_3', 'zone/metrics_4'])
        features = sorted(
            ds.list_features(
                None,
                group="node",
                use_cache=False))
        self.assertEqual(features,
                         ['metrics_0', 'metrics_1', 'metrics_10', 'metrics_11',
                          'metrics_12', 'metrics_2', 'metrics_3',
                          'metrics_4', 'metrics_5', 'metrics_6',
                             'metrics_7', 'metrics_8', 'metrics_9', ])

        features = sorted(
            ds.list_features(
                ds._associated_data_[0],
                group="node",
                use_cache=False))
        self.assertEqual(features,
                         ['metrics_0', 'metrics_1', 'metrics_10', 'metrics_11',
                          'metrics_12', 'metrics_2', 'metrics_3',
                          'metrics_4', 'metrics_5', 'metrics_6',
                             'metrics_7', 'metrics_8', 'metrics_9', ])
        data = ds.get("node/metrics_1")
        self.assertEqual(data.shape, (2, 18))
        self.assertTrue(numpy.allclose(data[:], numpy.array([[60.208866, 91.235115, 25.287159,
                                                              52.169613, 50.000668, 13.444662,
                                                              75.868774, 20.130577, 99.07312,
                                                              81.7369, 84.11479, 36.72673,
                                                              34.91565, 98.78891, 43.68803,
                                                              69.298256, 30.138458, 14.655322],
                                                             [73.76867, 63.415592, 22.013329,
                                                              99.66734, 6.058753, 29.229004,
                                                              70.42513, 23.531456, 96.65522,
                                                              71.14644, 38.865566, 65.52718,
                                                              38.103924, 98.274895, 11.936297,
                                                              41.059917, 52.544235, 11.852329]])))
        data = ds.get("node/metrics_3")
        self.assertEqual(data.shape, (2, 18))
        self.assertTrue(numpy.allclose(data[:], numpy.array([[4.997373, 3.5991755, 70.214554,
                                                              49.85214, 2.4331465, 36.47857,
                                                              98.310455, 26.719603, 40.37639,
                                                              18.485182, 61.91034, 3.0241032,
                                                              60.081615, 75.43359, 86.26279,
                                                              13.0893955, 79.91462, 69.50662],
                                                             [38.42152, 71.49772, 78.77744,
                                                              62.538296, 48.02889, 52.152515,
                                                              68.207306, 12.370132, 94.703545,
                                                              84.25535, 29.536356, 31.391562,
                                                              11.548034, 56.365326, 98.3486,
                                                              70.53159, 78.71963, 64.26292]])))
        data = ds["node/metrics_1"]
        self.assertTrue(numpy.allclose(data[:], numpy.array([[60.208866, 91.235115, 25.287159,
                                                              52.169613, 50.000668, 13.444662,
                                                              75.868774, 20.130577, 99.07312,
                                                              81.7369, 84.11479, 36.72673,
                                                              34.91565, 98.78891, 43.68803,
                                                              69.298256, 30.138458, 14.655322],
                                                             [73.76867, 63.415592, 22.013329,
                                                              99.66734, 6.058753, 29.229004,
                                                              70.42513, 23.531456, 96.65522,
                                                              71.14644, 38.865566, 65.52718,
                                                              38.103924, 98.274895, 11.936297,
                                                              41.059917, 52.544235, 11.852329]])))
        data2 = ds["node/metrics_3"]  # noqa
        # Above used to change data[:] values at some point
        # Making sure it does not.
        self.assertTrue(numpy.allclose(data[:], numpy.array([[60.208866, 91.235115, 25.287159,
                                                              52.169613, 50.000668, 13.444662,
                                                              75.868774, 20.130577, 99.07312,
                                                              81.7369, 84.11479, 36.72673,
                                                              34.91565, 98.78891, 43.68803,
                                                              69.298256, 30.138458, 14.655322],
                                                             [73.76867, 63.415592, 22.013329,
                                                              99.66734, 6.058753, 29.229004,
                                                              70.42513, 23.531456, 96.65522,
                                                              71.14644, 38.865566, 65.52718,
                                                              38.103924, 98.274895, 11.936297,
                                                              41.059917, 52.544235, 11.852329]])))
        data = ds.get("node/metrics_1", cycles=slice(1, 2), elements=[47, 79])
        self.assertEqual(data.shape, (1, 2))
        ds.associate("tests/baselines/images/brain_398.ascii.pgm", "pgm")
        info = ds.describe_feature("node/metrics_1")
        self.assertEqual(info["size"], (2, 18))
        self.assertEqual(info["format"], "hdf5")
        h5 = ds.open(Id=kosh_id, mode="r")
        self.assertIsInstance(h5, h5py._hl.files.File)
        self.assertEqual(h5.mode, "r")
        h5.close()
        h5 = ds.open(Id=kosh_id, mode="r+")
        self.assertEqual(h5.mode, "r+")
        h5.close()
        store.close()
        os.remove(kosh_db)

    def test_npy(self):
        a = numpy.array([[1, 2, 3], [4, 5, 6]])
        name = "kosh_random_npy_{}.npy".format(random.randint(0, 23434434))
        numpy.save(name, a)

        store, kosh_db = self.connect()
        ds = store.create()
        ds.associate(name, "npy")
        self.assertEqual(ds.list_features(), ["ndarray", ])
        data = ds.get("ndarray")
        self.assertEqual(data.shape, (2, 3))
        self.assertTrue(numpy.allclose(a, data))
        info = ds.describe_feature("ndarray")
        self.assertEqual(info["size"], (2, 3))
        self.assertEqual(info["format"], "numpy")
        self.assertEqual(info["type"], a.dtype)
        store.close()
        os.remove(kosh_db)

    def assertAllClose(self, a, b):
        self.assertTrue(numpy.allclose(a, b))

    def test_numpy_loadtxt(self):
        store, kosh_db = self.connect()
        ds = store.create()
        pth = "tests/baselines/npy"

        # just numbers
        ds.associate(
            os.path.join(
                pth,
                "example_columns_no_header.txt"),
            "numpy/txt")
        self.assertEqual(ds.list_features(), ["features"])
        d1 = ds["features"]
        all = d1[:]
        self.assertEqual(all.shape, (25, 6))

        k1s = [
            slice(6, None, 3),
            6,
            6,
            slice(6, 8),
            slice(6, 8),
            slice(6, 8),
            6,
            slice(None, None, -1),
            slice(None, None, None),
            slice(6, 10, -3),
            slice(6, -2, 3),
            slice(-6, -2, 3),
            slice(-6, 23, 3),
            slice(-6, None, 3),
            slice(None, -6, 3),
            slice(-6, None, -3),
            slice(None, -6, -3),
            -7,
        ]
        k2s = [
            slice(None, None, 2),
            slice(None, None, None),
            slice(None, None, -2),
            slice(None, None, None),
            slice(None, None, 2),
            2,
            2,
            slice(None, None, None),
            slice(None, None, -1),
        ]
        while len(k2s) < len(k1s):
            k2s.append(slice(None, None, None))

        for k1, k2 in zip(k1s, k2s):
            print(k1, k2)
            tmp = d1[k1, k2]
            self.assertAllClose(tmp, all[k1, k2])
        # header with features
        ds.dissociate(os.path.join(pth, "example_columns_no_header.txt"))
        ds.associate(
            os.path.join(
                pth,
                "example_first_line_header_with_column_names.txt"),
            "numpy/txt",
            metadata={
                "features_line": 0})
        self.assertEqual(
            ds.list_features(), [
                "time", "zeros", "ones", "twos", "threes", "fours"])
        self._check_feature_values(ds)
        ds.dissociate(
            os.path.join(
                pth,
                "example_first_line_header_with_column_names.txt"))
        ds.associate(
            os.path.join(
                pth,
                "example_three_hashed_header_rows.txt"),
            "numpy/txt",
            metadata={
                "features_line": 2})
        self.assertEqual(
            ds.list_features(), [
                "time", "zeros", "ones", "twos", "threes", "fours"])
        self._check_feature_values(ds)
        ds.dissociate(
            os.path.join(
                pth,
                "example_three_hashed_header_rows.txt"))
        ds.associate(
            os.path.join(
                pth,
                "example_non_hashed_header_rows.txt"),
            "numpy/txt",
            metadata={
                "features_line": 5,
                "skiprows": 6})
        self.assertEqual(
            ds.list_features(), [
                "time", "zeros", "ones", "twos", "threes", "fours"])
        self._check_feature_values(ds)
        ds.dissociate(os.path.join(pth, "example_non_hashed_header_rows.txt"))
        id_ = ds.associate(
            os.path.join(
                pth,
                "example_tab_separated_column_names.txt"),
            "numpy/txt",
            metadata={
                "features_line": 0})
        self.assertEqual(
            ds.list_features(), [
                "time", "zeros", "ones", "twos", "threes", "fours"])
        asso = store._load(id_)
        asso.features_separator = " "
        self.assertEqual(
            ds.list_features(), [
                "time", "zeros", "ones", "twos", "threes", "fours"])
        self.assertEqual(
            ds.list_features(use_cache=False), [
                "time\tzeros\tones\ttwos\tthrees\tfours"])
        asso.features_separator = "\t"
        self.assertEqual(
            ds.list_features(use_cache=False), [
                "time", "zeros", "ones", "twos", "threes", "fours"])
        ds.dissociate(
            os.path.join(
                pth,
                "example_tab_separated_column_names.txt"))
        id_ = ds.associate(
            os.path.join(
                pth,
                "example_column_names_in_header_via_constant_width.txt"),
            "numpy/txt",
            metadata={
                "features_line": 0, "columns_width": 10})
        self.assertEqual(
            ds.list_features(), [
                "time", "zeros col", "ones  col", "twos col", "threes col", "fours"])
        store.close()
        os.remove(kosh_db)

    def _check_feature_values(self, ds):
        for i, feature in enumerate(ds.list_features()[1:]):
            z = ds[feature][3:23:4]
            self.assertEqual(z.shape, (5,))
            z = numpy.average(z)
            self.assertTrue(z < i + 1)
            self.assertTrue(z > i)


if __name__ == "__main__":
    A = KoshTestLoaders()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

import os
from koshbase import KoshTest
import kosh
import numpy


class KoshTestLoaders(KoshTest):
    def test_loader(self):
        store, kosh_db = self.connect()
        ds = store.create(metadata={"key1": 1, "key2": "A"})
        ds.associate(
            "tests/baselines/mash/node_extracts2/node_extracts2.hdf5", "hdf5")
        l = store._find_loader(ds._associated_data_[0])
        self.assertEqual(sorted(l.known_types()), ["hdf5"])
        self.assertEqual(l.known_load_formats("file"), [])
        os.remove(kosh_db)

    def test_generic_loader(self):
        store, kosh_db = self.connect()
        ds = store.create(metadata={"key1": 1, "key2": "A"})
        ds.associate("setup.py", "ascii")
        l = store._find_loader(ds._associated_data_[0])
        self.assertIsInstance(l, kosh.loaders.core.KoshFileLoader)
        self.assertEqual(sorted(l.known_types()), ["file"])
        self.assertEqual(l.known_load_formats("file"), [])
        self.assertIsInstance(ds.get(None), list)
        os.remove(kosh_db)


    def test_images(self):
        store, kosh_db = self.connect()
        ds = store.create(metadata={"key1": 1, "key2": "A"})
        ds.associate(
            "tests/baselines/images/LLNLiconWHITE.png", "png")
        features = sorted(ds.list_features())
        self.assertEqual(features, ["image",])
        ds.get("image")
        # Duplicate features names URI should be added
        ds.associate(
            "tests/baselines/images/wci_logo.gif", "gif")
        features = sorted(ds.list_features())
        self.assertEqual(features, ["image_tests/baselines/images/LLNLiconWHITE.png","image_tests/baselines/images/wci_logo.gif"])


        ds = store.create(metadata={"key1": 1, "key2": "A"})
        ds.associate("tests/baselines/images/buffalo.pgm", "pgm")
        img = ds.get("image")
        self.assertEqual(img.shape, (321, 481))

        ds = store.create(metadata={"key1": 1, "key2": "A"})
        ds.associate("tests/baselines/images/brain_398.ascii.pgm", "pgm")
        img = ds.get("image")
        self.assertEqual(img.shape, (486, 720))

        info = ds.describe_feature("image")

        self.assertEqual(sorted(info.keys()),["format", "max_value", "size"])
        self.assertEqual(info["format"], "pgm (P2)")
        self.assertEqual(info["max_value"], 255)
        ds.associate("tests/baselines/images/wci_logo.gif", mime_type="gif")
        self.assertEqual(ds.list_features(), ['image_tests/baselines/images/brain_398.ascii.pgm',
                                              'image_tests/baselines/images/wci_logo.gif'])  # URI is now added to feature to deambiguous them
        info = ds.describe_feature("image_tests/baselines/images/wci_logo.gif")
        self.assertEqual(info["size"], (595, 517))
        data = ds.get("image_tests/baselines/images/wci_logo.gif")
        self.assertEqual(data.shape, info["size"][::-1])

    def test_hdf5(self):
        store, kosh_db = self.connect()
        ds = store.create(metadata={"key1": 1, "key2": "A"})
        ds.associate(
            "tests/baselines/mash/node_extracts2/node_extracts2.hdf5", "hdf5")
        features = sorted(ds.list_features())
        self.assertEqual(features,
                         ['cycles', 'direction', 'elements', 'node/metrics_0',
                          'node/metrics_1', 'node/metrics_10', 'node/metrics_11',
                          'node/metrics_12', 'node/metrics_2', 'node/metrics_3',
                          'node/metrics_4', 'node/metrics_5', 'node/metrics_6',
                             'node/metrics_7', 'node/metrics_8', 'node/metrics_9',
                             'zone/metrics_0', 'zone/metrics_1', 'zone/metrics_2',
                          'zone/metrics_3', 'zone/metrics_4'])

        features = sorted(ds.list_features(None,"node"))
        self.assertEqual(features,
                         ['metrics_0', 'metrics_1', 'metrics_10', 'metrics_11',
                          'metrics_12', 'metrics_2', 'metrics_3',
                          'metrics_4', 'metrics_5', 'metrics_6',
                             'metrics_7', 'metrics_8', 'metrics_9', ])
        
        features = sorted(ds.list_features(ds._associated_data_[0],"node"))
        self.assertEqual(features,
                         ['metrics_0', 'metrics_1', 'metrics_10', 'metrics_11',
                          'metrics_12', 'metrics_2', 'metrics_3',
                          'metrics_4', 'metrics_5', 'metrics_6',
                             'metrics_7', 'metrics_8', 'metrics_9', ])
        data = ds.get("node/metrics_1")
        self.assertEqual(data.shape, (2, 18))
        data = ds.get("node/metrics_1", cycles=slice(1,2), elements=[15,21])
        self.assertEqual(data.shape, (1,2))
        ds.associate("tests/baselines/images/brain_398.ascii.pgm", "pgm")
        info = ds.describe_feature("node/metrics_1")
        self.assertEqual(info["size"], (2,18))
        self.assertEqual(info["format"], "hdf5")
        os.remove(kosh_db)


    def test_mash(self):
        store, kosh_db = self.connect()
        # Create many datasets
        ds = store.create(metadata={"key1": 1, "key2": "A"})
        self.assertEqual(len(ds.search()), 0)
        ds.associate("tests/baselines/mash/node_extracts2", "mash")

        features = ds.list_features()
        self.assertEqual(features, ['zone/skew', 'zone/stretch', 'zone/taper', 'zone/average energy', 'zone/zone pressure', 'node/min corner volume', 'node/min face area', 'node/min side area',
                                    'node/min side volume', 'node/node volume', 'node/node pressure', 'node/node temperature', 'node/node velocity', 'node/skew', 'node/stretch', 'node/taper', 'node/average energy', 'node/zone pressure'])
        files = ds.search(mime_type="mash")
        mash_file = files[0].open()
        self.assertEqual(mash_file.metrics_avail, {'zone': ['skew', 'stretch',
                                                            'taper', 'average energy',
                                                            'zone pressure'],
                                                   'node': ['min corner volume', 'min face area',
                                                            'min side area', 'min side volume', 'node volume',
                                                            'node pressure', 'node temperature',
                                                            'node velocity', 'skew', 'stretch', 'taper',
                                                            'average energy', 'zone pressure'],
                                                   'scalarRlxData': [], 'dimRlxData': []})
        self.assertEqual(mash_file.proc_ids, {'zone': [[20, 21, 22, 23]], 'node': [[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]], 'scalarRlxData': [
                         [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]], 'dimRlxData': [[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]]})
        self.assertEqual(mash_file.ids, {'zone': [20, 21, 22, 23], 'node': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], 'scalarRlxData': [
                         10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], 'dimRlxData': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]})
        data = mash_file.get("zone/skew")
        self.assertEqual(data.shape, (2, 4, 1))
        data = mash_file.get("zone/skew", cycles=[1, ])
        self.assertEqual(data.shape, (1, 4, 1))
        data = mash_file.get("zone/skew", elements=[20, 22])
        self.assertEqual(data.shape, (2, 2, 1))
        data2 = mash_file.get("zone/skew", elements=[22, 20])
        self.assertEqual(data2.shape, (2, 2, 1))
        self.assertFalse(numpy.allclose(data, data2))
        data2 = mash_file.get("zone/skew", elements=[20, 21])
        self.assertEqual(data2.shape, (2, 2, 1))
        self.assertFalse(numpy.allclose(data, data2))
        data2 = mash_file.get("zone/skew", cycles=[1, ], elements=[20, 21])
        self.assertEqual(data2.shape, (1, 2, 1))
        data = ds.get("zone/skew")
        self.assertEqual(data.shape, (2, 4, 1))
        self.assertEqual([a.id for a in mash_file.getAxisList("dimRlxData")], [
                         "cycles", "elements", "direction", "metrics"])
        axes = mash_file.getAxisList("zone")
        self.assertEqual([a.id for a in axes], [
                         "cycles", "elements", "metrics"])
        os.remove(kosh_db)

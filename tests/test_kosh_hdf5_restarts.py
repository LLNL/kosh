
from __future__ import print_function
from koshbase import KoshTest
import numpy


class KoshTestLoaders(KoshTest):
    def test_restart_loader(self):
        store, kosh_db = self.connect()

        D = store.create()
        D.associate("tests/baselines/hdf5/restarts.hdf5", mime_type="hdf5")


        restart = False
        features = D.list_features(restarts=restart)
        self.assertEqual(features, ['001/cycles', '002/cycles', '003/cycles', '004/cycles', '005/cycles', '006/cycles', '007/cycles', '008/cycles', '009/cycles', '010/cycles', '011/cycles', '012/cycles', '013/cycles', 'cycles', 'dimRlxData/001/feature_0', 'dimRlxData/002/feature_0', 'dimRlxData/003/feature_0', 'dimRlxData/004/feature_0', 'dimRlxData/005/feature_0', 'dimRlxData/006/feature_0', 'dimRlxData/007/feature_0', 'dimRlxData/008/feature_0', 'dimRlxData/009/feature_0', 'dimRlxData/010/feature_0', 'dimRlxData/011/feature_0', 'dimRlxData/012/feature_0', 'dimRlxData/013/feature_0', 'dimRlxData/feature_0', 'direction', 'elements', 'feature_8', 'node/001/feature_1', 'node/001/feature_2', 'node/001/feature_3', 'node/002/feature_1', 'node/002/feature_2', 'node/002/feature_3', 'node/003/feature_1', 'node/003/feature_2', 'node/003/feature_3', 'node/004/feature_1', 'node/004/feature_2', 'node/004/feature_3', 'node/005/feature_1', 'node/005/feature_2', 'node/005/feature_3', 'node/006/feature_1', 'node/006/feature_2', 'node/006/feature_3', 'node/007/feature_1', 'node/007/feature_2', 'node/007/feature_3', 'node/008/feature_1', 'node/008/feature_2', 'node/008/feature_3', 'node/009/feature_1', 'node/009/feature_2', 'node/009/feature_3', 'node/010/feature_1', 'node/010/feature_2', 'node/010/feature_3', 'node/011/feature_1', 'node/011/feature_2', 'node/011/feature_3', 'node/012/feature_1', 'node/012/feature_2', 'node/012/feature_3', 'node/013/feature_1', 'node/013/feature_2', 'node/013/feature_3', 'node/feature_1', 'node/feature_2', 'node/feature_3', 'scalarRlxData/001/feature_4', 'scalarRlxData/001/feature_5', 'scalarRlxData/001/feature_6', 'scalarRlxData/001/feature_7', 'scalarRlxData/002/feature_4', 'scalarRlxData/002/feature_5', 'scalarRlxData/002/feature_6', 'scalarRlxData/002/feature_7', 'scalarRlxData/003/feature_4', 'scalarRlxData/003/feature_5', 'scalarRlxData/003/feature_6', 'scalarRlxData/003/feature_7', 'scalarRlxData/004/feature_4', 'scalarRlxData/004/feature_5', 'scalarRlxData/004/feature_6', 'scalarRlxData/004/feature_7', 'scalarRlxData/005/feature_4', 'scalarRlxData/005/feature_5', 'scalarRlxData/005/feature_6', 'scalarRlxData/005/feature_7', 'scalarRlxData/006/feature_4', 'scalarRlxData/006/feature_5', 'scalarRlxData/006/feature_6', 'scalarRlxData/006/feature_7', 'scalarRlxData/007/feature_4', 'scalarRlxData/007/feature_5', 'scalarRlxData/007/feature_6', 'scalarRlxData/007/feature_7', 'scalarRlxData/008/feature_4', 'scalarRlxData/008/feature_5', 'scalarRlxData/008/feature_6', 'scalarRlxData/008/feature_7', 'scalarRlxData/009/feature_4', 'scalarRlxData/009/feature_5', 'scalarRlxData/009/feature_6', 'scalarRlxData/009/feature_7', 'scalarRlxData/010/feature_4', 'scalarRlxData/010/feature_5', 'scalarRlxData/010/feature_6', 'scalarRlxData/010/feature_7', 'scalarRlxData/011/feature_4', 'scalarRlxData/011/feature_5', 'scalarRlxData/011/feature_6', 'scalarRlxData/011/feature_7', 'scalarRlxData/012/feature_4', 'scalarRlxData/012/feature_5', 'scalarRlxData/012/feature_6', 'scalarRlxData/012/feature_7', 'scalarRlxData/013/feature_4', 'scalarRlxData/013/feature_5', 'scalarRlxData/013/feature_6', 'scalarRlxData/013/feature_7', 'scalarRlxData/feature_4', 'scalarRlxData/feature_5', 'scalarRlxData/feature_7', 'zone/001/feature_1', 'zone/001/feature_2', 'zone/001/feature_3', 'zone/002/feature_1', 'zone/002/feature_2', 'zone/002/feature_3', 'zone/003/feature_1', 'zone/003/feature_2', 'zone/003/feature_3', 'zone/004/feature_1', 'zone/004/feature_2', 'zone/004/feature_3', 'zone/005/feature_1', 'zone/005/feature_2', 'zone/005/feature_3', 'zone/006/feature_1', 'zone/006/feature_2', 'zone/006/feature_3', 'zone/007/feature_1', 'zone/007/feature_2', 'zone/007/feature_3', 'zone/008/feature_1', 'zone/008/feature_2', 'zone/008/feature_3', 'zone/009/feature_1', 'zone/009/feature_2', 'zone/009/feature_3', 'zone/010/feature_1', 'zone/010/feature_2', 'zone/010/feature_3', 'zone/011/feature_1', 'zone/011/feature_2', 'zone/011/feature_3', 'zone/012/feature_1', 'zone/012/feature_2', 'zone/012/feature_3', 'zone/013/feature_1', 'zone/013/feature_2', 'zone/013/feature_3', 'zone/feature_1', 'zone/feature_2', 'zone/feature_3'])
        restart = True
        features = D.list_features(restarts=restart, use_cache=False)
        self.assertEqual(features, ['cycles (13 restarts)', 'dimRlxData/feature_0 (13 restarts)', 'direction', 'elements', 'feature_8', 'node/feature_1 (13 restarts)', 'node/feature_2 (13 restarts)', 'node/feature_3 (13 restarts)', 'scalarRlxData/feature_4 (13 restarts)', 'scalarRlxData/feature_5 (13 restarts)', 'scalarRlxData/feature_7 (13 restarts)', 'zone/feature_1 (13 restarts)', 'zone/feature_2 (13 restarts)', 'zone/feature_3 (13 restarts)', 'scalarRlxData/feature_6 (13 restarts but not used on original set)'])



        info = D.describe_feature(features[5], restarts=restart)
        self.assertEqual(info["restarts"], 13)
        self.assertEqual(info["size"], (100,15))
        self.assertEqual(info["dimensions"], [{'name': 'cycles', 'first': 15996, 'last': 16095, 'length': 100}, {'name': 'elements', 'first': 10308, 'last': 10322, 'length': 15}])

        info = D.describe_feature("node/013/feature_1")
        self.assertEqual(info["size"], (36,15))
        self.assertEqual(info["dimensions"], [{'name': '013/cycles', 'first': 16003, 'last': 16038, 'length': 36}, {'name': 'elements', 'first': 10308, 'last': 10322, 'length': 15}])
        print("feature", features[5])
        self.assertEqual(D.get(features[5]).shape, (100,15))
        self.assertEqual(D.get(features[5], restart=13).shape, (36,15))
        self.assertEqual(D.get("node/013/feature_1").shape, (36,15))
        self.assertEqual(D.get(features[5], restart=13, cycles=slice(0,None)).shape, (43,15))
        self.assertEqual(D.get("cycles").shape, (100,))
        self.assertEqual(D.get("013/cycles").shape, (36,))
        self.assertEqual(D.get("cycles", restart=13).shape, (36,))
        self.assertEqual(D.get("cycles", restart=13, cycles=slice(0,None)).shape, (43,))
        self.assertEqual(D.get("cycles", restart=13, cycles=slice(0,None))[0], 15996)
        self.assertEqual(D.get("cycles", restart=13, cycles=slice(0,None))[-1], 16038)
        self.assertEqual(D.get("cycles", restart=13, cycles=slice(0,15)).shape, (15,))
        self.assertEqual(D.get("cycles", restart=13, cycles=slice(0,15))[0], 15996)
        self.assertEqual(D.get("cycles", restart=13, cycles=slice(0,15))[-1], 16010)
        self.assertEqual(D.get("elements").shape, (15,))
        self.assertEqual(D.get("node/013/feature_1", cycles=slice(0,None)).shape, (43,15))
        self.assertEqual(D.get("node/feature_1", restart=13, cycles=slice(0,None)).shape, (43,15))
        self.assertEqual(D.get("node/013/feature_1", cycles=slice(2,12)).shape, (10,15))
        self.assertEqual(D.get("node/feature_1", restart=13, cycles=slice(2,12)).shape, (10,15))
        self.assertEqual(D.get("node/013/feature_1", cycles=slice(2,12,3)).shape, (4,15))
        self.assertEqual(D.get("node/feature_1", restart=13, cycles=slice(2,12,3)).shape, (4,15))
        a = D.get("node/013/feature_1", cycles=slice(-20, -2, 3))
        self.assertEqual(a.shape, (6,15))
        b = D.get("node/013/feature_1", cycles=slice(-2, -20, -3))
        self.assertEqual(b.shape, (6,15))
        self.assertFalse(numpy.allclose(a, b[::-1]))
        a = D.get("node/013/feature_1", cycles=slice(0, None))
        self.assertEqual(a.shape, (43,15))
        b = D.get("node/013/feature_1", cycles=slice(None, None, -1))
        self.assertEqual(b.shape, (43,15))
        self.assertTrue(numpy.allclose(a, b[::-1]))
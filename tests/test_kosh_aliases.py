from __future__ import print_function
import koshbase
import os
import numpy as np
import kosh


class TestKoshAliases(koshbase.KoshTest):

    def test_aliases(self):

        store, kosh_db = self.connect()

        alias_dict = {'param1': ['PARAM1', 'P1'],
                      'param2': ['PARAM2', 'P2'],
                      'param3': ['PARAM3', 'P3'],
                      'param4': ['PARAM4', 'P4', 'cycles'],
                      'param5': 'node/metrics_5',
                      'P6': ['node/metrics_6'],
                      'P0': 'metrics_0'}

        dataset = store.create(id='test1', metadata={'myMetaData1': 1,
                                                     'alias_feature': alias_dict})

        print(dataset)

        dataset.add_curve([0, 1, 2], 'c1', 'param1')
        dataset.add_curve([0, 10, 20], 'c1', 'param2')
        dataset.add_curve([0, 100, 200], 'c2', 'param3')
        dataset.add_curve([0, 1000, 2000], 'c2', 'PARAM4')
        dataset.add_curve([0, 2, 4], 'C3', 'param1')
        dataset.add_curve([0, 20, 40], 'C3', 'param2')
        dataset.add_curve([0, 2000, 4000], 'C4', 'P4')
        dataset.associate("tests/baselines/node_extracts2/node_extracts2.hdf5",
                          mime_type="hdf5", absolute_path=False)

        val = list(store.find(id_pool='test1'))[0]
        print(val.list_features())

        # This will pass because exact match was found
        np.testing.assert_array_equal(val['c1/param1'][:], [0, 1, 2])

        # This will fail because there is more than one 'PARAM2' ['c1/param2', 'C3/param2']
        with self.assertRaises(ValueError):
            val['C2/PARAM2'][:]

        # This will pass because there is one 'P3' ['c2/param3']
        np.testing.assert_array_equal(val['c1/P3'][:], [0, 100, 200])

        # This will fail because there is more than one 'param4' ['c2/PARAM4', 'C4/P4', 'cycles']
        with self.assertRaises(ValueError):
            val['param4'][:]

        # This will pass because there is one 'param5' ['node/metrics_5']
        self.assertTupleEqual(val['param5'][:].shape, (2, 18))

        # This will pass because there is one 'P6' ['node/metrics_6']
        self.assertTupleEqual(val['P6'][:].shape, (2, 18))

        # This will fail because there is more than one 'P0' ['node/metrics_0', 'zone/metrics_0']
        with self.assertRaises(ValueError):
            val['P0'][:]

        alias_dict = {'P7': ['node/metrics_7'],
                      'P0': 'metrics_10',
                      'param2': 'no_feature',
                      'param3': 'test_feature',
                      'C4/P4': 'PARAM4'}

        dataset.alias_feature = alias_dict

        # This will pass because there is one 'P7' ['node/metrics_7']
        self.assertTupleEqual(val['P7'][:].shape, (2, 18))

        # This will pass because there is one 'P0' ['node/metrics_10']
        self.assertTupleEqual(val['P0'][:].shape, (2, 18))

        # This will fail because there is no 'param1' in dataset or in alias_feature
        with self.assertRaises(ValueError):
            val['param1'][:]

        # This will fail because there is no 'param2' matching alias even though there is ['c1/param2', 'C3/param2']
        with self.assertRaises(ValueError):
            val['param2'][:]

        # This will fail because there is no 'param3' matching alias even though there is ['c2/param3']
        with self.assertRaises(ValueError):
            val['param3'][:]

        # This will pass because there is an exact match of 'C4/P4' even though there is an alias
        np.testing.assert_array_equal(val['C4/P4'][:], [0, 2000, 4000])

        alias_dict = {'param1': 'param1',
                      'param2': 'c1/param2'}

        dataset.alias_feature = alias_dict

        # This will fail because there is more than one 'param1' ['C3/param1', 'c1/param1']
        with self.assertRaises(ValueError):
            val['param1'][:]

        # This will pass because there is one 'param2' ['c1/param2'] even though there is ['c1/param2', 'C3/param2']
        np.testing.assert_array_equal(val['param2'][:], [0, 10, 20])

        # Using get
        np.testing.assert_array_equal(val.get('param2'), [0, 10, 20])

        class ten_times(kosh.transformers.KoshTransformer):

            types = {"numpy": ["numpy"]}

            def transform(self, input, format):
                return 10 * input[:]

        # Using transformer
        np.testing.assert_array_equal(val.get('param2', transformers=[ten_times()]), [0, 100, 200])

        store.close()
        os.remove(kosh_db)

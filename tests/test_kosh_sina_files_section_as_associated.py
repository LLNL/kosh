from __future__ import print_function
from koshbase import KoshTest
import sina.utils
import numpy


class KoshTestSinaFiles(KoshTest):
    def test_sina_files_section(self):
        store, kosh_db = self.connect()
        rec = sina.utils.convert_json_to_records_and_relationships(
            "tests/baselines/sina/sina_curve_rec_mimes.json")[0][0]
        store.__record_handler__.insert(rec)
        dataset = list(store.find())[0]
        dataset.associate("tests/baselines/images/LLNLiconWHITE.png", "png")
        asso = dataset._associated_data_
        self.assertEqual(len(asso), 2)
        selfie = [x.split("__uri__")[0] for x in asso]
        self.assertTrue(dataset.id in selfie)
        features = sorted(dataset.list_features())
        self.assertEqual(features,
                         ['cycles', 'direction', 'elements', 'image', 'node',
                          'node/metrics_0', 'node/metrics_1', 'node/metrics_10',
                          'node/metrics_11', 'node/metrics_12', 'node/metrics_2',
                          'node/metrics_3', 'node/metrics_4', 'node/metrics_5',
                          'node/metrics_6', 'node/metrics_7', 'node/metrics_8',
                          'node/metrics_9', 'zone', 'zone/metrics_0', 'zone/metrics_1',
                          'zone/metrics_2', 'zone/metrics_3', 'zone/metrics_4'])
        zmet = dataset["zone/metrics_2"][:][:]
        self.assertTrue(numpy.allclose(zmet, [[63.823303, 30.278461, 53.4284, 41.42346],
                                              [88.843475, 13.9937315, 53.60822, 58.209667]]))
        zmet = dataset.get("zone/metrics_2", Id=dataset.id)[:]
        self.assertTrue(numpy.allclose(zmet, [[63.823303, 30.278461, 53.4284, 41.42346],
                                              [88.843475, 13.9937315, 53.60822, 58.209667]]))
        zmet = dataset.get("zone/metrics_2", Id="{}__uri__{}".format(
            dataset.id, "tests/baselines/node_extracts2/node_extracts2.hdf5"))[:]
        self.assertTrue(numpy.allclose(zmet, [[63.823303, 30.278461, 53.4284, 41.42346],
                                              [88.843475, 13.9937315, 53.60822, 58.209667]]))

        self.maxDiff = None
        print_str = """KOSH DATASET
	id: obj1
	name: ???
	creator: ???

--- Attributes ---
	param1: 1
	param2: 2
	param3: 3.3
	param4: string
	param5: {}
--- Associated Data (2)---
	Mime_type: hdf5
		tests/baselines/node_extracts2/node_extracts2.hdf5 ( obj1 )
	Mime_type: png
		/g/g19/cdoutrix/git/kosh/tests/baselines/images/LLNLiconWHITE.png ( {} )
--- Ensemble (0)---
\t[]
--- Ensemble Attributes ---
""".format(dataset.param5, list(dataset.find(mime_type="png", ids_only=True))[0])  # noqa

    def test_sina_files_section_with_curves(self):
        store, kosh_db = self.connect()
        rec = sina.utils.convert_json_to_records_and_relationships(
            "tests/baselines/sina/sina_curve_rec_mimes_and_curves.json")[0][0]
        store.__record_handler__.insert(rec)
        dataset = list(store.find())[0]
        asso = dataset._associated_data_
        self.assertEqual(len(asso), 2)
        selfie = [x.split("__uri__")[0] for x in asso]
        self.assertTrue(dataset.id in selfie)
        features = sorted(dataset.list_features())
        self.assertEqual(features,
                         ['cycles', 'direction', 'elements', 'node',
                          'node/metrics_0', 'node/metrics_1', 'node/metrics_10',
                          'node/metrics_11', 'node/metrics_12', 'node/metrics_2',
                          'node/metrics_3', 'node/metrics_4', 'node/metrics_5',
                          'node/metrics_6', 'node/metrics_7', 'node/metrics_8',
                          'node/metrics_9',
                          'timeplot_1', 'timeplot_1/feature_a', 'timeplot_1/feature_b',
                          'timeplot_1/time', 'timeplot_1/value',
                          'zone', 'zone/metrics_0', 'zone/metrics_1',
                          'zone/metrics_2', 'zone/metrics_3', 'zone/metrics_4'])
        zmet = dataset["zone/metrics_2"][:][:]
        self.assertTrue(numpy.allclose(zmet, [[63.823303, 30.278461, 53.4284, 41.42346],
                                              [88.843475, 13.9937315, 53.60822, 58.209667]]))
        zmet = dataset.get("zone/metrics_2", Id=dataset.id)[:]
        self.assertTrue(numpy.allclose(zmet, [[63.823303, 30.278461, 53.4284, 41.42346],
                                              [88.843475, 13.9937315, 53.60822, 58.209667]]))
        zmet = dataset.get("zone/metrics_2", Id="{}__uri__{}".format(
            dataset.id, "tests/baselines/node_extracts2/node_extracts2.hdf5"))[:]
        self.assertTrue(numpy.allclose(zmet, [[63.823303, 30.278461, 53.4284, 41.42346],
                                              [88.843475, 13.9937315, 53.60822, 58.209667]]))

        self.maxDiff = None
        print_str = """KOSH DATASET
	id: obj1
	name: ???
	creator: ???

--- Attributes ---
	param1: 1
	param2: 2
	param3: 3.3
	param4: string
	param5: {}
--- Associated Data (2)---
	Mime_type: hdf5
		tests/baselines/node_extracts2/node_extracts2.hdf5 ( obj1 )
	Mime_type: sina/curve
		internal ( timeplot_1 )
--- Ensembles (0)---
\t[]
--- Ensemble Attributes ---
""".format(dataset.param5)  # noqa

        self.assertEqual(print_str.strip(), str(dataset).strip())

    def test_sina_files_section_with_curves_and_badmime(self):
        store, kosh_db = self.connect()
        rec = sina.utils.convert_json_to_records_and_relationships(
            "tests/baselines/sina/sina_curve_rec_mimes_and_curves_and_badmime.json")[0][0]
        store.__record_handler__.insert(rec)
        dataset = list(store.find())[0]
        asso = dataset._associated_data_
        self.assertEqual(len(asso), 3)
        selfie = [x.split("__uri__")[0] for x in asso]
        self.assertTrue(dataset.id in selfie)
        features = sorted(dataset.list_features())
        self.assertEqual(features,
                         ['cycles', 'direction', 'elements', 'node',
                          'node/metrics_0', 'node/metrics_1', 'node/metrics_10',
                          'node/metrics_11', 'node/metrics_12', 'node/metrics_2',
                          'node/metrics_3', 'node/metrics_4', 'node/metrics_5',
                          'node/metrics_6', 'node/metrics_7', 'node/metrics_8',
                          'node/metrics_9',
                          'timeplot_1', 'timeplot_1/feature_a', 'timeplot_1/feature_b',
                          'timeplot_1/time', 'timeplot_1/value',
                          'zone', 'zone/metrics_0', 'zone/metrics_1',
                          'zone/metrics_2', 'zone/metrics_3', 'zone/metrics_4'])
        zmet = dataset["zone/metrics_2"][:][:]
        self.assertTrue(numpy.allclose(zmet, [[63.823303, 30.278461, 53.4284, 41.42346],
                                              [88.843475, 13.9937315, 53.60822, 58.209667]]))
        zmet = dataset.get("zone/metrics_2", Id=dataset.id)[:]
        self.assertTrue(numpy.allclose(zmet, [[63.823303, 30.278461, 53.4284, 41.42346],
                                              [88.843475, 13.9937315, 53.60822, 58.209667]]))
        self.assertTrue(numpy.allclose(zmet, [[63.823303, 30.278461, 53.4284, 41.42346],
                                              [88.843475, 13.9937315, 53.60822, 58.209667]]))

        print_str = """KOSH DATASET
	id: obj1
	name: ???
	creator: ???

--- Attributes ---
	param1: 1
	param2: 2
	param3: 3.3
	param4: string
	param5: {}
--- Associated Data (3)---
	Mime_type: hdf5
		tests/baselines/node_extracts2/node_extracts2.hdf5 ( obj1 )
	Mime_type: image/png
		foo.png ( obj1 )
	Mime_type: sina/curve
		internal ( timeplot_1 )
--- Ensembles (0)---
\t[]
--- Ensemble Attributes ---
""".format(dataset.param5)  # noqa
        self.assertEqual(print_str.strip(), str(dataset).strip())


if __name__ == "__main__":
    A = KoshTestSinaFiles()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

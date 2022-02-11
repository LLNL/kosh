from __future__ import print_function
from koshbase import KoshTest
import sina.utils
import kosh
import numpy
import os


class DIVIDE(kosh.KoshOperator):
    types = {"numpy": ["numpy", ]}

    def operate(self, *inputs, ** kargs):
        out = numpy.array(inputs[0], dtype=numpy.float64)
        for input_ in inputs[1:]:
            out /= numpy.array(input_, dtype=numpy.float64)
        return out


class KoshTestSinaCurves(KoshTest):
    def test_walk_function(self):
        my_dict = {"a": 1, "c": 2, "b": {"aa": 5, "a": 6, 5: {"t": 7}}}

        walked = kosh.utils.walk_dictionary_keys(my_dict)
        self.assertEqual(
            walked, [
                'a', 'b', 'b/5', 'b/5/t', 'b/a', 'b/aa', 'c'])
        walked = kosh.utils.walk_dictionary_keys(my_dict, "_@_")
        self.assertEqual(
            walked, [
                'a', 'b', 'b_@_5', 'b_@_5_@_t', 'b_@_a', 'b_@_aa', 'c'])

    def test_Sina_curves(self):
        store, kosh_db = self.connect()
        rec = sina.utils.convert_json_to_records_and_relationships(
            "tests/baselines/sina/sina_curve_rec.json")[0][0]
        store.__record_handler__.insert(rec)
        dataset = list(store.find())[0]
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
	Mime_type: image/png
		foo.png ( obj1 )
	Mime_type: sina/curve
		internal ( timeplot_1 )
--- Ensembles (0)---
\t[]
--- Ensemble Attributes ---
""".format(dataset.param5)  # noqa

        self.assertEqual(str(dataset).strip(), print_str.strip())
        features = dataset.list_features()
        self.assertEqual(features, ['timeplot_1',
                                    'timeplot_1/feature_a',
                                    'timeplot_1/feature_b',
                                    'timeplot_1/time',
                                    'timeplot_1/value',
                                    ]
                         )
        # Curve not existing
        with self.assertRaises(ValueError):
            dataset["timeplot_1/tiime"]

        # single curve (independent)
        self.assertTrue(numpy.allclose(
            dataset["timeplot_1/time"][:], [0, 1, 2]))
        # all curves
        self.assertTrue(numpy.allclose(dataset["timeplot_1"][:], [
                        [0, 1, 2], [1, 2, 3], [10., 20, 30.3], [10., 15, 20.]]))
        # some curves out of order
        self.assertTrue(numpy.allclose(dataset[["timeplot_1/value", "timeplot_1/time", "timeplot_1/feature_a"]][:],
                                       [[10., 15., 20.], [0, 1, 2], [1, 2, 3]]))

        os.remove(kosh_db)

    def test_operators_on_curves(self):
        store, kosh_db = self.connect()
        store.import_dataset("tests/baselines/sina/sina_curve_rec.json")
        dataset = list(store.find())[0]
        fa = dataset["timeplot_1/feature_a"]
        fb = dataset["timeplot_1/feature_b"]
        dv = DIVIDE(fa, fb)
        self.assertTrue(numpy.allclose(dv[:], [0.1, 0.1, 0.0990099]))
        os.remove(kosh_db)


if __name__ == "__main__":
    A = KoshTestSinaCurves()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

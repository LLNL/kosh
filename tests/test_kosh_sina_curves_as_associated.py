from __future__ import print_function
from koshbase import KoshTest
import sina.utils
import kosh
import numpy
import os


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
        dataset = list(store.search())[0]
        print_str = """KOSH DATASET
	id: obj1
	name:???
	creator: ???

--- Attributes ---
	initial_angle: 30
	max_density: 3
	presets: {}
	revision: 12-4-11
	total_energy: 12.2
--- Associated Data (2)---
	Mime_type: image/png
		foo.png ( obj1 )
	Mime_type: sina/curve
		internal ( timeplot_1 )
""".format(dataset.presets)  # noqa
        self.assertEqual(str(dataset), print_str)
        features = dataset.list_features()
        self.assertEqual(features, ['timeplot_1',
                                    'timeplot_1/mass',
                                    'timeplot_1/time',
                                    'timeplot_1/value',
                                    'timeplot_1/volume',
                                    ]
                         )
        # Curve not exisiting
        with self.assertRaises(ValueError):
            dataset["timeplot_1/tiime"]

        # single curve (independent)
        self.assertTrue(numpy.allclose(
            dataset["timeplot_1/time"][:], [0, 1, 2]))
        # all curves
        self.assertTrue(numpy.allclose(dataset["timeplot_1"][:], [
                        [0, 1, 2], [12, 11, 8], [10.5, 1.4, 2.2], [10., 14, 22.2]]))
        # some curves out of order
        self.assertTrue(numpy.allclose(dataset[["timeplot_1/value", "timeplot_1/time", "timeplot_1/mass"]][:],
                                       [[10.5, 1.4, 2.2], [0, 1, 2], [12, 11, 8]]))

        os.remove(kosh_db)


if __name__ == "__main__":
    A = KoshTestSinaCurves()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

from __future__ import print_function
import kosh
from koshbase import KoshTest
import os
import numpy


class KoshTestCurves(KoshTest):
    def test_add_curves(self):
        store, kosh_db = self.connect()
        ds = store.create()
        # simple curve add
        ds.add_curve([1, 2, 3])
        features = ds.list_features()
        self.assertEqual(len(features), 2)
        self.assertEqual(features, ['curve_set_0', 'curve_set_0/curve_0'])
        # delete the feature
        # Need to pass a curveset
        with self.assertRaises(ValueError):
            ds.remove_curve_or_curve_set("curve_0")
        # Delete with curveset
        ds.remove_curve_or_curve_set("curve_0", "curve_set_0")
        features = ds.list_features()
        self.assertEqual(len(features), 0)
        # Remove from non existing curveset
        ds.add_curve([1, 2, 3])
        with self.assertRaises(ValueError):
            ds.remove_curve_or_curve_set("curve_0", "curve_set_1")
        features = ds.list_features()
        self.assertEqual(len(features), 2)
        # Remove wrong name
        with self.assertRaises(ValueError):
            ds.remove_curve_or_curve_set("curve_1", "curve_set_0")
        features = ds.list_features()
        self.assertEqual(len(features), 2)
        ds.remove_curve_or_curve_set("curve_set_0/curve_0")
        # Give the curve a name
        ds.add_curve([1, 2, 3], None, "my_curve")
        features = ds.list_features()
        self.assertEqual(len(features), 2)
        self.assertTrue("curve_set_0/my_curve" in features)
        # multiple curves
        ds.add_curve([1, 2, 4], curve_name="my_other_curve")
        features = ds.list_features()
        self.assertEqual(len(features), 3)
        self.assertTrue("curve_set_0/my_curve" in features)
        self.assertTrue("curve_set_0/my_other_curve" in features)
        # auto numbering
        ds.add_curve([1, 2, 4])
        features = ds.list_features()
        self.assertEqual(len(features), 4)
        self.assertTrue("curve_set_0/my_curve" in features)
        self.assertTrue("curve_set_0/my_other_curve" in features)
        self.assertTrue("curve_set_0/curve_0" in features)
        ds.add_curve([1, 2, 4], curve_name="another_curve")
        ds.add_curve([1, 2, 4])
        features = ds.list_features()
        self.assertEqual(len(features), 6)
        self.assertTrue("curve_set_0/my_curve" in features)
        self.assertTrue("curve_set_0/my_other_curve" in features)
        self.assertTrue("curve_set_0/curve_0" in features)
        self.assertTrue("curve_set_0/curve_1" in features)
        self.assertTrue("curve_set_0/another_curve" in features)

        # Remove all the curves in a curveset
        ds.remove_curve_or_curve_set("curve_set_0")
        features = ds.list_features()
        self.assertEqual(len(features), 0)

        # Curve set but no name
        ds.add_curve([1, 2, 3], curve_set="foo")
        features = ds.list_features()
        self.assertEqual(len(features), 2)
        self.assertTrue("foo/curve_0" in features)

        # Multiple curve_sets
        ds.add_curve([3, 2, 1])
        features = ds.list_features()
        self.assertEqual(len(features), 4)
        self.assertTrue("curve_set_0/curve_0" in features)

        ds.remove_curve_or_curve_set("curve_set_0")
        ds.remove_curve_or_curve_set("foo")

        # Test dependent vs independent
        ds.add_curve([1, 2, 3])
        ds.add_curve([1, 2, 3], "curve_set_0")
        rec = ds.get_record()
        cs = rec.raw["curve_sets"]["curve_set_0"]
        self.assertTrue("curve_0" in cs["independent"])
        self.assertTrue("curve_1" in cs["dependent"])
        ds.remove_curve_or_curve_set("curve_set_0")

        # Force dependent
        ds.add_curve([1, 2, 3], independent=False)
        ds.add_curve([1, 2, 3], curve_set="curve_set_0")
        rec = ds.get_record()
        cs = rec.raw["curve_sets"]["curve_set_0"]
        self.assertTrue("curve_0" in cs["dependent"])
        self.assertTrue("curve_1" in cs["independent"])
        ds.remove_curve_or_curve_set("curve_set_0")
        # Force dependent twice
        ds.add_curve([1, 2, 3], independent=False)
        ds.add_curve([1, 2, 3], curve_set="curve_set_0", independent=False)
        rec = ds.get_record()
        cs = rec.raw["curve_sets"]["curve_set_0"]
        self.assertTrue("curve_0" in cs["dependent"])
        self.assertTrue("curve_1" in cs["dependent"])
        ds.remove_curve_or_curve_set("curve_set_0")
        store.close()
        os.remove(kosh_db)

    def test_find_curve_and_curveset_from_name(self):
        store, kosh_db = self.connect()
        ds = store.create()
        ds.add_curve([1, 2, 3], "curve_set/with/slash", "curve/with/slash",)
        ds.add_curve([1, 2, 3, 4], "curve_set/with",
                     "slash/curve/with/slash", )
        res = kosh.utils.find_curveset_and_curve_name(
            "curve_set/with/slash/curve/with/slash", ds.get_record())
        self.assertEqual(
            res,
            (('curve_set/with',
              'slash/curve/with/slash'),
             ('curve_set/with/slash',
              'curve/with/slash')))
        res = kosh.utils.find_curveset_and_curve_name("bad", ds.get_record())
        self.assertEqual(res, ())
        res = kosh.utils.find_curveset_and_curve_name(
            "curve_set/with/slash", ds.get_record())
        self.assertEqual(res, (("curve_set/with/slash", None),))
        store.close()
        os.remove(kosh_db)

    def test_kosh_curve_with_slashes(self):
        store, kosh_db = self.connect()
        ds = store.create()
        ds.add_curve([1, 2, 3], "curve_set/with/slash", "curve/with/slash")
        f = ds.list_features()
        self.assertEqual(f, ['curve_set/with/slash',
                         'curve_set/with/slash/curve/with/slash'])
        self.assertTrue(numpy.allclose(
            ds['curve_set/with/slash/curve/with/slash'][:], [1, 2, 3]))
        self.assertTrue(numpy.allclose(
            ds[('curve_set/with/slash', 'curve/with/slash')][:], [1, 2, 3]))
        ds.add_curve([1, 2, 3, 4], "curve_set/with", "slash/curve/with/slash")
        f = ds.list_features()
        self.assertEqual(f,
                         [('curve_set/with',
                           None),
                          ('curve_set/with',
                           'slash/curve/with/slash'),
                             ('curve_set/with/slash',
                              None),
                             ('curve_set/with/slash',
                              'curve/with/slash')])
        self.assertTrue(numpy.allclose(
            ds[('curve_set/with', 'slash/curve/with/slash')][:], [1, 2, 3, 4]))
        self.assertTrue(numpy.allclose(
            ds[('curve_set/with/slash', 'curve/with/slash')][:], [1, 2, 3]))
        ds.add_curve([1, 2, 3, 4, 5], "curve_set/with", "a_curve/slash")
        f = ds.list_features()
        self.assertEqual(f,
                         [('curve_set/with',
                           None),
                          ('curve_set/with',
                           'a_curve/slash'),
                             ('curve_set/with',
                              'slash/curve/with/slash'),
                             ('curve_set/with/slash',
                              None),
                             ('curve_set/with/slash',
                              'curve/with/slash')])
        self.assertTrue(numpy.allclose(
            ds['curve_set/with/a_curve/slash'][:], [1, 2, 3, 4, 5]))
        good = [numpy.array([1, 2, 3, 4]), numpy.array([1, 2, 3, 4, 5])]
        for index, array in enumerate(ds[("curve_set/with", None)][:]):
            self.assertTrue(numpy.allclose(array, good[index]))
        for index, array in enumerate(ds["curve_set/with"][:]):
            self.assertTrue(numpy.allclose(array, good[index]))

        with self.assertRaises(ValueError):
            ds.remove_curve_or_curve_set("curve_set/with", "foo")
        store.close()
        os.remove(kosh_db)


if __name__ == "__main__":
    A = KoshTestCurves()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

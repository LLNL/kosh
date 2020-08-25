import koshbase
import kosh
import time
import numpy
import shutil
import os

with open("kosh_transformers_chaining_example.ascii", "w") as f:
    f.write("1 2 3 4 5 6 7 8 9")


class StringsLoader(kosh.loaders.KoshLoader):
    types = {"ascii": ["numlist", "a_format", "another_format"]}

    def extract(self):
        with open(self.obj.uri) as f:
            return [int(x) for x in f.read().split()]

    def list_features(self):
        return ["numbers", ]


class Ints2Np(kosh.transformers.KoshTransformer):
    types = {"numlist": ["numpy"]}

    def transform(self, input, format):
        return numpy.array(input)


class Even(kosh.transformers.KoshTransformer):
    types = {"numpy": ["numpy"]}

    def transform(self, input, format):
        return numpy.take(input, numpy.argwhere(
            numpy.mod(input, 2) == 0))[:, 0]


class SlowDowner(kosh.transformers.KoshTransformer):
    types = {"numpy": ["numpy"]}

    def __init__(self, sleep_time=1, cache_dir="kosh_cache", cache=False):
        super(
            SlowDowner,
            self).__init__(
            cache_dir=cache_dir,
            cache=cache,
            sleep_time=sleep_time)
        self.sleep_time = sleep_time

    def transform(self, input, format):
        # Fakes a slow operation
        time.sleep(self.sleep_time)
        return input


class FakeLoader(kosh.loaders.KoshLoader):
    types = {"fake": ["numpy", ]}

    def extract(self):
        return numpy.arange(1000, dtype=numpy.float32)

    def list_features(self):
        return "range_1000"


class TestKoshTransformers(koshbase.KoshTest):
    def testTransformers(self):
        store, db_uri = self.connect()

        store.add_loader(StringsLoader)
        ds = store.create(name="testit")
        ds.associate(
            "kosh_transformers_chaining_example.ascii",
            mime_type="ascii")

        ds.list_features()

        ds.get("numbers")

        with self.assertRaises(Exception):
            ds.get("numbers", format="numpy")

        start = time.time()
        ds.get(
            "numbers",
            transformers=[
                Ints2Np(),
                SlowDowner(1),
                Even(),
                SlowDowner(2)])
        self.assertGreater(time.time() - start, 3.)

        start = time.time()
        ds.get(
            "numbers",
            format="numpy",
            transformers=[
                Ints2Np(),
                SlowDowner(1),
                Even(),
                SlowDowner(2)])
        # Make sure cache was off
        self.assertGreater(time.time() - start, 3.)

        start = time.time()
        # Make sure cache dir is not here
        if os.path.exists("kosh_cache"):
            shutil.rmtree("kosh_cache")
        ds.get(
            "numbers",
            format="numpy",
            transformers=[
                Ints2Np(),
                SlowDowner(
                    2,
                    cache_dir="kosh_cache",
                    cache=True),
                Even(),
                SlowDowner(1)])
        # Still should be about same length
        self.assertGreater(time.time() - start, 3.)

        start = time.time()
        ds.get(
            "numbers", format="numpy", transformers=[
                Ints2Np(), SlowDowner(
                    2, cache_dir="kosh_cache", cache=True), Even(), SlowDowner(
                    1, cache_dir="kosh_cache", cache=True)])
        # Should have showed 2 sec
        self.assertLess(time.time() - start, 2.)
        self.assertGreater(time.time() - start, 1.)

        start = time.time()
        ds.get(
            "numbers", format="numpy", transformers=[
                Ints2Np(), SlowDowner(
                    2, cache_dir="kosh_cache", cache=True), Even(), SlowDowner(
                    1, cache_dir="kosh_cache", cache=True)])
        # Should have showed an extra 1 sec
        self.assertLess(time.time() - start, 1.)
        if os.path.exists("kosh_cache"):
            shutil.rmtree("kosh_cache")
        os.remove(db_uri)
        os.remove("kosh_transformers_chaining_example.ascii")

    def test_splitter(self):
        store, db_uri = self.connect()

        store.add_loader(FakeLoader)
        ds = store.create(name="testit")
        ds.associate("fake_uri", mime_type="fake")

        ds.list_features()

        data = ds.get(
            "range_1000",
            cache=False,
            transformers=[
                kosh.transformers.Splitter(
                    n_splits=1,
                    random_state=45),
            ])
        self.assertEqual(len(data), 1)
        data = data[0]
        self.assertEqual(len(data), 2)  # No validation here
        train, test = data
        self.assertEqual(len(train), 900)
        self.assertEqual(len(test), 100)
        self.assertTrue(numpy.allclose(train[-5:], [544, 892, 643, 414, 971]))
        self.assertTrue(numpy.allclose(test[-5:], [418, 978, 65, 96, 286]))
        data = ds.get(
            "range_1000",
            cache=False,
            transformers=[
                kosh.transformers.Splitter(
                    n_splits=1,
                    random_state=45),
            ])
        train, test = data[0]
        self.assertTrue(numpy.allclose(train[-5:], [544, 892, 643, 414, 971]))
        self.assertTrue(numpy.allclose(test[-5:], [418, 978, 65, 96, 286]))

        # Random state changes means new set
        data = ds.get(
            "range_1000",
            cache=False,
            transformers=[
                kosh.transformers.Splitter(
                    n_splits=1,
                    random_state=40),
            ])
        train, test = data[0]
        self.assertFalse(numpy.allclose(train[-5:], [544, 892, 643, 414, 971]))
        self.assertFalse(numpy.allclose(test[-5:], [418, 978, 65, 96, 286]))

        # Add validation
        data = ds.get(
            "range_1000",
            cache=False,
            transformers=[
                kosh.transformers.Splitter(
                    n_splits=1,
                    random_state=73,
                    test_size=.15,
                    train_size=.75,
                    validation_size=.1),
            ])
        self.assertEqual(len(data), 1)
        data = data[0]
        self.assertEqual(len(data), 3)  # No validation here
        train, test, validation = data
        self.assertEqual(len(train), 750)
        self.assertEqual(len(test), 150)
        self.assertEqual(len(validation), 100)
        self.assertTrue(numpy.allclose(train[-5:], [784, 394, 942, 146, 918]))
        self.assertTrue(numpy.allclose(test[-5:], [16, 138, 174, 146, 150]))
        self.assertTrue(numpy.allclose(validation[-5:], [162, 69, 59, 6, 52]))
        os.remove(db_uri)

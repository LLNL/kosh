import kosh
import numpy
import os
from koshbase import KoshTest


class MyLoader(kosh.KoshLoader):
    types = {"test": ["numpy", ]}

    def __getitem__(self, key):
        length = int(os.path.basename(self.obj.uri))
        if isinstance(key, int):
            if 0 <= key < length:
                return numpy.array(key)
            elif -length <= key < 0:
                return length - (key + 1)
            else:
                raise ValueError("Index {} is out of range".format(key))
        elif isinstance(key, slice):
            start = key.start
            stop = key.stop
            step = key.step
            if start is None:
                start = 0
            if step is None:
                step = 1
            if stop is None:
                stop = length
            if -length < start < 0:
                start += length
            if -length < stop < 0:
                stop += length
            return numpy.arange(start, stop, step, dtype=numpy.float64)
        else:
            raise ValueError("Invalid key value: {}".format(key))

    def extract(self):
        length = int(os.path.basename(self.obj.uri))
        return numpy.arange(length)

    def list_features(self):
        return ["test", ]


class Flip(kosh.transformers.KoshTransformer):
    types = {"numpy": ["numpy", ]}

    def transform(self, input, format):
        if isinstance(input, numpy.int64):
            return input
        else:
            return input[::-1]


class Flip2(Flip):
    types = {"numpy": ["numpy", ]}

    def __getitem_propagate__(self, key, input_index):
        if isinstance(key, int):
            return -1 - key
        elif isinstance(key, slice):
            if key.stop is None:
                start = 0
            else:
                start = -key.stop
            if key.start is None:
                stop = None
            else:
                stop = -key.start
            return slice(start, stop, key.step)
        else:
            return None


class ADD(kosh.KoshOperator):

    types = {"numpy": ["numpy", ]}

    def operate(self, *inputs, **kargs):
        out = inputs[0]
        for input_ in inputs[1:]:
            out += input_
        return out

    def __getitem_propagate__(self, key, input_index):
        return key


class KoshTestBackPropagate(KoshTest):
    def testGetItemKosh(self):
        store, db_uri = self.connect()
        store.add_loader(MyLoader)
        dataset = store.create()
        length = 1000
        dataset.associate(str(length), "test")
        feature = dataset["test"]
        self.assertTrue(numpy.allclose(feature(), numpy.arange(length)))
        self.assertTrue(numpy.allclose(feature[:3], [0, 1, 2]))
        self.assertTrue(numpy.allclose(
            feature[-3:], [length - 3., length - 2., length - 1.]))
        os.remove(db_uri)

    def testGetItemOutofMemoryKosh(self):
        store, db_uri = self.connect()
        store.add_loader(MyLoader)
        dataset = store.create()
        length = 1000000000000000000000000000
        dataset.associate(str(length), "test")
        feature = dataset["test"]
        with self.assertRaises(ValueError):
            feature()
        self.assertTrue(numpy.allclose(feature[:3], [0, 1, 2]))
        self.assertTrue(numpy.allclose(
            feature[-3:], [length - 3., length - 2., length - 1.]))
        os.remove(db_uri)

    def testGetItemKoshTransformerNoPropagate(self):
        store, db_uri = self.connect()
        store.add_loader(MyLoader)
        dataset = store.create()
        length = 1000000
        dataset.associate(str(length), "test")
        feature = dataset.get_execution_graph("test", transformers=[Flip(), ])
        self.assertTrue(numpy.allclose(feature(), numpy.arange(length)[::-1]))
        # Transformer does not propagate,   hence extract is called in full
        # And then the subset is applyied and sent to transformer.
        # Here that means 0,1,2,3,4 will be flipped, not the last 4!
        self.assertFalse(numpy.allclose(feature[:5], feature()[:5]))
        os.remove(db_uri)

    def testGetItemKoshTransformerNoPropagateOutofMemory(self):
        store, db_uri = self.connect()
        store.add_loader(MyLoader)
        dataset = store.create()
        length = 1000000000000000000000
        dataset.associate(str(length), "test")
        feature = dataset.get_execution_graph("test", transformers=[Flip(), ])
        # Transformer does not propagate,   hence extract is called in full
        # And then the subset is applyied and sent to transformer.
        # here the full call leads to memory issues
        with self.assertRaises(ValueError):
            feature[:5]
        os.remove(db_uri)

    def testGetItemKoshTransformerPropagate(self):
        store, db_uri = self.connect()
        store.add_loader(MyLoader)
        dataset = store.create()
        length = 100
        dataset.associate(str(length), "test")
        feature = dataset.get_execution_graph("test", transformers=[Flip2(), ])
        # Transformer does propagate
        self.assertTrue(numpy.allclose(
            feature[:5], [length - 1., length - 2., length - 3., length - 4., length - 5.]))
        os.remove(db_uri)

    def testGetItemKoshTransformerPropagateDoubePass(self):
        store, db_uri = self.connect()
        store.add_loader(MyLoader)
        dataset = store.create()
        length = 1000000000000000000000
        dataset.associate(str(length), "test")
        # Flip twice so essentially do nothing
        feature = dataset.get_execution_graph("test", transformers=[Flip2(), Flip2()])
        # Transformer does propagate
        self.assertTrue(numpy.allclose(feature[:5], [0., 1., 2., 3., 4.]))
        os.remove(db_uri)

    def testGetItemKoshOperatorPropagate(self):
        store, db_uri = self.connect()
        store.add_loader(MyLoader)
        dataset = store.create()
        length = 1000
        dataset.associate(str(length), "test")
        feature = dataset.get_execution_graph("test", transformers=[Flip2(), ])
        # Flip twice so essnetially do nothing
        feature2 = dataset.get_execution_graph(
            "test", transformers=[Flip2(), Flip2()])

        A = ADD(feature, feature2)
        self.assertTrue(numpy.allclose(A[:3], [float(length) - 1, ] * 3))
        os.remove(db_uri)


if __name__ == "__main__":
    A = KoshTestBackPropagate()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

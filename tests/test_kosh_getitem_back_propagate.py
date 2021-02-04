import kosh
import numpy
import os
from koshbase import KoshTest


length = 100000000000000000000000000000000000000000000000000000
class MyLoader(kosh.KoshLoader):
    types = {"test":["numpy",]}
    def __getitem__(self, key):
        if isinstance(key, int):
            if 0 <= key < length:
                return numpy.array(key)
            elif -length <= key < 0:
                return length - (key + 1)
            else:
                raise ValueError("Index {} is out of range".format(key))
        elif isinstance(key, slice):
            if 0 <= key.start < length:
                start = key.start
            elif -length < key.start < 0:
                start = length + key.start
            if 0 <= key.stop < length:
                stop = key.stop
            elif -length < key.stop < 0:
                stop = length + key.stop

            if key.step is not None:
                return numpy.arange(start, stop, key.step)
            else:
                return numpy.arange(start, stop)
        else:
            raise ValueError("Invalid key value: {}".format(key))

    def extract(self):
        return numpy.arange(length)

    def list_features(self):
        return ["test",]

class Flip(kosh.transformers.KoshTransformer):
    types = {"numpy": ["numpy",]}
    def transform(self, input, format):
        if isinstance(input, numpy.int64):
            return input
        else:
            return input[::-1]

class Flip2(Flip):
    types = {"numpy": ["numpy",]}
    def __getitem_propagate__(self, key):
        if isinstance(key, int):
            return -1 - key
        elif isinstance(key, slice):
            return slice(-key.stop, -key.start, key.step)
        else:
            return None

class ADD(kosh.KoshOperator):
    types = {"numpy":["numpy",]}
    def operate(self, *inputs, **kargs):
        out = inputs[0]
        for input_ in inputs[1:]:
            out += input_
        return out
    def __getitem_propagate__(self, key):
        return key
class KoshTestBackPropagate(KoshTest):
    def testGetItemKosh(self):
        store, db_uri = self.connect()
        store.add_loader(MyLoader)
        dataset = store.create()
        feature = dataset["test"]
        with self.assertRaises(Exception) as err:
            print(feature())
        print(err)
        os.remove(db_uri)

    def tstGetItemKosh(self):
        store, db_uri = self.connect()
        store.add_loader(MyLoader)
        dataset = store.create()
        feature = dataset["test"]
        os.remove(db_uri)


"""
print(feature[7])
print(feature[4:7])

feature2 = dataset.get_io_graph("test", transformers=[Flip(),])
try:
    print(feature2())
except:
    print("still can't load full")
try:
    print(feature2[3:7])
except:
    print("but now can't load a slice")
feature3 = dataset.get_io_graph("test", transformers=[Flip2(),])

#print(feature3[3:7])
#print(feature[3:7])

#feature5 = dataset.get_io_graph("numbers", transformers=[Flip2(), Flip2()])
#print(feature5[3:7])

feature5 = dataset.get_io_graph("test", transformers=[Flip3(),])
A = ADD(feature, feature3)
print("******************************************************")
print("******************************************************")
print("******************************************************")
print("******************************************************")
print(A[3:7])
"""
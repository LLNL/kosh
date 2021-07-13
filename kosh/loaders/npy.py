import numpy
from .core import KoshLoader


class NpyLoader(KoshLoader):
    types = {"npy": ["numpy", ]}

    def open(self, mode='c'):
        return numpy.load(self.uri, mmap_mode=mode)

    def extract(self, feature='ndarray', format='numpy'):
        return self.open()

    def list_features(self, *args, **kargs):
        return ["ndarray", ]

    def describe_feature(self, feature):
        info = {}
        feature = self.open()
        info["size"] = feature.shape
        info["format"] = "numpy"
        info["type"] = feature.dtype
        return info

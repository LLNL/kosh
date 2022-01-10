from .core import KoshLoader
import numpy


def read_chunk(f, end='\n'):
    out = ""
    cont = True
    while cont:
        r = f.read(1).decode()
        if r == end:
            return out
        out += r


class PGMLoader(KoshLoader):
    types = {"pgm": ["numpy", ]}

    def __init__(self, obj, **kargs):
        """PGMLoader for Kosh to be able to read in pgm image files
        :param obj: Kosh obj reference
        """
        super(PGMLoader, self).__init__(obj, **kargs)

    def open(self, mode="rb"):
        """open the pgm reader
        :param mode: mode to open the file in, defaults to 'r'
        :type mode: str, optional
        :return: Image file
        :rtype: file
        """
        return open(self.uri, mode)

    def extract(self):
        """get a feature

        :return: numpy array
        :rtype: numpy.ndarray
        """
        with open(self.uri, "rb") as f:
            magic = f.read(2).decode()
            if magic == 'P2':  # ASCII encoded
                _ = f.read(2)
                _ = read_chunk(f)
                dims = read_chunk(f)
                w, h = [int(x) for x in dims.split()]
                max_value = int(read_chunk(f))
                n = w*h
                data = numpy.array([int(x) for x in f.read().decode().split()]).reshape(h, w)
            elif magic == 'P5':  # binary encoded
                _ = f.read(1)
                dims = read_chunk(f)
                w, h = [int(x) for x in dims.split()]
                max_value = int(read_chunk(f))
                n = w*h
                data = numpy.frombuffer(
                    f.read(), dtype='u1' if max_value < 256 else 'u2', count=n).reshape(h, w)
            else:
                raise ValueError("Cannot read pgm magic number {magic}".format(magic=magic))

        return data

    def list_features(self):
        """list_features lists features available

        :return: list of features you can retrievea ["iamge", ] in our case
        :rtype: list
        """

        return ["image", ]

    def describe_feature(self, feature):
        """describe_feature describe the feature as a dictionary

        :param feature: feature to describe
        :type feature: str
        :return: dictionary with attributes describing the feature: 'size', 'format', 'max_value'
        :rtype: dict
        """
        with open(self.uri, "rb") as f:
            magic = f.read(2).decode()
            if magic == 'P2':  # ASCII encoded
                _ = f.read(2)
                _ = read_chunk(f)
                dims = read_chunk(f)
                w, h = [int(x) for x in dims.split()]
                max_value = int(read_chunk(f))
            elif magic == 'P5':  # binary encoded
                _ = f.read(1)
                dims = read_chunk(f)
                w, h = [int(x) for x in dims.split()]
                max_value = int(read_chunk(f))
            else:
                raise ValueError("Cannot read pgm magic number {magic}".format(magic=magic))

        return {"size": (h, w), "format": "pgm ({magic})".format(magic=magic), "max_value": max_value}

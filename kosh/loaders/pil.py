from .core import KoshLoader
from PIL import Image
import numpy


class PILLoader(KoshLoader):
    types = {"png": ["numpy", "bytes"],
             "gif": ["numpy", "bytes"],
             "image": ["numpy", "bytes"],
             "tiff": ["numpy", "bytes"]}

    def __init__(self, obj):
        """ImageLoader for Kosh to be able to read in pillow readable image files

        :param KoshLoader: Kosh loaders base class
        :type KoshLoader: KoshLoader
        :param obj: Kosh obj reference
        """
        super(PILLoader, self).__init__(obj)

    def open(self, mode="r"):
        """open the mash reader

        :return: Image file from PIL
        """
        return Image.open(self.obj.uri)

    def extract(self):
        """get a feature

        :param feature: in this case element/metric
        :type feature: str
        :param format: desired output format (numpy only for now)
        :type format: str
        :return: numpy array
        :rtype: numpy.ndarray
        """
        if self.format == "numpy":
            return numpy.array(self.open())
        elif self.format == "bytes":
            obj = self.open()
            raw = obj.tobytes()
            return raw

    def list_features(self):
        """list_features lists features available

        :return: list of features you can retrieve
        :rtype: list
        """

        return ["image", ]

    def describe_feature(self, feature):
        """describe_feature describe the feature as a dictionary

        :param feature: feature to describe
        :type feature: str
        :return: dictionary with attributes describing the feature
        :rtype: dict
        """
        image = self.open()
        return {"size": image.size, "mode": image.mode, "format": image.format}

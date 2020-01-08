# Core module for our Kosh data access
from abc import ABCMeta, abstractmethod
from .loaders import MashLoader, KoshLoader, KoshFileLoader


class KoshAgent(object):
    """Class to manage permissions etc..."""


class KoshStoreClass(object, metaclass=ABCMeta):
    def __init__(self, sync):
        self.loaders = []
        self.storeLoader = KoshLoader
        self.add_loader(KoshFileLoader)
        self.add_loader(MashLoader)
        self.__sync__ = sync
        self.__sync__dict__ = {}

    agent = KoshAgent()

    @abstractmethod
    def search(self):
        """search store

        :raises NotImplementedError: Needs to be implemented for each engine
        """
        raise NotImplementedError()

    @abstractmethod
    def open(self):
        """open an object in the store

        :raises NotImplementedError: Needs to be implemented for each engine
        """
        raise NotImplementedError()

    @abstractmethod
    def create(self):
        """create a dataset

        :raises NotImplementedError: Needs to be implemented for each engine
        """
        raise NotImplementedError()

    def add_loader(self, loader):
        self.loaders.append(loader)


def KoshStore(engine, sync=True, *args, **kargs):
    """KoshStore return a store based on a specific engine

    :param engine: The engine used by the store (currently sina only)
    :type engine: str
    :param sync: Does Kosh sync automatically to the db (True) or on demand (False)
    :type sync: bool
    :raises RuntimeError: [description]
    :return: [description]
    :rtype: [type]
    """
    known_engines = ["sina", ]
    # Initialize and returns access class
    if engine.lower() == "sina":
        from .sina import KoshSinaStore
        return KoshSinaStore(sync=sync, *args, **kargs)
    else:
        raise RuntimeError(
            "Unknown engine type {}, supported engines: {}".format(
                engine, known_engines))


class KoshDataset(object):
    def __str__(self):
        st = ""
        st += "KOSH DATASET\n"
        st += "\tid: {}\n".format(self.__id__)
        st += "\tname:{}\n".format(self.__name__)
        st += "\tcreator: {}\n".format(self.__creator__)
        atts = self.__attributes__
        if len(atts) > 0:
            st += "\n--- Attributes ---\n"
            for a in sorted(atts):
                if a == "_associated_data_":
                    continue
                st += "\t{}: {}\n".format(a, atts[a])
        if self._associated_data_ is not None:
            st += "--- Associated Data ({})---\n".format(
                len(self._associated_data_))
            for a in self._associated_data_:
                st2 = str(self.open(a))
                st += "\n\t".join(st2.split("\n"))
        return st

    def open(self, Id=None, loader=None):
        """open an object associated with a dataset

        :param Id: id of object to open, defaults to None which means first one.
        :type Id: str, optional
        :param loader: loader to use for this object, defaults to None
        :type loader: KoshLoader, optional
        :raises RuntimeError: object id not associated with dataset
        :return: object ready to be used
        """
        if Id is None:
            if len(self._associated_data_) > 0:
                Id = self._associated_data_[0]
            else:
                for Id in self._associated_data_:
                    return self.__store__.open(Id, loader)
        elif Id not in self._associated_data_:
            raise RuntimeError(f"object {Id} is not associated with this dataset")
        return self.__store__.open(Id, loader)

    def list_features(self, Id=None, *args, **kargs):
        """list_features list features available

        :param Id: id of object to get list of features from, defaults to None which means all
        :type Id: str, optional
        :raises RuntimeError: object id not associated with dataset
        :return: list of features available
        :rtype: list
        """
        features = []
        if Id is None:
            for a in self._associated_data_:
                ld = self.__store__._find_loader(a)
                features += ld.list_features(*args, **kargs)
        elif Id not in self._associated_data_:
            raise RuntimeError(f"object {Id} is not associated with this dataset")
        else:
            ld = self.__store__._find_loader(Id)
            features = ld.list_features(*args, **kargs)
        return features

    def get(self, feature=None, Id=None, loader=None, *args, **kargs):
        """get data for a specific feature

        :param feature: feature (variable) to read, defaults to None
        :type feature: str, optional if loader does not require this
        :param Id: object to read in, defaults to None
        :type Id: str, optional
        :param loader: loader to use to get data, defaults to None means pick for me
        :raises RuntimeException: could not get feature
        :raises RuntimeError: object id not associated with dataset
        :return: [description]
        :rtype: [type]
        """
        possible_ids = []
        # we need to figure which associated data has the feature
        if Id is None:
            for a in self._associated_data_:
                ld = self.__store__._find_loader(a)
                if feature in ld.list_features() or feature is None:
                    possible_ids.append(a)
        elif Id not in self._associated_data_:
            raise RuntimeError(f"object {Id} is not associated with this dataset")
        else:
            possible_ids = [Id, ]
        for Id in possible_ids:
            try:
                op = self.open(Id, loader=loader)
                return op.get(feature, *args, **kargs)
            except Exception:
                pass
        raise Exception("could not get feature '{}' from dataset '{}'".format(
            feature, self.__id__))

    def __dir__(self):
        """__dir__ list functions and attributes associated with dataset
        :return: functions, methods, attribute associated with this dataset
        :rtype: list
        """
        current = set(super(KoshDataset, self).__dir__())
        try:
            atts = set(self.listattributes() + self.__protected__)
        except Exception:
            atts = set()
        return list(current.union(atts))

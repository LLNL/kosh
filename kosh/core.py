# Core module for our Kosh data access
from abc import ABCMeta, abstractmethod
from .loaders import KoshLoader, KoshFileLoader, PGMLoader
try:
    from .loaders import MashLoader
except ImportError:
    pass
try:
    from .loaders import KoshHDF5Loader
except ImportError:
    pass
try:
    from .loaders import PILLoader
except ImportError:
    pass


class KoshAgent(object):
    """Class to manage permissions etc..."""


class KoshStoreClass(object):
    __metaclass__ = ABCMeta

    def __init__(self, sync):
        self.loaders = {}
        self.storeLoader = KoshLoader
        self.add_loader(KoshFileLoader)
        try:
            self.add_loader(KoshHDF5Loader)
        except Exception:
            pass  # no h5py module?
        try:
            self.add_loader(PILLoader)
        except Exception:
            pass  # no PIL?
        self.add_loader(PGMLoader)
        try:
            self.add_loader(MashLoader)
        except Exception:
            pass  # no MashExtract?
        self.__sync__ = sync
        self.__sync__dict__ = {}
        self.__sync__deleted__ = {}

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

    @abstractmethod
    def save_loader(self):
        """saves a loader to the store

        :raises NotImplementedError: Needs to be implemented for each engine
        """
        raise NotImplementedError()

    def add_loader(self, loader, save=False):
        """Adds a loader to the store

        :param loader: The Kosh loader you want to add to the store
        :type loader: KoshLoader
        :param save: Do we also save it in store for later re-use
        :type save: bool

        :return: None
        :rtype: None
        """
        for k in loader.types:
            if k in self.loaders:
                self.loaders[k].append(loader)
            else:
                self.loaders[k] = [loader, ]
        if save:  # do we save it in store
            self.save_loader(loader)

    def is_synchronous(self):
        """is_synchronous is store is synchronous mode

        :return: synchronous or not
        :rtype: bool
        """
        return self.__sync__

    def synchronous(self, mode=None):
        """Change sync mode for the store

        :param mode: The mode to True means synchronous mode, False means asynchronous, None  means switch
                     anything else is ignored and it simply returns the mode
        :type mode: bool
        :return: current synchronization mode
        :rtype: bool
        """

        if mode is None:
            self.__sync__ = not self.__sync__
        elif mode in [True, False]:
            if mode and not self.__sync__:  # Going to go to always sync on need to sync first
                self.sync()
            self.__sync__ = mode
        return self.__sync__

    @abstractmethod
    def add_user(self):
        """Adds a user to the store

        :raises NotImplementedError: Needs to be implemented for each engine
        """
        raise NotImplementedError()

    @abstractmethod
    def add_user_to_group(self):
        """Adds a user to group(s)

        :raises NotImplementedError: Needs to be implemented for each engine
        """
        raise NotImplementedError()

    @abstractmethod
    def add_group(self):
        """Adds a group to the store

        :raises NotImplementedError: Needs to be implemented for each engine
        """
        raise NotImplementedError()


def KoshStore(engine="sina", sync=True, *args, **kargs):
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
        try:
            st += "\tname:{}\n".format(self.__name__)
        except Exception:
            st += "\tname:???\n"
        try:
            st += "\tcreator: {}\n".format(self.__creator__)
        except Exception:
            st += "\tcreator: ???\n"
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
            # Let's organize per mime_type
            associated = {}
            for a in self._associated_data_:
                a_obj = self.__store__._load(a)
                st2 = "{a_obj.uri} ( {a} )".format(a_obj=a_obj, a=a)
                if a_obj.mime_type not in associated:
                    associated[a_obj.mime_type] = [st2, ]
                else:
                    associated[a_obj.mime_type].append(st2)
            for mime in sorted(associated):
                st += "\tMime_type: {mime}".format(mime=mime)
                for uri in sorted(associated[mime]):
                    st += "\n\t\t{uri}".format(uri=uri)
                st += "\n"
        return st

    def _repr_pretty_(self, p, cycle):
        p.text(self.__str__())

    def open(self, Id=None, loader=None, *args, **kargs):
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
            raise RuntimeError("object {Id} is not associated with this dataset".format(Id=Id))
        return self.__store__.open(Id, loader, *args, **kargs)

    def list_features(self, Id=None, *args, **kargs):
        """list_features list features available if multiple associated data lead to duplicate feature name
        then the associated_data uri gets appended to feature name

        :param Id: id of associated object to get list of features from, defaults to None which means all
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
            if len(features) != len(set(features)):
                # duplicate features we need to redo
                # Adding uri to feature name
                ided_features = []
                for a in self._associated_data_:
                    obj = self.__store__._load(a)
                    ld = self.__store__._find_loader(a)
                    these_features = ld.list_features(*args, **kargs)
                    for feature in these_features:
                        if features.count(feature) > 1:  # duplicate
                            ided_features.append("{feature}_@_{obj.uri}".format(feature=feature, obj=obj))
                        else:  # not duplicate name
                            ided_features.append(feature)
                features = ided_features
        elif Id not in self._associated_data_:
            raise RuntimeError("object {Id} is not associated with this dataset".format(Id=Id))
        else:
            ld = self.__store__._find_loader(Id)
            features = ld.list_features(*args, **kargs)
        return features

    def describe_feature(self, feature, Id=None, **kargs):
        """describe a feature

        :param feature: feature (variable) to read, defaults to None
        :type feature: str, optional if loader does not require this
        :param Id: id of associated object to get list of features from, defaults to None which means all
        :type Id: str, optional
        :param kargs: keywords to pass to list_features (optional)
        :type kargs: keyword=value
        :raises RuntimeError: object id not associated with dataset
        :return: dictionary describing the feature
        :rtype: dict
        """
        loader = None
        if Id is None:
            for a in self._associated_data_:
                ld = self.__store__._find_loader(a)
                if feature in ld.list_features(**kargs) or \
                        (feature[:-len(ld.obj.uri)-3] in ld.list_features()
                         and feature[-len(ld.obj.uri):] == ld.obj.uri):
                    loader = ld
                    break
        elif Id not in self._associated_data_:
            raise RuntimeError("object {Id} is not associated with this dataset".format(Id=Id))
        else:
            loader = self.__store__._find_loader(Id)
        return loader.describe_feature(feature)

    def get(self, feature=None, format=None, Id=None, loader=None, *args, **kargs):
        """get data for a specific feature

        :param feature: feature (variable) to read, defaults to None
        :type feature: str, optional if loader does not require this
        :param format: desired format after extraction
        :type format: str
        :param Id: object to read in, defaults to None
        :type Id: str, optional
        :param loader: loader to use to get data, defaults to None means pick for me
        :raises RuntimeException: could not get feature
        :raises RuntimeError: object id not associated with dataset
        :return: [description]
        :rtype: [type]
        """
        if feature is None:
            out = []
            for feat in self.list_features():
                out.append(self.get(Id=None, feature=feat, format=format, loader=loader, *args, **kargs))
            return out
        possible_ids = []
        possible_formats = []
        # we need to figure which associated data has the feature
        if Id is None:
            for a in self._associated_data_:
                ld = self.__store__._find_loader(a)
                if feature in ld.list_features() or\
                        feature is None or\
                        (feature[:-len(ld.obj.uri)-3] in ld.list_features() and
                         feature[-len(ld.obj.uri):] == ld.obj.uri):
                    possible_ids.append(a)
            if possible_ids == []:  # All failed but could be something about the feature
                possible_ids = self._associated_data_[:1]
        elif Id not in self._associated_data_:
            raise RuntimeError("object {Id} is not associated with this dataset".format(Id=Id))
        else:
            possible_ids = [Id, ]
        error = None
        for Id in possible_ids:
            try:
                ld = self.__store__._find_loader(Id)
                possible_formats += ld.known_load_formats(ld.obj.mime_type)
                if (feature[:-len(ld.obj.uri)-3] in ld.list_features() and feature[-len(ld.obj.uri):] == ld.obj.uri):
                    tmp = ld.get(feature[:-len(ld.obj.uri)-3], format, *args, **kargs)
                else:
                    tmp = ld.get(feature, format, *args, **kargs)
                return tmp
            except Exception as err:  # noqa
                error = err
                import traceback
                traceback.print_exc()
                pass
        msg = "could not get feature '{feature}'".format(feature=feature)
        msg += " from dataset '{self.__id__}' in format {format},".format(self=self)
        msg += " possible formats are: {possible_formats}".format(possible_formats=possible_formats)
        if error is not None:
            msg += "\nError: {error}".format(error=error)
        raise Exception(msg)

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

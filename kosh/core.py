# Core module for our Kosh data access
from abc import ABCMeta, abstractmethod
from .loaders import KoshLoader, KoshFileLoader, PGMLoader, get_graph
from kosh.utils import compute_fast_sha
import warnings
import os
import kosh
import time
import fcntl
import copy
import collections
try:
    from .loaders import HDF5Loader
except ImportError:
    pass
try:
    from .loaders import PILLoader
except ImportError:
    pass
try:
    from .loaders import UltraLoader
except ImportError:
    pass
try:
    from .loaders import SidreMeshBlueprintFieldLoader
except ImportError:
    pass
from .loaders import JSONLoader


class KoshAgent(object):
    """Class to manage permissions etc..."""


class KoshStoreClass(object):
    """Base Store Class for Kosh backend to build uppon"""
    __metaclass__ = ABCMeta

    def __init__(self, sync, verbose=False, use_lock_file=False):
        """Constructor
        :param sync: Does this store constantly sync with db
        :type sync: bool
        :param verbose: Print warning messages and such
        :type verbose: bool
        :param use_lock_file: If you receive sqlite threads access error, turning this on might help
        :type use_lock_file: bool
        """
        self.use_lock_file = use_lock_file
        self.loaders = {}
        self.storeLoader = KoshLoader
        self.add_loader(KoshFileLoader)
        self.add_loader(JSONLoader)
        try:
            self.add_loader(HDF5Loader)
        except Exception:  # no h5py module?
            if verbose:
                warnings.warn("Could not add hdf5 loader, check if you have h5py installed."
                              " Pass verbose=False when creating the store to turn this message off")
        try:
            self.add_loader(PILLoader)
        except Exception:  # no PIL?
            if verbose:
                warnings.warn("Could not add pil loader, check if you have pillow installed."
                              " Pass verbose=False when creating the store to turn this message off")
        self.add_loader(PGMLoader)
        try:
            self.add_loader(UltraLoader)
        except Exception:  # no pydv?
            if verbose:
                warnings.warn("Could not add ultra files loader, check if you have pydv installed."
                              " Pass verbose=False when creating the store to turn this message off")
        try:
            self.add_loader(SidreMeshBlueprintFieldLoader)
        except Exception:  # no conduit?
            if verbose:
                warnings.warn("Could not add sidre blueprint meshfield loader, check if you have conduit installed."
                              " Pass verbose=False when creating the store to turn this message off")
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
        # We add a loader we need to clear the cache
        self._cached_loaders = collections.OrderedDict()
        for k in loader.types:
            if k in self.loaders:
                self.loaders[k].append(loader)
            else:
                self.loaders[k] = [loader, ]

        if save:  # do we save it in store
            self.save_loader(loader)

    def lock(self):
        """Attempts to lock the store, helps when many concurrent requests are made to the store"""
        if not self.use_lock_file:
            return
        locked = False
        while not locked:
            try:
                self.lock_file = open(self.db_uri + ".handle", "w")
                fcntl.lockf(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except Exception:
                time.sleep(0.1)

    def unlock(self):
        """Unlocks the store so other can access it"""
        if not self.use_lock_file:
            return
        fcntl.lockf(self.lock_file, fcntl.LOCK_UN)
        self.lock_file.close()
        # Wrapping this in a try/except
        # In case concurrency by same user
        # already removed the file
        try:
            os.remove(self.lock_file.name)
        except Exception:
            pass

    def __del__(self):
        """delete the KoshStore object"""
        if not self.use_lock_file:
            return
        name = self.lock_file.name
        self.lock_file.close()
        if os.path.exists(name):
            os.remove(name)

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

    def export_dataset(self, dataset_Id):
        """exports a dataset

        :param dataset_Id: Id of datset to export
        """
        return self.open(dataset_Id).export()

    def import_dataset(self, dataset, match_attributes=["name", ]):
        """import a dataset that was exported from another store
        :param dataset: Dataset object exported by another store, or a dataset
        :type dataset: json or kosh.KoshDataset
        :return: dataset
        :rtype: KoshSinaDataset
        """
        if isinstance(dataset, KoshDataset):
            dataset = dataset.export()
        min_ver = dataset["minimum_kosh_version"]
        if min_ver is not None and kosh.__version__ < min_ver:
            raise ValueError("Cannot import dataset it requires min kosh version of {}, we are at: {}".format(
                min_ver, kosh.__version__))

        # Ok now we need to see if dataset already exist?
        match_dict = {}
        for attribute in match_attributes:
            match_dict[attribute] = dataset["attributes"][attribute]

        matching = self.search(**match_dict)

        if len(matching) > 1:
            raise ValueError("dataset criterias: {} matches multiple ({}) "
                             "datasets in store {}, try changing 'matching_attributes' when calling"
                             " this function".format(
                                 match_dict, len(matching), self.db_uri))
        elif len(matching) == 1:
            # All right we do have a possible conflict here
            match = matching[0]
            match_attributes = match.listattributes(dictionary=True)
            # ok we have some match let's make sure there is no conflict
            for att in set(match_attributes).intersection(dataset["attributes"].keys()):
                if match_attributes[att] != dataset["attributes"][att]:
                    # TODO ERROR HANDLING (--force options?)
                    raise ValueError("Attribute '{}':'{}' differs from existing dataset in store ('{}')".format(
                        att, dataset["attributes"][att], match_attributes[att]))
            # Ok at this point no conflict!
            match.update(dataset["attributes"])
        else:  # Non existent dataset
            match = self.create(metadata=dataset["attributes"])

        # now we need to handle associated files
        lst = [(x.pop("uri"), x.pop("mime_type"), x.pop("associated"), x)
               for x in copy.deepcopy(dataset["associated"])]
        if len(lst) > 0:
            uris, mime_types, asso, meta = zip(*lst)
            match.associate(uris, mime_types, metadata=meta, absolute_path=False)
        return match

    def reassociate(self, target, source=None, absolute_path=True):
        """This function allows to re-associate data whose uri might have changed

        The source can be the original uri or sha and target is the new uri to use.
        :param target: New uri
        :type target: str
        :param source: uri or sha (long or short of reassociate) to reassociate
                       with target, if None then the short uri from target will be used
        :type source: str or None
        :param absolute_path: if file exists should we store its absolute_path
        :type absolute_path: bool
        :return: None
        :rtype: None
        """

        # First let's convert to abs path if necessary
        if absolute_path:
            if os.path.exists(target):
                target = os.path.abspath(target)
            if source is not None and os.path.exists(source):
                source = os.path.abspath(source)

        # Now, did we pass a source for uri to replace?
        if source is None:
            source = kosh.utils.compute_fast_sha(target)

        # Ok now let's get all associated uri that match
        # Fist assuming it's a fast_sha search all "kosh files" that match this
        matches = self.search(kosh_type="file", fast_sha=source, ids_only=True)
        # Now it could be simply a uri
        matches += self.search(kosh_type="file", uri=source, ids_only=True)
        # And it's quite possible it's a long_sha too
        matches += self.search(kosh_type="file", long_sha=source, ids_only=True)

        # And now let's do the work
        for match_id in matches:
            try:
                match = self._load(match_id)
                match.uri = target
            except Exception:
                pass

    def cleanup_files(self, dry_run=False, interactive=False, **dataset_search_keys):
        """Cleanup the store from references to dead files
        You can filter associated objects for each dataset by passing key=values
        e.g mime_type=hdf5 will only dissociate non-existing files associated with mime_type hdf5
        some_att=some_val will only dissociate non-exisiting files associated and having the attribute
        'some_att' with value of 'some_val'
        returns list of uris to be removed.
        :param dry_run: Only does a dry_run
        :type dry_run: bool
        :param interactive: interactive mode, ask before dissociating
        :type interactive: bool
        :returns: list of uris (to be) removed.
        :rtype: list
        """
        missings = []
        datasets = self.search()
        for dataset in datasets:
            missings += dataset.cleanup_files(dry_run=dry_run,
                                              interactive=interactive, **dataset_search_keys)
        return missings


def KoshStore(db_uri=None, engine="sina", sync=True, verbose=False, *args, **kargs):
    """KoshStore return a store based on a specific engine

    :param db_uri: URI to access backend database
    :type db_uri: str
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
        return KoshSinaStore(db_uri=db_uri, sync=sync, verbose=verbose, *args, **kargs)
    else:
        raise RuntimeError(
            "Unknown engine type {}, supported engines: {}".format(
                engine, known_engines))


class KoshDataset(object):
    def __str__(self):
        """string representation"""
        st = ""
        st += "KOSH DATASET\n"
        st += "\tid: {}\n".format(self.__id__)
        try:
            st += "\tname:{}\n".format(self.__name__)
        except Exception:
            st += "\tname:???\n"
        try:
            st += "\tcreator: {}\n".format(self.creator)
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

    def cleanup_files(self, dry_run=False, interactive=False, **search_keys):
        """Cleanup the dataset from references to dead files
        You can filter associated objects by passing key=values
        e.g mime_type=hdf5 will only dissociate non-existing files associated with mime_type hdf5
        some_att=some_val will only dissociate non-exisiting files associated and having the attribute
        'some_att' with value of 'some_val'
        returns list of uris to be removed.
        :param dry_run: Only does a dry_run
        :type dry_run: bool
        :param interactive: interactive mode, ask before dissociating
        :type interactive: bool
        :returns: list of uris (to be) removed.
        :rtype: list
        """
        print_some = False
        missings = []
        for associated in self.search(**search_keys):
            clean = 'n'
            if not os.path.exists(associated.uri):  # Ok this is gone
                missings.append(associated.uri)
                if not print_some and (interactive or dry_run):
                    print_some = True
                if dry_run:  # Dry run
                    clean = 'n'
                elif interactive:
                    clean = input("\tDo you want to dissociate {} (mime_type: {})? [Y/n]".format(
                        associated.uri, associated.mime_type)).strip()
                    if len(clean) > 0:
                        clean = clean[0]
                        clean = clean.lower()
                    else:
                        clean = 'y'
                else:
                    clean = 'y'
                if clean == 'y':
                    self.dissociate(associated.uri)
        return missings

    def _repr_pretty_(self, p, cycle):
        """Pretty display in Ipython"""
        p.text(self.__str__())

    def list_attributes(self, dictionary=False):
        """list_attributes list all non protected attributes

        :parm dictionary: return a dictionary of value/pair rather than just attributes names
        :type dictionary: bool

        :return: list of attributes set on object
        :rtype: list
        """
        raise NotImplementedError

    def export(self):
        """Exports this dataset
        :return: datset and its associated data
        :rtype: dict"""
        output_dict = {
            "minimum_kosh_version": None,
            "kosh_version": kosh.__version__,
            "attributes": self.list_attributes(dictionary=True)
        }
        associated_records = []
        for associated in self._associated_data_:
            a = self.__store__._load(associated)
            associated_records.append(a.list_attributes(dictionary=True))
        output_dict["associated"] = associated_records
        return output_dict

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

    def list_features(self, Id=None, loader=None, use_cache=True, *args, **kargs):
        """list_features list features available if multiple associated data lead to duplicate feature name
        then the associated_data uri gets appended to feature name

        :param Id: id of associated object to get list of features from, defaults to None which means all
        :type Id: str, optional
        :param loader: loader to use to search for feature, will return ONLY features that the loader knows about
        :type loader: kosh.loaders.KoshLoader
        :param use_cache: If features is found on cache use it (default: True)
        :type use_cache: bool
        :raises RuntimeError: object id not associated with dataset
        :return: list of features available
        :rtype: list
        """
        if use_cache and self.__dict__["__features__"].get(Id, {}).get(loader, None) is not None:
            return self.__dict__["__features__"][Id][loader]
        # Ok no need to sync any of this we will not touch the code
        saved_sync = self.__store__.is_synchronous()
        if saved_sync:
            # we will not update any rec in here, turnin off sync
            # it makes things much d=faster
            backup = self.__store__.__sync__dict__
            self.__store__.__sync__dict__ = {}
            self.__store__.synchronous()
        features = []
        loaders = []
        associated_data = self._associated_data_
        if Id is None:
            for associated in associated_data:
                if loader is None:
                    ld, _ = self.__store__._find_loader(associated)
                else:
                    if associated not in self.__store__._cached_loaders:
                        self.__store__._cached_loaders[associated] = loader(self.__store__._load(associated))
                    ld = self.__store__._cached_loaders[associated]
                loaders.append(ld)
                features += ld._list_features(*args, use_cache=use_cache, **kargs)
            if len(features) != len(set(features)):
                # duplicate features we need to redo
                # Adding uri to feature name
                ided_features = []
                for index, associated in enumerate(associated_data):
                    obj = self.__store__._load(associated)
                    ld = loaders[index]
                    these_features = ld._list_features(*args, use_cache=use_cache, **kargs)
                    for feature in these_features:
                        if features.count(feature) > 1:  # duplicate
                            ided_features.append("{feature}_@_{obj.uri}".format(feature=feature, obj=obj))
                        else:  # not duplicate name
                            ided_features.append(feature)
                features = ided_features
        elif Id not in self._associated_data_:
            raise RuntimeError("object {Id} is not associated with this dataset".format(Id=Id))
        else:
            ld, _ = self.__store__._find_loader(Id)
            features = ld._list_features(*args, use_cache=use_cache, **kargs)
        features_id = self.__dict__["__features__"].get(Id, {})
        features_id[loader] = features
        self.__dict__["__features__"][Id] = features_id
        if saved_sync:
            # we need to restore sync mode
            self.__store__.__sync__dict__ = backup
            self.__store__.synchronous()
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
                ld, _ = self.__store__._find_loader(a)
                if feature in ld._list_features(**kargs) or \
                        (feature[:-len(ld.obj.uri) - 3] in ld._list_features()
                         and feature[-len(ld.obj.uri):] == ld.obj.uri):
                    loader = ld
                    break
        elif Id not in self._associated_data_:
            raise RuntimeError("object {Id} is not associated with this dataset".format(Id=Id))
        else:
            loader, _ = self.__store__._find_loader(Id)
        return loader.describe_feature(feature)

    def get_execution_graph(self, feature=None, Id=None, loader=None, transformers=[], *args, **kargs):
        """get data for a specific feature
        :param feature: feature (variable) to read, defaults to None
        :type feature: str, optional if loader does not require this
        :param Id: object to read in, defaults to None
        :type Id: str, optional
        :param loader: loader to use to get data,
                       defaults to None means pick for me
        :type loader: kosh.loaders.KoshLoader
        :param transformers: A list of transformers to use after the data is loaded
        :type transformers: kosh.transformer.KoshTranformer
        :returns: [description]
        :rtype: [type]
        """
        if feature is None:
            out = []
            for feat in self.list_features():
                out.append(self.get_execution_graph(Id=None,
                                                    feature=feat,
                                                    format=format,
                                                    loader=loader,
                                                    transformers=transformers,
                                                    *args, **kargs))
            return out
        # Need to make sure transformers are a list
        if not isinstance(transformers, (list, tuple)):
            transformers = [transformers, ]
        # we need to figure which associated data has the feature
        if not isinstance(feature, list):
            features = [feature, ]
        else:
            features = feature
        possibles = {}
        inter = None
        union = set()
        for feature_ in features:
            possible_ids = []
            if Id is None:
                for a in self._associated_data_:
                    a_obj = self.__store__._load(a)
                    if loader is None:
                        ld, _ = self.__store__._find_loader(a)
                    else:
                        if a_obj.mime_type in loader.types:
                            ld = loader(a_obj)
                        else:
                            continue
                    if ("_@_" not in feature_ and feature_ in ld._list_features()) or\
                            feature_ is None or\
                            (feature_[:-len(ld.obj.uri) - 3] in ld._list_features() and
                             feature_[-len(ld.obj.uri):] == ld.obj.uri):
                        possible_ids.append(a)
                if possible_ids == []:  # All failed but could be something about the feature
                    raise ValueError("Cannot find feature {} in dataset".format(feature_))
            elif Id not in self._associated_data_:
                raise RuntimeError("object {Id} is not associated with this dataset".format(Id=Id))
            else:
                possible_ids = [Id, ]
            if inter is None:
                inter = set(possible_ids)
            else:
                inter = inter.intersection(set(possible_ids))
            union = union.union(set(possible_ids))
            possibles[feature_] = possible_ids

        if len(inter) != 0:
            union = inter

        ids = {}
        # Now let's go through each possible uri
        # and group features in thems
        for id_ in union:
            matching_features = []
            for feature_ in features:
                if feature_ in possibles and id_ in possibles[feature_]:
                    matching_features.append(feature_)
                    del(possibles[feature_])
            if len(matching_features) > 0:
                ids[id_] = matching_features

        out = []
        for id_ in ids:
            features = ids[id_]
            for Id in possible_ids:
                tmp = None
                try:
                    if loader is None:
                        ld, mime_type = self.__store__._find_loader(Id)
                    else:
                        if Id not in self.__store__._cached_loaders:
                            a_obj = self.__store__._load(Id)
                            self.__store__._cached_loaders[Id] = loader(a_obj)
                            mime_type = a_obj.mime_type
                        ld = self.__store__._cached_loaders[Id]
                    # Essentially make a copy
                    # Because we want to attach the feature to it
                    # But lets not lose the cached list_features
                    saved_listed_features = ld.__dict__["_KoshLoader__listed_features"]
                    ld = ld.__class__(ld.obj)
                    ld.__dict__["_KoshLoader__listed_features"] = saved_listed_features
                    # Ensures there is a possible path to format
                    get_graph(mime_type, ld, transformers)
                    final_features = []
                    for feature_ in features:
                        if (feature_[:-len(ld.obj.uri) - 3] in ld._list_features()
                                and feature_[-len(ld.obj.uri):] == ld.obj.uri):
                            final_features.append(
                                feature_[:-len(ld.obj.uri) - 3])
                        else:
                            final_features.append(feature_)
                    if len(final_features) == 1:
                        final_features = final_features[0]
                    tmp = ld.get_execution_graph(final_features,
                                                 transformers=transformers)
                    ld.feature = final_features
                    ExecGraph = kosh.exec_graphs.KoshExecutionGraph(tmp)
                except Exception:
                    import traceback
                    traceback.print_exc()
                    ExecGraph = kosh.exec_graphs.KoshExecutionGraph(tmp)
                out.append(ExecGraph)

        if len(out) == 1:
            return out[0]
        else:
            return out

    def get(self, feature=None, format=None, Id=None, loader=None, group=False, transformers=[], *args, **kargs):
        """get data for a specific feature
        :param feature: feature (variable) to read, defaults to None
        :type feature: str, optional if loader does not require this
        :param format: desired format after extraction
        :type format: str
        :param Id: object to read in, defaults to None
        :type Id: str, optional
        :param loader: loader to use to get data,
                       defaults to None means pick for me
        :type loader: kosh.loaders.KoshLoader
        :param group: group multiple features in one get call, assumes loader can handle this
        :type group: bool
        :param transformers: A list of transformers to use after the data is loaded
        :type transformers: kosh.transformer.KoshTranformer
        :raises RuntimeException: could not get feature
        :raises RuntimeError: object id not associated with dataset
        :returns: [description]
        :rtype: [type]
        """
        G = self.get_execution_graph(feature=feature, Id=Id, loader=loader, transformers=transformers, *args, **kargs)
        if isinstance(G, list):
            return [g.traverse(format=format, *args, **kargs) for g in G]
        else:
            return G.traverse(format=format, *args, **kargs)

    def __getitem__(self, feature):
        """Shortcut to access a feature or list of
        :param feature: feature(s) to access in dataset
        :type feature: str or list of str
        :returns: (list of) access point to feature requested
        :rtype: (list of) kosh.execution_graph.KoshIoGraph
        """
        return self.get_execution_graph(feature)

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

    def reassociate(self, target, source=None, absolute_path=True):
        """This function allows to re-associate data whose uri might have changed

        The source can be the original uri or sha and target is the new uri to use.
        :param target: New uri
        :type target: str
        :param source: uri or sha (long or short of reassociate)
                       to reassociate with target, if None then the short uri from target will be used
        :type source: str or None
        :param absolute_path: if file exists should we store its absolute_path
        :type absolute_path: bool
        :return: None
        :rtype: None
        """
        # First let's convert to abs path if necessary
        if absolute_path:
            if os.path.exists(target):
                target = os.path.abspath(target)
            if source is not None and os.path.exists(source):
                source = os.path.abspath(source)

        # Now, did we pass a source for uri to replace?
        if source is None:
            source = compute_fast_sha(target)

        # Ok now let's get all associated uri that match
        # Fist assuming it's a fast_sha
        matches = self.search(fast_sha=source)
        # Now it could be simply a uri
        matches += self.search(uri=source)
        # And it's quite possible it's a long_sha too
        matches += self.search(long_sha=source)

        # And now let's do the work
        for match in matches:
            match.uri = target

    def validate(self):
        """If dataset has a schema then make sure all attributes pass the schema"""
        if self.schema is not None:
            self.schema.validate(self)

    def searchable_source_attributes(self):
        """Returns all the attributes of associated sources
        :return: List of all attributes you can use to search sources in the dataset
        :rtype: set
        """
        searchable = set()
        for source in self.search():
            searchable = searchable.union(source.listattributes())
        return searchable

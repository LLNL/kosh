from kosh.transformers import kosh_cache_dir
import os
import hashlib
import pickle
from kosh.exec_graphs import KoshExecutionGraph
import numpy
from ..core_sina import KoshSinaObject, KoshSinaFile
from ..utils import get_graph
from ..dataset import KoshDataset
from ..ensemble import KoshEnsemble


class KoshGenericObjectFromFile(object):
    """Kosh object pointing to a file"""

    def __init__(self, *args, **kwds):
        self.args = args
        self.kwds = kwds
        self.file_obj = open(*self.args, **self.kwds)

    def __enter__(self):
        self.file_obj = open(*self.args, **self.kwds)
        return self.file_obj

    def __exit__(self, *args):
        self.file_obj.close()

    def get(self, *args, **kargs):
        """Reads the file all arguments are ignored"""
        return self.file_obj.read()


class KoshLoader(KoshExecutionGraph):
    """
    :param types: types is a dictionary on known type that can be loaded
    as key and export format as value, defaults to {"dataset": []}
    :type types: dict
    """
    types = {"dataset": []}

    def __init__(self, obj, mime_type=None, uri=None):
        """KoshLoader generic Kosh loader
        :param obj: object the loader will try to load from
        :type obj: object
        :param mime_type: If you want to force the mime_type to use
        :type mime_type: str
        :param uri: If you want/need to force the uri to use
        :type uri: str
        """
        self.signature = hashlib.sha256(repr(self.__class__).encode())
        self.signature = self.update_signature(obj.id)
        if mime_type is None:
            self._mime_type = obj.mime_type
        else:
            self._mime_type = mime_type
        if uri is None:
            try:
                self.uri = obj.uri
            except AttributeError:  # Not uri on this
                self.uri = None
        else:
            self.uri = uri
        rec = obj.__store__.get_record(obj.id)
        if (rec["type"] not in obj.__store__._kosh_reserved_record_types and mime_type is None)\
                or rec["type"] in ["__kosh_storeinfo__", obj.__store__._ensembles_type]:
            self._mime_type = "dataset"
        if self._mime_type not in self.types:
            open_anything = False
            for t in self.types:
                if t == "dataset":  # special skipping
                    continue
                if len(self.types[t]) == 0:
                    open_anything = True
            if not open_anything:
                raise RuntimeError(
                    "will not be able to load object of type {mime_type}".format(mime_type=self._mime_type))
        self.obj = obj
        self.__listed_features = None

    def known_types(self):
        """Lists types of Kosh objects this loader can handle

        :return: list of Kosh type this loader can handle
        :rtype: list
        """
        return list(self.types.keys())

    def known_load_formats(self, atype):
        """Lists all the formats this loader knows how to export to for a given type

        :param atype: type we wish to to the formats for
        :type format: str
        :return: list of format this type can be exported to by the loader
        :rtype: list
        """
        return self.types.get(atype, [])

    def open(self, mode="r"):
        """Open function
        :param mode: mode to open the object in
        :type mode: str
        :return: opened object
        :rtype: object"""
        return self

    def update_signature(self, *args, **kargs):
        """Updated the signature based to a set of args and kargs
        :param *args: as many arguments as you want
        :type *args: list
        :param **kargs: key=value style arguments
        :type **kargs: dict
        :return: updated signature
        :rtype: str
        """
        signature = self.signature.copy()
        for arg in args:
            signature.update(repr(arg).encode())
        for kw in kargs:
            signature.update(repr(kw).encode())
            signature.update(repr(kargs[kw]).encode())
        return signature

    def get_execution_graph(self, feature, transformers=[]):
        """Generates the execution graph to extract a feature and possibly transform it.

        :param feature: desired feature
        :type feature: str
        :param format: desired output format
        :type format: str
        :param transformers: A list of transformers to use after the data is loaded
        :type transformers: kosh.transformer.KoshTranformer
        :return: execution graph to get to the possibly transformed feature
        :rtype: networkx.OrderDiGraph
        """
        # Let's get the execution path
        G = get_graph(self._mime_type, self, transformers)
        return G

    def get(self, feature, format=None, transformers=[],
            use_cache=True, cache_file_only=False, cache_dir=None,
            **kargs):
        """Extracts a feature and possibly transforms it
        :param feature: desired feature
        :type feature: str
        :param format: desired output format
        :type format: str
        :param transformers: A list of transformers to use after the data is loaded
        :type transformers: kosh.transformer.KoshTranformer
        :param use_cache: Try to use cached data if available
        :type use_cache: bool
        :param cache_file_only: If True, simply return name of cache_file
        :type cache_file_only: bool
        :param cache_dir: where do we cache the result?
        :type cache_dir: str
        **kargs will be stored on loader object
        format and feature are stored on the object for extraction by extraction functions
        This function calls first the loader's preprocess function
        This is followed by an actual data extraction via the 'extract' function
        Finally 'postprocess' is called on the extracted data

        Reserved keyword:
        preprocess: function use to preprocess (default to self.preprocess)
        postprocess: function use to postprocess (default to self.postprocess)
        """

        if cache_dir is None:
            cache_dir = kosh_cache_dir
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.cache_file_only = cache_file_only
        self.feature = feature
        self._user_passed_parameters = (None, kargs)
        G = self.get_execution_graph(feature, transformers=transformers)
        return KoshExecutionGraph(G).traverse(format=format, **kargs)

    def extract_(self, format):
        if format is None:
            format = self.types[self._mime_type][0]
        if len(self.types) != 0 and format not in self.types[self._mime_type]:
            raise ValueError("Loader cannot output type {self._mime_type} to {format} format".format(
                self=self, format=format))
        self.format = format
        _, kargs = self._user_passed_parameters
        signature = self.update_signature(self.feature, format, **kargs).hexdigest()
        if self.cache_file_only is True:
            # Ok user just wants to know where cache should be (plus/minus extesions)
            return signature
        # Let's generate the signatures for each step of the path.
        # And try to load it
        cache_success = False
        if self.use_cache:
            try:
                data = self.load(signature)
                cache_success = True
            except Exception:
                pass
            if cache_success:
                return data

        kargs.get("preprocess", self.preprocess)()
        data = self.extract()
        data = kargs.get("postprocess", self.postprocess)(data)
        return data

    def save(self, cache_file, content):
        """Pickles some data to a cache file
        :param cache_file: name of cache file, will be joined with self.cache_dir
        :type cache_file: str
        :param content: content to save to cache
        :type content: object
        """
        with open(os.path.join(self.cache_dir, cache_file), "wb") as f:
            pickle.dump(content, f)

    def load(self, cache_file):
        """Loads content from cache
        :param cache_file: name of cache file, will be joined with self.cache_dir
        :type cache_file: str
        :return: unpickled data
        :rtpye: object
        """
        with open(os.path.join(self.cache_dir, cache_file), "rb") as f:
            data = pickle.load(f)
        return data

    def _list_features(self, *args, **kargs):
        """Wrapper on top of list_features to snatch from cache rather than calling every time"""
        use_cache = kargs.pop("use_cache", True)
        if self.__listed_features is None or not use_cache:
            try:
                self.__listed_features = self.list_features(*args, **kargs)
            except Exception:
                # Broken loader at the moment
                self.__listed_features = []
        out = self.__listed_features
        # Reset
        if not use_cache:
            self.__listed_features = None
        return out

    def list_features(self, *args, **kargs):
        """list_features Given the obj it's loading return a list of features (variables)
        it can extract

        :return: list of available features from this loader
        :rtype: list
        """
        return []

    def describe_feature(self, feature):
        """describe_feature describe the feature as a dictionary

        :param feature: feature to describe
        :type feature: str
        :return: dictionary with attributes describing the feature
        :rtype: dict
        """
        raise NotImplementedError("describe_feature method not implemented")

    def preprocess(self):
        """preprocess sets things up for the extract function

        This should be preceeded by a call to 'get' which stored its args
        in self._user_passed_parameters
        """
        return

    def extract(self, feature, format):
        """extract this function does the heavy lifting of the extraction
        it needs to be implemented by each loader.

        We recommend returning pointer to the data as much as possible

        :raises NotImplementedError:
        """
        raise NotImplementedError

    def postprocess(self, data):
        """postprocess Given the extracted data apply some post processing to it

        :param data: result of the extract function
        :type data: any
        :return: post processed
        :rtype: any
        """
        return data


class KoshFileLoader(KoshLoader):
    """Kosh loader to load content from files"""
    types = {"file": []}

    def __init__(self, obj, **args):
        super(KoshFileLoader, self).__init__(obj)

    def open(self, mode='r'):
        """open/load the matching Kosh Sina File

        :param mode: mode to open the file in, defaults to 'r'
        :type mode: str, optional
        :return: Kosh File object
        """
        return KoshGenericObjectFromFile(self.uri, mode)

    def extract(self, feature, format):
        """extract return a feature from the loaded object.

        :param feature: variable to read from file
        :type feature: str
        :param format: desired output format
        :type format: str
        :return: data
        """
        with open(self.uri) as f:
            return f.read()

    def list_features(self, *args, **kargs):
        """list_features list features in file,

        :return: list of features available in file
        :rtype: list
        """
        return []

    def describe_feature(self, feature):
        """describe a feature

        :param feature: feature (variable) to read, defaults to None
        :type feature: str, optional if loader does not require this
        :return: dictionary describing the feature
        :rtype: dict
        """
        if feature not in self.list_features():
            raise ValueError("feature {feature} is not available".format(feature=feature))
        return {}


class KoshSinaLoader(KoshLoader):
    """Sina base class for loaders"""
    types = {"dataset": ["numpy", ]}

    def __init__(self, obj, **kargs):
        """KoshSinaLoader generic sina-based loader
        """
        super(KoshSinaLoader, self).__init__(obj, **kargs)

    def open(self, *args, **kargs):
        """open the object
        """
        record = self.obj.__store__.get_record(self.obj.id)
        if record["type"] not in self.obj.__store__._kosh_reserved_record_types:
            return KoshDataset(
                self.obj.id, store=self.obj.__store__, record=record)
        if record["type"] == self.obj.__store__._sources_type:
            return KoshSinaFile(
                self.obj.id, store=self.obj.__store__, record=record)
        elif record["type"] == self.obj.__store__._ensembles_type:
            return KoshEnsemble(
                self.obj.id, store=self.obj.__store__, record=record)
        else:
            return KoshSinaObject(self.obj.id, self.obj.__store__, record["type"], protected=[
            ], record_handler=self.obj.__store__.__record_handler__, record=record)

    def list_features(self):
        record = self.obj.__store__.get_record(self.obj.id)
        # Using set in case a variable is both in independent and dependent
        # Dependent would win when getting the data
        curves = set()
        for curve in record["curve_sets"]:
            curves.add(curve)
            for curve_type in ["independent", "dependent"]:
                for name in record["curve_sets"][curve][curve_type]:
                    curves.add("{}/{}".format(curve, name))
        return sorted(curves)

    def extract(self, *args, **kargs):
        features = self.feature
        if not isinstance(features, list):
            features = [self.feature, ]
        record = self.obj.__store__.get_record(self.obj.id)
        out = []
        for feature in features:
            sp = feature.split("/")
            # Here we are assuming the curve root name cannot have "/" in it
            curve_root = record["curve_sets"][sp[0]]
            if len(sp) > 1:
                curve_name = "/".join(sp[1:])
                if curve_name in curve_root["dependent"]:
                    curve = curve_root["dependent"][curve_name]["value"]
                else:
                    curve = curve_root["independent"][curve_name]["value"]
                out.append(numpy.array(curve))
            else:
                # we want all curves
                all = []
                # Matching order (indep/dep) that we used in list_features
                for curve_type in ["independent", "dependent"]:
                    # Same order as list_features()
                    for curve_name in sorted(curve_root[curve_type].keys()):
                        curve = curve_root[curve_type][curve_name]["value"]
                        all.append(numpy.array(curve))
                out.append(all)

        if not isinstance(self.feature, list):
            return out[0]
        else:
            return out

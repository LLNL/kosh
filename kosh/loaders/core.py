from kosh.transformers import get_path, kosh_cache_dir
import os
import hashlib
import pickle


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


class KoshLoader(object):
    """
    :param types: types is a dictionary on known type that can be loaded
    as key and export format as value, defaults to {"dataset": []}
    :type types: dict
    """
    types = {"dataset": []}

    def __init__(self, obj):
        """KoshLoader generic Kosh loader
        :param obj: object the loader will try to load from
        :type obj: object
        """
        self.signature = hashlib.sha256(repr(self.__class__).encode())
        self.signature = self.update_signature(obj.__id__)
        mime_type = obj.mime_type
        if mime_type == obj.__store__._dataset_record_type:
            mime_type = "dataset"
        if mime_type not in self.types:
            open_anything = False
            for t in self.types:
                if t == "dataset":  # datasets are special skipping
                    continue
                if len(self.types[t]) == 0:
                    open_anything = True
            if not open_anything:
                raise RuntimeError("will not be able to load object of type {mime_type}".format(mime_type=mime_type))
        self.obj = obj

    def known_types(self):
        """known_types list types of Kosh objects it can handle

        :return: list of Kosh type it understands
        :rtype: list
        """
        return list(self.types.keys())

    def known_load_formats(self, atype):
        """known_load_formats list all the formats it knows how to export to

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
        :param **kargs: key=value style argmunets
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

    def get(self, feature, format=None, transformers=[],
            use_cache=True, cache_file_only=False, cache_dir=None, **kargs):
        """get extract a feature
        *args and **kargs will be stored on loader object
        format and feature are stored on the object for extraction by extraction functions
        This function calls first the loader's preprocess function
        This is followed by an actual data extraction via the 'extract' function
        Finally 'postprocess' is called on the extracted data

        Reserved keyword:
        preprocess: function use to preprocess (default to self.preprocess)
        postprocess: function use to postprocess (default to self.postprocess)
        batch: to return data as a generator (not necessarily implemented yet)
        shuffle: to shuffle the data, we recommend True/False (not necessarily implemented yet)

        Hints: clustering and such maybe implemented in pre and postprocess

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
        :return: extracted feature
        :rtype: ???
        """
        if cache_dir is None:
            cache_dir = kosh_cache_dir
        self.cache_dir = cache_dir
        # first let's get the execution path
        path = get_path(self.obj.mime_type, self, transformers, format)
        frmt = path[1][0]
        if frmt is None:
            frmt = self.types[self.obj.mime_type][0]
        if len(self.types) != 0 and frmt not in self.types[self.obj.mime_type]:
            raise ValueError("Loader cannot output type {self.obj.mime_type} to {format} format".format(
                self=self, format=format))
        self.format = frmt
        self.feature = feature
        self._user_passed_parameters = (None, kargs)
        signature = self.update_signature(feature, self.format, **kargs).hexdigest()
        if cache_file_only is True:
            # Ok user just wants to know where cache should be (plus/minus extesions)
            return signature
        # Let's generate the signatures for each step of the path.
        # And try to load it
        signatures = [signature, ]
        for i, p in enumerate(path[1:-1], start=1):
            signatures.append(p[1].update_signature(signatures[-1], path[i-1][0]).hexdigest())

        cache_success = False
        if use_cache:
            for i, p in enumerate(path[-2:0:-1]):
                try:
                    data = p[1].load(signatures[len(signatures)-i-1])
                    cache_success = True
                    for j, p in enumerate(path[-i-1:-1], start=len(signatures)-i):
                        try:
                            data = p[1].transform_(data, path[j+1][0], signature=signatures[j])
                        except Exception:
                            pass
                except Exception:
                    pass
            if cache_success:
                return data

        kargs.get("preprocess", self.preprocess)()
        data = self.extract()
        data = kargs.get("postprocess", self.postprocess)(data)
        for i, p in enumerate(path[1:-1], start=1):
            # Get the transformer and tell it to return it
            # in format that next transformer wants
            # the last item is the output it only has the format
            data = p[1].transform_(data, path[i+1][0], signature=signatures[i])
        return data

    def save(self, cache_file, content):
        """Pickle some data to a cache file
        :param cache_file: name of cache file, will be joined with self.cache_dir
        :type cache_file: str
        :param content: content to save to cache
        :type content: object
        """
        with open(os.path.join(self.cache_dir, cache_file), "wb") as f:
            pickle.dump(content, f)

    def load(self, cache_file):
        """loads content from cache
        :param cache_file: name of cache file, will be joined with self.cache_dir
        :type cache_file: str
        :return: unpickled data
        :rtpye: object
        """
        with open(os.path.join(self.cache_dir, cache_file), "rb") as f:
            data = pickle.load(f)
        return data

    def list_features(self):
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
        """preprocess sets things up for te extract function

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

    def __init__(self, obj):
        super(KoshFileLoader, self).__init__(obj)

    def open(self, mode='r'):
        """open/load the matching Kosh Sina File

        :param mode: mode to open the file in, defaults to 'r'
        :type mode: str, optional
        :return: Kosh File object
        """
        return KoshGenericObjectFromFile(self.obj.uri, mode)

    def extract(self, feature, format):
        """extract return a feature from the loaded object.

        :param feature: variable to read from file
        :type feature: str
        :param format: desired output format
        :type format: str
        :return: data
        """
        with open(self.obj.uri) as f:
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

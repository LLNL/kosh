try:
    import h5py

    class KoshHDF5Object(h5py.File):
        def get(self, feature, *args, **kargs):
            """KoshHDF5Object Kosh hdf5 file repr

            :param feature: variable to access in hdf5 file
            :type feature: str
            :return: data
            :rtype: numpy.ndarray
            """
            return self[feature]

    has_hdf5 = True
except ImportError:
    has_hdf5 = False


class KoshGenericObjectFromFile(object):
    def __init__(self, *args, **kwds):
        self.args = args
        self.kwds = kwds
        self.file_obj = open(*self.args, **self.kwds)

    def __enter__(self):
        self.file_obj = open(*self.args, **self.kwds)
        return self.file_obj

    def __exit__(self, *args):
        self.file_obj.close()

    def get(self, feature, *args, **kargs):
        return self.file_obj.read()


class KoshLoader(object):
    def __init__(self, obj, types={"dataset": []}):
        """KoshLoader generic Kosh loader
        :param obj: object
        :param types: types is a dictionary on known type that can be loaded
        as key and export format as value, defaults to {"dataset": []}
        :type types: dict, optional
        """
        if obj.mime_type not in types:
            open_anything = False
            for t in types:
                if t == "dataset":  # datasets are special skipping
                    continue
                if len(types[t]) == 0:
                    open_anything = True
            if not open_anything:
                raise RuntimeError(f"will not be able to load object of type {obj.mime_type}")
        self.types = types
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
        return self.types.get(format, [])

    def open(self, mode="r"):
        return self

    def get(self, feature, format=None, *args, **kargs):
        """get extract a feature
        *args and **kargs will be stored on loader object
        format and feature are stored on the object for extraction by extraction functions
        This function calls first the loader's preprocess function
        This is followed by an actual data extraction via the 'extract' function
        Finally 'postprocess' is called on the extracted data

        Reserved keyword:
        batch: to return data as a generator
        shuffle: to shuffle the data, we recommend True/False

        Hints: clustering and such maybe implemented in pre and postprocess

        :param feature: desired feature
        :type feature: str
        :param format: desired output format
        :type format: str
        :return: extracted feature
        """
        if format is None:
            format = self.types[self.obj.mime_type][0]
        if len(self.types) != 0 and format not in self.types[self.obj.mime_type]:
            raise ValueError(f"Loader cannot output type {self.obj.mime_type} to {format} format")
        self.format = format
        self.feature = feature
        self._user_passed_parameters = args, kargs
        self.preprocess()
        data = self.extract(feature, format)
        return self.postprocess(data)

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
    def __init__(self, obj, types={"file": [], "hdf5": ["numpy", ]}):
        super(KoshFileLoader, self).__init__(obj, types)

    def open(self, mode='r'):
        """open/load the matching Kosh SIna File

        :param mode: mode to open the file in, defaults to 'r'
        :type mode: str, optional
        :return: Kosh File object
        """
        if self.obj.mime_type == "hdf5" and has_hdf5:
            return KoshHDF5Object(self.obj.uri, mode)
        else:
            return KoshGenericObjectFromFile(self.obj.uri, mode)

    def extract(self, feature, *args, **kargs):
        """extract return a feature from the loaded object.

        :param feature: variable to read from file
        :type feature: str
        :return: data
        """
        if self.obj.mime_type == "hdf5" and has_hdf5:
            f = h5py.File(self.obj.uri, "r")
            return f[feature]
        else:
            with open(self.obj.uri) as f:
                return f.read(*args, **kargs)

    def list_features(self, *args):
        """list_features list features in file,
        for hdf5 you can pass extra argument to navigate groups.

        :return: list of features available in file
        :rtype: list
        """
        if self.obj.mime_type == "hdf5" and has_hdf5:
            with h5py.File(self.obj.uri, "r") as f:
                keys = []
                if len(args) == 0:
                    for k in f.keys():
                        if hasattr(f[k], "keys"):
                            for k2 in f[k].keys():
                                keys.append(f"{k}/{k2}")
                        else:
                            keys.append(k)
                    return keys
                else:
                    return list(f[args[0]].keys())
        else:
            return []

    def describe_feature(self, feature):
        """describe a feature

        :param feature: feature (variable) to read, defaults to None
        :type feature: str, optional if loader does not require this
        :return: dictionary describing the feature
        :rtype: dict
        """
        if feature not in self.list_features():
            raise ValueError(f"feature {feature} is not available")
        info = {}
        if self.obj.mime_type == "hdf5" and has_hdf5:
            with h5py.File(self.obj.uri, "r") as f:
                feature = f[feature]
                info["size"] = feature.shape
                info["format"] = "hdf5"
                info["type"] = feature.dtype
                if hasattr(feature, "dims"):
                    dims = []
                    for d in feature.dims.keys():
                        specs = {}
                        specs["name"] = d.label
                        try:
                            specs["first"] = f[d.label][0]
                            specs["last"] = f[d.label][-1]
                            specs["length"] = len(f[d.label])
                        except Exception:
                            pass
                        dims.append(specs)
                    info["dimensions"] = dims
        else:
            return {}

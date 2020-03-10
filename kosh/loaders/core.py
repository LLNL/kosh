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
    """
    :param types: types is a dictionary on known type that can be loaded
    as key and export format as value, defaults to {"dataset": []}
    :type types: dict, optional
    """
    types = {"dataset": []}

    def __init__(self, obj):
        """KoshLoader generic Kosh loader
        :param obj: object
        """
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
                raise RuntimeError(f"will not be able to load object of type {mime_type}")
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
        return self

    def get(self, feature, format=None, *args, **kargs):
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
        :return: extracted feature
        """
        if format is None:
            format = self.types[self.obj.mime_type][0]
        if len(self.types) != 0 and format not in self.types[self.obj.mime_type]:
            raise ValueError(f"Loader cannot output type {self.obj.mime_type} to {format} format")
        self.format = format
        self.feature = feature
        self._user_passed_parameters = args, kargs
        kargs.get("preprocess", self.preprocess)()
        data = self.extract(feature, format)
        return kargs.get("postprocess", self.postprocess)(data)

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
    types = {"file": []}

    def __init__(self, obj):
        super(KoshFileLoader, self).__init__(obj)

    def open(self, mode='r'):
        """open/load the matching Kosh SIna File

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

    def list_features(self, *args):
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
            raise ValueError(f"feature {feature} is not available")
        return {}

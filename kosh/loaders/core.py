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

    def open(self):
        return self

    def get(self, feature, *args, **kargs):
        """return a feature from the loaded object."""
        raise NotImplementedError

    def list_features(self):
        return []


class KoshFileLoader(KoshLoader):
    def __init__(self, obj, types={"file": []}):
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

    def get(self, feature, *args, **kargs):
        """get return a feature from the loaded object.

        :param feature: variable to read from file
        :type feature: str
        :return: data
        """
        if self.obj.mime_type == "hdf5" and has_hdf5:
            with h5py.File(self.obj.uri) as f:
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

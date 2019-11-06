try:
    import h5py

    class KoshHDF5Object(h5py.File):
        def get(self, feature, *args, **kargs):
            return self[feature]

        def list_features(self):
            return self.keys()

    has_hdf5 = True
except ImportError:
    has_hdf5 = False


class KoshGenericObjectFromFile(object):
    def __init__(self, *args, **kwds):
        self.args = args
        self.kwds = kwds

    def __enter__(self):
        self.file_obj = open(*self.args, **self.kwds)
        return self.file_obj

    def __exit__(self, *args):
        self.file_obj.close()

    def list_features(self):
        return []

    def get(self):
        return self.file_obj.read()


class KoshLoader(object):
    def __init__(self, types, store):
        """ types is a dictionary on known type that can be loaded
        as key and export format as values"""
        self.types = types
        self.__store__ = store

    def known_types(self):
        return list(self.types.keys())

    def known_export_format(self, format):
        return self.types.get(format, [])

    def open(self, Id):
        raise NotImplementedError

    def get_feature(self, Id, feature, *args, **kargs):
        """return a feature from the loaded object"""
        raise NotImplementedError


class KoshFileLoader(KoshLoader):
    def open(self, Id, mode='r'):
        obj = self.__store__._load(Id)
        if obj.mime_type == "hdf5" and has_hdf5:
            return KoshHDF5Object(obj.uri, mode)
        else:
            return KoshGenericObjectFromFile(obj.uri, mode)

    def get(self, Id, feature, *args, **kargs):
        """return a feature from the loaded object"""
        raise NotImplementedError
        file = self.open(Id)
        return file[feature](*args, **kargs)

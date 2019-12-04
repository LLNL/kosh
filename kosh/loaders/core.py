try:
    import h5py

    class KoshHDF5Object(h5py.File):
        def get(self, feature, *args, **kargs):
            return self[feature]

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

    def get(self):
        return self.file_obj.read()


class KoshLoader(object):
    def __init__(self, obj, types={"dataset": []}):
        """ types is a dictionary on known type that can be loaded
        as key and export format as values"""
        self.types = types
        self.obj = obj

    def known_types(self):
        return list(self.types.keys())

    def known_export_format(self, format):
        return self.types.get(format, [])

    def open(self):
        return self

    def get(self, feature, *args, **kargs):
        """return a feature from the loaded object"""
        raise NotImplementedError

    def list_features(self):
        return []


class KoshFileLoader(KoshLoader):
    def __init__(self, obj, types={"file": ["numpy"]}):
        super(KoshFileLoader, self).__init__(obj, types)

    def open(self, mode='r'):
        if self.obj.mime_type == "hdf5" and has_hdf5:
            return KoshHDF5Object(self.obj.uri, mode)
        else:
            return KoshGenericObjectFromFile(self.obj.uri, mode)

    def get(self, feature, *args, **kargs):
        """return a feature from the loaded object"""
        if self.obj.mime_type == "hdf5" and has_hdf5:
            with h5py.File(self.obj.uri) as f:
                return f[feature]
        else:
            with open(self.obj.uri) as f:
                return f.read(*args, **kargs)

    def list_features(self, *args):
        """ List features in file, for hdf5 you can pass extra argument to
navigate groups"""
        if self.obj.mime_type == "hdf5" and has_hdf5:
            with h5py.File(self.obj.uri) as f:
                if len(args) == 0:
                    return f.keys()
                else:
                    return f[args[0]].keys()
        else:
            return []

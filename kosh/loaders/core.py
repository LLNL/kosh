class KoshLoader(object):
    def __init__(self, types):
        """ types is a dictionary on known type that can be loaded
        as key and export format as values"""
        self.types = types

    def known_types(self):
        return list(self.types.keys())

    def known_export_format(self, format):
        return self.types.get(format, [])

    def loadFromStore(self, Id, *args, **kargs):
        raise RuntimeError("Not Implemented Yet")

    def open(self, Id):
        raise RuntimeError("Not Implemented Yet")

    def get(self, Id, *args, **kargs):
        file = self.open(Id)
        return file(*args, **kargs)


class KoshFileLoader(KoshLoader):
    def open(self, Id):
        import h5py
        record = self.loadFromStore(Id)
        if record.type == "hdf5":
            return h5py.File(record.uri)
        else:
            return open(record.uri)

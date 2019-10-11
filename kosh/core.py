# Core module for our Kosh data access
class KoshAgent(object):
    """Class to manage permissions etc..."""


class KoshStoreClass(object):
    agent = KoshAgent()
    def connect(self):
        """Connect to engine DB"""
        raise NotImplementedError()

    def search(self):
        """search datasets"""
        raise NotImplementedError()

    def open(self):
        """open dataset(s)"""
        raise NotImplementedError()

    def create(self):
        """return publisher object"""
        raise NotImplementedError()
    def add_loader(self, loader):
        self.loaders.append(loader)

class KoshData(object):
    def registerReader(self, name, reader):
        """adds a loader type """
        self.__readers__[name] = {reader}

    def get(self, type=None):
        """Method to get data"""
        raise NotImplementedError()

    def __repr__(self):
        """pretty print"""
        raise NotImplementedError()

class KoshFile(KoshData):
    def open(self, mode="r"):
        return open(self.path, mode)
    def __str__(self):
        st=""
        st += "\nKOSH FILE\n"
        st += "\tid: {}\n".format(self.__id__)
        st += "\tpath: {}\n".format(self.path)
        st += "\ttype: {}\n".format(self.type)
        atts = self.__attributes__
        if len(atts) > 0:
            st += "\n--- Attributes ---\n"
            for a in sorted(atts):
                st += "\t{}: {}\n".format(a, atts[a])
        return st

class KoshHDF5File(KoshFile):
    def open(self):
        import h5py
        return h5py.File(self.path)


class KoshArray(KoshData):
    def __init__(self, dimensions):
        self.__dimensions__ = dimensions
        self.__readers__ = {}
    

def KoshStore(engine, *args, **kargs):
    known_engines = ["cassandra", "sina"]
    if not engine.lower() in known_engines:
        raise RuntimeError("Unknown engine type {}, supported engines: {}".format(engine, self.known_engines))
    # Initialize and returns access class
    if engine.lower() == "cassandra":
        from .cassandra import KoshStoreCassandra
        return KoshStoreCassandra(*args, **kargs)
    elif engine.lower() == "sina":
        from .sina import KoshStoreSina
        return KoshStoreSina(*args, **kargs)

class KoshDataset(object):
    def __repr__(self):
        """pretty print"""
        raise NotImplementedError()
    def __str__(self):
        st=""
        st += "KOSH DATASET\n"
        st += "\tid: {}\n".format(self.__id__)
        st += "\tname:{}\n".format(self.__name__)
        st += "\tcreator: {}\n".format(self.__creator__)
        atts = self.__attributes__
        if len(atts) > 0:
            st += "\n--- Attributes ---\n"
            for a in sorted(atts):
                if a == "associated_data":
                    continue
                st += "\t{}: {}\n".format(a, atts[a])
        if self.associated_data is not None:
            st += "--- Associated Data ---\n"
            for a in self.associated_data:
                st2 = str(self.loadFromStore(a))
                st += "\n\t".join(st2.split("\n"))
        return st
    
    def add(self, source):
        """ Add data to datset"""
        if self.associated_data is None:
            self.associated_data = [source.__id__,]
        elif not source.__id__ in self.associated_data:
            self.associated_data.append(source.__id__)

    def loadFromStore(self, Id, loader=None):
        """ Get an object """
        return self.__store__.loadFromStore(Id, loader)

class KoshLoader(object):
    def __init__(self, types):
        """ types is a dictionary on known type that can be loaded as key and export format as values"""
        self.types = types
        print("NOT HERE!")
    def known_types(self):
        return list(self.types.keys())
    def known_export_format(self, format):
        return self.types.get(format, [])
    def loadFromStore(self, target_type, *args, **kargs):
        raise RuntimeError("Not Implemented Yet")

class KoshFileLoader(KoshLoader):
    def open(self, Id):
        record = self.loadFromStore(Id)
        if record.type == "hdf5":
                return h5py.File(record.path)
        else:
            return open(record.path)
    def get(self, Id, *args, **kargs):
        file = self.open(Id)
        return file(*args, **kargs)

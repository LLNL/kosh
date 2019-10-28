# Core module for our Kosh data access
from .loaders import KullLoader, KoshLoader


class KoshAgent(object):
    """Class to manage permissions etc..."""


class KoshStoreClass(object):
    def __init__(self):
        self.loaders = []
        self.storeLoader = KoshLoader({"dataset": []})
        self.add_loader(KullLoader(self))

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

    def schema(self, schema_name):
        return NotImplementedError("method not implemented yet")


class KoshData(object):
    def registerReader(self, name, reader):
        """adds a loader type """
        self.__readers__[name] = {reader}

    def get(self, type=None):
        """Method to get data"""
        raise NotImplementedError()


class KoshFile(KoshData):
    def open(self, mode="r"):
        if self.type == "hdf5":
            import h5py
            return h5py.File(self.uri, mode)
        else:
            return open(self.uri, mode)

    def __str__(self):
        st = ""
        st += "\nKOSH FILE\n"
        st += "\tid: {}\n".format(self.__id__)
        st += "\turi: {}\n".format(self.uri)
        st += "\ttype: {}\n".format(self.type)
        atts = self.__attributes__
        if len(atts) > 0:
            st += "\n--- Attributes ---\n"
            for a in sorted(atts):
                st += "\t{}: {}\n".format(a, atts[a])
        return st


class KoshArray(KoshData):
    def __init__(self, dimensions):
        self.__dimensions__ = dimensions


def KoshStore(engine, *args, **kargs):
    known_engines = ["cassandra", "sina"]
    if not engine.lower() in known_engines:
        raise RuntimeError(
            "Unknown engine type {}, supported engines: {}".format(
                engine, known_engines))
    # Initialize and returns access class
    if engine.lower() == "cassandra":
        from .cassandra import KoshStoreCassandra
        return KoshStoreCassandra(*args, **kargs)
    elif engine.lower() == "sina":
        from .sina import KoshSinaStore
        return KoshSinaStore(*args, **kargs)


class KoshDataset(object):
    def __str__(self):
        st = ""
        st += "KOSH DATASET\n"
        st += "\tid: {}\n".format(self.__id__)
        st += "\tname:{}\n".format(self.__name__)
        st += "\tcreator: {}\n".format(self.__creator__)
        atts = self.__attributes__
        if len(atts) > 0:
            st += "\n--- Attributes ---\n"
            for a in sorted(atts):
                if a == "__associated_data__":
                    continue
                st += "\t{}: {}\n".format(a, atts[a])
        if self.__associated_data__ is not None:
            st += "--- Associated Data ({})---\n".format(
                len(self.__associated_data__))
            for a in self.__associated_data__:
                st2 = str(self.load(a))
                st += "\n\t".join(st2.split("\n"))
        return st

    def add(self, source):
        """ Add data to datset"""
        print("In associated data:", self.__associated_data__)
        if self.__associated_data__ is None:
            self.__associated_data__ = [source.__id__, ]
        elif source.__id__ not in self.__associated_data__:
            self.__associated_data__ += [source.__id__, ]
        print("In associated data (end):", self.__associated_data__)

    def load(self, Id, loader=None):
        """ Get an object from store"""
        return self.__store__.load(Id, loader)

    def open(self, Id, loader=None):
        """ Open an object from store"""
        return self.__store__.open(Id, loader)

    # def get(self, Id, loader=None, *args, **kargs):
    #    """ Open an object from store"""
    #    return self.__store__.get(Id, loader=loader, *args, **kargs)

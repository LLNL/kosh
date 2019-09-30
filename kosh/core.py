# Core module for our Kosh data access
class KoshAgent(object):
    """Class to manage permissions etc..."""


class KoshStoreBaseClass(object):
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

    def publish(self):
        """return publisher object"""
        raise NotImplementedError()


class KoshDataBaseClass(object):
    def get(self, type=None):
        """Method to get data"""
        raise NotImplementedError()

    def __repr__(self):
        """pretty print"""
        raise NotImplementedError()


class KoshArrayBaseClass(KoshDataBaseClass):
    def __init__(self, dimensions):
        self.__dimensions__ = dimensions
        self.__readers__ = {}
    
    def __repr__(self):
        """pretty print"""
        raise NotImplementedError()

    def registerReader(self, name, reader):
        """adds a loader type """
        self.__readers__[name] = {reader}


def KoshStore(engine, *args, **kargs):
    known_engines = ["cassandra"]
    if not engine.lower() in known_engines:
        raise RuntimeError("Unknown engine type {}, supported engines: {}".format(engine, self.known_engines))
    # Initialize and returns access class
    if engine.lower() == "cassandra":
        from .cassandra import KoshStoreCassandra
        return KoshStoreCassandra(*args, **kargs)

class KoshDatasetBaseClass(object):
    def __repr__(self):
        """pretty print"""
        raise NotImplementedError()

class KoshLoader(object):
    def load(self, target_type, *args, **kargs):
        raise RuntimeError("Not Implemented Yet")
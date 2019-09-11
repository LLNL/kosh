# Core module for our DKosh data access
class DKoshStoreBaseClass(object):
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


def DKoshStore(engine, *args, **kargs):
    known_engines = ["cassandra"]
    if not engine.lower() in known_engines:
        raise RuntimeError("Unknown engine type {}, supported engines: {}".format(engine, self.known_engines))
    # Initialize and returns access class
    if engine.lower() == "cassandra":
        from .cassandra import DKoshStoreCassandra
        return DKoshStoreCassandra(*args, **kargs)

class DKoshDatasetBaseClass(object):
    def __repr__(self):
        """pretty print"""
        raise NotImplementedError()

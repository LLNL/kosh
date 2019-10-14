import uuid
from kosh.core import KoshStoreClass,  KoshDataset, KoshArray
from kosh.core import KoshFile, KoshHDF5File, KoshFileLoader, KoshLoader

class KoshSinaObject(object):
    def __init__(self, Id, koshType, protected, record_handler):
        self.__dict__["__record_handler__"] = record_handler
        self.__dict__["__protected__"] = [
            "__id__", "__type__", "__protected__", "__record_handler__"] + protected
        self.__dict__["__id__"] = Id
        self.__dict__["__type__"] = koshType

    def __getattr__(self, name):
        if name in self.__dict__["__protected__"]:
            return self.__dict__[name]
        record = self.__record_handler__.get(self.__id__)
        if name == "__attributes__":
            return self.__getattributes__()
        if not name in record["data"]:
            raise RuntimeError(
                "Object {} does not have {} attribute".format(self.__id__, name))
        return record["data"][name]["value"]

    def __setattr__(self, name, value):
        if name in self.__protected__:
            self.__dict__[name] = value
            return
        record = self.__record_handler__.get(self.__id__)
        record["data"][name]={"value": value}
        self.__record_handler__.delete(self.__id__)
        self.__record_handler__.insert(record)

    def __delattr__(self, name):
        if name in self.__protected__:
            return
        record = self.__record_handler__.get(self.__id__)
        del(record["data"][name])
        self.__record_handler__.delete(self.__id__)
        self.__record_handler__.insert(record)
    def listattributes(self):
        record = self.__record_handler__.get(self.__id__)
        return list(record["data"].keys())
    def __getattributes__(self):
        record = self.__record_handler__.get(self.__id__)
        attributes = {}
        for a in record["data"]:
            attributes[a] = record["data"][a]["value"]
        return attributes


class KoshDatasetSina(KoshSinaObject, KoshDataset):
    def __init__(self,datasetId, record_handler, store):
        KoshSinaObject.__init__(self, datasetId, "dataset",
                                    protected=["__name__", "__creator__", "__store__"],
                                    record_handler=record_handler)
        record = self.__record_handler__.get(self.__id__)
        self.__creator__=record["data"]["creator"]["value"]
        self.__name__= record["data"]["name"]["value"]
        self.__store__ = store
    
    def add_file(self, path, filetype, metadata={}):
        """ Add a file as a source of data for this dataset
        required: path and type of file
        optional: metadata"""
        if filetype == "hdf5":
            self.add(KoshHDF5SinaFile(path=path, filetype="hdf5", metadata=metadata, record_handler=self.__record_handler__))
        else:
            self.add(KoshSinaFile(path=path, filetype=filetype, metadata=metadata, record_handler=self.__record_handler__))

class KoshSinaFile(KoshSinaObject, KoshFile):
    def __init__(self, Id=None, path="", filetype=None, metadata={}, record_handler=None):
        if Id is None:
            Id = uuid.uuid1().hex
            record = Record(id=Id, type="file")
            if path == "":
                raise RuntimeError("You need to pass a path")
            record.add_data("path", path)
            record.add_data("type", filetype)
            record_handler.insert(record)
        else:
            try:
                record = record_handler.get(Id)
            except:
                record = Record(id=Id, type="file")
                if path == "":
                    raise RuntimeError("You need to pass a path")
                record.add_data("path", path)
                record.add_data("type", filetype)
                record_handler.insert(record)
        files_filter = record_handler.get_all_of_type("file", ids_only=True)


        KoshSinaObject.__init__(self, Id, "file",
                                    protected=["path", "type"],
                                    record_handler=record_handler)
        self.path = path
        self.type = filetype


class KoshSinaHDF5File(KoshHDF5File, KoshSinaFile):
    pass

class KoshSinaLoader(KoshLoader):
    def __init__(self, types, record_handler):
        """ types is a dictionary on known type that can be loaded as key and export format as values"""
        self.types = types
        self.__record_handler = record_handler

    def loadFromStore(self, Id,*args, **kargs):
        record = self.__record_handler.get(Id)
        if record["type"] == "dataset":
            return KoshDatasetSina(datasetId, record_handler=self.__record_handler)
        else:
            raise RuntimeError("Cannot load a {} yet".format(record["type"]))

class KoshSinaFileLoader(KoshFileLoader, KoshSinaLoader):
    def __init__(self, types, record_handler):
        self.types = types
        self.__record_handler = record_handler

    def loadFromStore(self, Id,*args, **kargs):
        record = self.__record_handler.get(Id)
        if record["type"] == "file":
            if record["data"]["type"] == "hdf5":
                return KoshHDF5SinaFile(Id=Id, filetype="hdf5", record_handler=self.__record_handler)
            else:
                return KoshSinaFile(Id=Id, filetype=record["data"]["type"]["value"], record_handler=self.__record_handler)
        elif record["type"] != "file":
            raise RuntimeError("Cannot load record of type {}".format(record["type"]))



        
class KoshStoreSina(KoshStoreClass):
    def __init__(self, username, sql='sql', db_path=None, node_ip_list=["192.168.64.8",], keyspace=None):
        if sql == "sql":
            import sina.datastores.sql as sina
            self.__factory = sina.DAOFactory(db_path=db_path)
        elif sql == 'cass':
            import sina.datastores.cass as sina
            self.__factory = sina.DAOFactory(keyspace=keyspace, node_ip_list=node_ip_list)
        from sina.model import Record
        from sina.utils import DataRange
        global Record, DataRange
        self.__record_handler = self.__factory.create_record_dao()
        users_filter = self.__record_handler.get_all_of_type("user", ids_only=True)
        names_filter = self.__record_handler.data_query(username=username)
        inter_recs = set(users_filter).intersection(set(names_filter))
        if len(inter_recs) == 0:
            raise RuntimeError("Unknown user: {}".format(username))
        elif len(inter_recs) > 1:
            raise RuntimeError("Internal errors, more than one user match!")
        self.__user_id__ = list(inter_recs)[0]
        self.loaders = []
        self.add_loader(KoshSinaFileLoader({"file": ["numpy", "binary"]}, self.__record_handler))


    def create(self, name=None, datasetId=None, metadata={}):
        """create a new (possibly named) dataset"""
        if name is None:
            name="Unnamed Dataset"
        if datasetId is None:
            Id = uuid.uuid1().hex
        else:
            if datsetId in self.__record_handler.get_all_of_type("dataset", ids_only=True):
                raise RuntimeError("Dataset id {} already exists".format(datasetId))
            Id = datasetId
        ds=Record(id=Id, type="dataset")
        ds.add_data("creator", self.__user_id__)
        ds.add_data("name", name)
        ds.add_data("associated_data", None)
        for k in metadata:
            ds.add_data(k, metadata[k])
        self.__record_handler.insert(ds)
        ds=KoshDatasetSina(Id, record_handler=self.__record_handler, store=self)
        return ds

    def open(self, Id, loader=None):
        """returns an associated source to a specific format, possibly via a specified loader"""
        record = self.__record_handler.get(Id)
        if loader is None:
            for l in self.loaders:
                if record["type"] in l.known_types():
                    return l.open(Id)
        else:
            return loader.open(Id)

    def loadFromStore(self, Id, loader=None):
        """returns an associated source to a specific format, possibly via a specified loader"""
        record = self.__record_handler.get(Id)
        if loader is None:
            for l in self.loaders:
                if record["type"] in l.known_types():
                        return l.loadFromStore(Id)
        else:
            return loader.loadFromStore(Id)

    def get(self, Id, format=None, loader=None, *args, **kargs):
        """returns an associated source"""
        record = self.__record_handler.get(Id)
        if loader is None:
            for l in self.loaders:
                if record["type"] in l.known_types():
                    if format is None or format in l.known_export_formats():
                        return l.get(Id, format, *args, **kargs)
        else:
            return loader.get(Id, format, *args, **kargs)


    def search(self, *atts, **keys):
        """ Search cassandra for datasets matching some metadata
        arguments are the metadata name we are looking for e.g search("attr1", "attr2") 
        you can further restrict by specifying exact value for a metadata via key=value
        """
        sina_kargs = {}
        #for att in atts:
        #    sina_kargs[att] = DataRange(min=None, max=None)
        sina_kargs.update(keys)

        match = self.__record_handler.data_query(**sina_kargs)
        ds_filter = self.__record_handler.get_all_of_type("dataset", ids_only=True)
        inter_recs = set(match).intersection(set(ds_filter))
        for rec in inter_recs:
            yield self.open(rec)

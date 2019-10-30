import uuid
from kosh.core import KoshStoreClass, KoshDataset
from kosh.core import KoshFile
from kosh.loaders import KoshFileLoader, KoshLoader


class KoshSinaObject(object):
    def __init__(self, Id, koshType, protected, record_handler):
        self.__dict__["__record_handler__"] = record_handler
        self.__dict__["__protected__"] = [
            "__id__", "__type__", "__protected__",
            "__record_handler__", "__store__"] + protected
        self.__dict__["__id__"] = Id
        self.__dict__["__type__"] = koshType

    def __getattr__(self, name):
        if name in self.__dict__["__protected__"]:
            return self.__dict__[name]
        record = self.__record_handler__.get(self.__id__)
        if name == "__attributes__":
            return self.__getattributes__()
        if name not in record["data"]:
            raise AttributeError(
                "Object {} does not have {} attribute".format(self.__id__,
                                                              name))
        return record["data"][name]["value"]

    def __setattr__(self, name, value):
        if name in self.__protected__:  # Cannot set protected attributes
            # self.__dict__[name] = value
            return
        record = self.__record_handler__.get(self.__id__)
        record["data"][name] = {"value": value}
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
        attributes = list(record["data"].keys())
        for att in self.__protected__:
            if att in attributes:
                attributes.remove(att)
        return sorted(attributes)

    def __getattributes__(self):
        record = self.__record_handler__.get(self.__id__)
        attributes = {}
        for a in record["data"]:
            attributes[a] = record["data"][a]["value"]
        return attributes


class KoshSinaFile(KoshSinaObject, KoshFile):
    def __init__(self, Id=None, uri="", mimetype=None,
                 metadata={}, store=None):
        if Id is None:
            Id = uuid.uuid1().hex
            record = Record(id=Id, type="file")
            if uri == "":
                raise RuntimeError("You need to pass a uri to the data")
            record.add_data("uri", uri)
            record.add_data("type", mimetype)
            store.__record_handler__.insert(record)
        else:
            try:
                record = store.__record_handler__.get(Id)
            except BaseException:
                record = Record(id=Id, type="file")
                if uri == "":
                    raise RuntimeError("You need to pass a uri to the data")
                record.add_data("uri", uri)
                record.add_data("type", mimetype)
                store.__record_handler__.insert(record)

        KoshSinaObject.__init__(self, Id, "file",
                                protected=[],
                                record_handler=store.__record_handler__)
        self.__dict__["__store__"] = store


class KoshSinaDataset(KoshSinaObject, KoshDataset):
    def __init__(self, datasetId, store):
        KoshSinaObject.__init__(self, datasetId, "dataset",
                                protected=[
                                    "__name__", "__creator__", "__store__",
                                    "__associated_data__"],
                                record_handler=store.__record_handler__)
        record = store.__record_handler__.get(self.__id__)
        self.__dict__["__creator__"] = record["data"]["creator"]["value"]
        self.__dict__["__name__"] = record["data"]["name"]["value"]
        self.__dict__["__store__"] = store
        self.__dict__["__record_handler__"] = store.__record_handler__
        self.__dict__["__associated_data__"] = [record["files"]
                                                [f]["kosh_id"] for f in record["files"]]

    def add_file(self, uri, mimetype, metadata={}):
        """ Add a file as a source of data for this dataset
        required: uri and mimetype of file
        optional: metadata"""
        rec = self.__record_handler__.get(self.__id__)
        rec.add_file(uri, mimetype)
        kosh_file = KoshSinaFile(
            uri=uri,
            mimetype=mimetype,
            metadata=metadata,
            store=self.__store__)
        rec["files"][uri]["kosh_id"] = kosh_file.__id__
        self.__record_handler__.delete(self.__id__)
        self.__record_handler__.insert(rec)
        self.add(kosh_file)
        return self.__store__.load(kosh_file.__id__)

    def search(self, *atts, **keys):
        """ Search associated data matching some metadata
        arguments are the metadata name we are looking for e.g
        search("attr1", "attr2")
        you can further restrict by specifying exact value for a metadata
        via key=value
        you can return ids only by using: ids_only=True
        """
        if self.__associated_data__ is None:
            return []
        sina_kargs = {}
        ids_only = keys.pop("ids_only", False)
        for att in atts:
            sina_kargs[att] = DataRange(min=None, max=None)
        sina_kargs.update(keys)

        if len(sina_kargs) == 0:
            inter_recs = self.__associated_data__
        else:
            match = self.__record_handler__.data_query(**sina_kargs)
            # instantly restrict to associated data
            inter_recs = set(match).intersection(set(self.__associated_data__))

        if ids_only:
            return list(inter_recs)
        else:
            return [self.load(rec) for rec in inter_recs]


class KoshSinaLoader(KoshLoader):
    def __init__(self, types, store):
        """ types is a dictionary on known type that can be loaded
        as key and export format as values"""
        self.types = types
        self.store = store

    def load(self, Id, *args, **kargs):
        record = self.store.__record_handler__.get(Id)
        if record["type"] == "dataset":
            return KoshSinaDataset(Id, store=self.store)
        elif record["type"] == "file":
            return KoshSinaFile(
                Id, store=self.store, uri=record["data"]["uri"]["value"],
                mimetype=record["data"]["type"]["value"])
        else:
            return KoshSinaObject(Id, record["type"], protected=[
            ], record_handler=self.store.__record_handler__)

    def open(self, Id, *args, **kargs):
        return self.load(Id, *args, **kargs)


class KoshSinaFileLoader(KoshFileLoader, KoshSinaLoader):
    def __init__(self, types, store):
        self.types = types
        self.__dict__["__store__"] = store
        self.__dict__["__record_handler__"] = store.__record_handler__

    def load(self, Id, *args, **kargs):
        record = self.__record_handler__.get(Id)
        if record["type"] == "file":
            return KoshSinaFile(Id=Id, uri=record["data"]["uri"]["value"],
                                mimetype=record["data"]["type"]["value"],
                                store=self.__store__)
        elif record["type"] != "file":
            raise RuntimeError(
                "Cannot load record of type {}".format(
                    record["type"]))


class KoshSinaStore(KoshStoreClass):
    def __init__(self, username, sql='sql', db_path=None,
                 node_ip_list=["192.168.64.8", ], keyspace=None):
        KoshStoreClass.__init__(self)
        if sql == "sql":
            import sina.datastores.sql as sina
            self.__factory = sina.DAOFactory(db_path=db_path)
        elif sql == 'cass':
            import sina.datastores.cass as sina
            self.__factory = sina.DAOFactory(
                keyspace=keyspace, node_ip_list=node_ip_list)
        from sina.model import Record
        from sina.utils import DataRange
        global Record, DataRange
        self.__dict__["__record_handler__"] = self.__factory.create_record_dao()
        users_filter = self.__record_handler__.get_all_of_type(
            "user", ids_only=True)
        names_filter = self.__record_handler__.data_query(username=username)
        inter_recs = set(users_filter).intersection(set(names_filter))
        if len(inter_recs) == 0:
            raise RuntimeError("Unknown user: {}".format(username))
        elif len(inter_recs) > 1:
            raise RuntimeError("Internal errors, more than one user match!")
        self.__user_id__ = list(inter_recs)[0]
        self.storeLoader = KoshSinaLoader({"dataset": []}, self)
        self.add_loader(self.storeLoader)
        self.add_loader(KoshSinaFileLoader(
            {"file": ["numpy", "binary"]}, self))

    def create(self, name=None, datasetId=None, metadata={}):
        """create a new (possibly named) dataset"""
        if name is None:
            name = "Unnamed Dataset"
        if datasetId is None:
            Id = uuid.uuid1().hex
        else:
            if datasetId in self.__record_handler__.get_all_of_type(
                    "dataset", ids_only=True):
                raise RuntimeError(
                    "Dataset id {} already exists".format(datasetId))
            Id = datasetId
        ds = Record(id=Id, type="dataset")
        ds.add_data("creator", self.__user_id__)
        ds.add_data("name", name)
        ds.add_data("__associated_data__", None)
        for k in metadata:
            ds.add_data(k, metadata[k])
        self.__record_handler__.insert(ds)
        ds = KoshSinaDataset(Id, store=self)
        return ds

    def open(self, Id, loader=None):
        """returns an associated source to a specific format,
        possibly via a specified loader"""
        record = self.__record_handler__.get(Id)
        if loader is None:
            # sometime types have subtypes (e.g 'file') let's look if we
            # understand a subtype
            if "type" in record["data"]:
                for l in self.loaders:
                    if record["data"]["type"]["value"] in l.known_types():
                        return l.open(Id)
            # Ok could not open the actual subtype, looking at generic type
            for l in self.loaders:
                if record["type"] in l.known_types():
                    return l.open(Id)
        else:
            return loader.open(Id)

    def load(self, Id, loader=None):
        """returns an associated source to a specific format,
        possibly via a specified loader"""
        record = self.__record_handler__.get(Id)
        if loader is None:
            # sometime types have subtypes (e.g 'file') let's look if we
            # understand a subtype
            if "type" in record["data"]:
                for l in self.loaders:
                    if record["data"]["type"]["value"] in l.known_types():
                        return l.load(Id)
            # Ok could not open the actual subtype, looking at generic type
            for l in self.loaders:
                if record["type"] in l.known_types():
                    return l.load(Id)
        else:
            return loader.load(Id)

    def get(self, Id, format=None, loader=None, *args, **kargs):
        """returns an associated source"""
        record = self.__record_handler__.get(Id)
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
        you can return ids only by using: ids_only=True
        """  # noqa
        sina_kargs = {}
        ids_only = keys.pop("ids_only", False)
        # Until fix in sina
        if len(atts) != 0:
            raise NotImplementedError("Need key/value at the moment")
        for att in atts:
            sina_kargs[att] = DataRange(min=-1.e99999, max=1.e99999)
        sina_kargs.update(keys)

        ds_filter = self.__record_handler__.get_all_of_type(
            "dataset", ids_only=True)
        if len(sina_kargs) != 0:  # no restriction, all datsets
            match = self.__record_handler__.data_query(**sina_kargs)
            inter_recs = set(match).intersection(set(ds_filter))
        else:
            inter_recs = list(ds_filter)

        if ids_only:
            return list(inter_recs)
        else:
            return [self.open(rec) for rec in inter_recs]

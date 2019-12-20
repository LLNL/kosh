import uuid
from kosh.core import KoshStoreClass, KoshDataset
from kosh.loaders import KoshLoader
import warnings


class KoshSinaObject(object):
    """KoshSinaObject Base class for sina objects
    """
    def __init__(self, Id, store, koshType,
                 record_handler, protected=[], metadata={}):
        """__init__ sina object base class

        :param Id: id to use forunique identification, if None is passed set for you via uui4()
        :type Id: str
        :param store: Kosh store associated
        :type store: KoshSinaStore
        :param koshType: type of Kosh object (dataset, file, project, ...)
        :type koshType: str
        :param record_handler: sina record handler object
        :type record_handler: RecordDAO
        :param protected: list of protected parameters, e.g internal params not to be stored
        :type protected: list, optional
        :param metadata: dictionary of attributes/value to initialize object with, defaults to {}
        :type metadata: dict, optional
        """
        if Id is None:
            Id = uuid.uuid4().hex
            record = Record(id=Id, type="file")
            store.__record_handler__.insert(record)
        else:
            try:
                record = store.__record_handler__.get(Id)
            except BaseException:
                record = Record(id=Id, type="file")
                store.__record_handler__.insert(record)

        self.__dict__["__record_handler__"] = record_handler
        self.__dict__["__protected__"] = [
            "__id__", "__type__", "__protected__",
            "__record_handler__", "__store__"] + protected
        self.__dict__["__id__"] = Id
        self.__dict__["__type__"] = koshType
        self.__dict__["__store__"] = store
        for att, value in metadata.items():
            setattr(self, att, value)

    def __getattr__(self, name):
        """__getattr__ get an attribute

        :param name: attribute to retrieve
        :type name: str
        :raises AttributeError: could not retireve attribute
        :return: requested attribute value
        """
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
        """__setattr__ set an attribute on an object

        :param name: name of attribute
        :type name: str
        :param value: value to set attribute to
        """
        if name in self.__protected__:  # Cannot set protected attributes
            # self.__dict__[name] = value
            return
        record = self.__record_handler__.get(self.__id__)
        record["data"][name] = {"value": value}
        self.__record_handler__.delete(self.__id__)
        self.__record_handler__.insert(record)

    def __delattr__(self, name):
        """__delattr__ deletes an attribute

        :param name: attribute to delete
        :type name: str
        """
        if name in self.__protected__:
            return
        record = self.__record_handler__.get(self.__id__)
        del(record["data"][name])
        self.__record_handler__.delete(self.__id__)
        self.__record_handler__.insert(record)

    def listattributes(self):
        """listattributes list all non protected attributes

        :return: list of attributes set on object
        :rtype: list
        """
        record = self.__record_handler__.get(self.__id__)
        attributes = list(record["data"].keys())
        for att in self.__protected__:
            if att in attributes:
                attributes.remove(att)
        return sorted(attributes)

    def __getattributes__(self):
        """__getattributes__ return dictionary with pairs of attribute/value

        :return: dictionary with pairs of attribute/value
        :rtype: dict
        """
        record = self.__record_handler__.get(self.__id__)
        attributes = {}
        for a in record["data"]:
            attributes[a] = record["data"][a]["value"]
        return attributes


class KoshSinaFile(KoshSinaObject):
    """KoshSinaFile file representation in Kosh via Sina"""
    def open(self):
        """open opens the file
        :return: handle to file in open mode
        """
        return self.__store__.open(self.__id__)


class KoshSinaDataset(KoshSinaObject, KoshDataset):
    def __init__(self, datasetId, store):
        """KoshSinaDataset Sina representation of Kosh Dataset

        :param datasetId: dataset's unique Id
        :type datasetId: str
        :param store: store containing the dataset
        :type store: KoshSinaStore
        """
        KoshSinaObject.__init__(self, datasetId, koshType="dataset",
                                protected=[
                                    "__name__", "__creator__", "__store__",
                                    "__associated_data__"],
                                record_handler=store.__record_handler__,
                                store=store)
        record = store.__record_handler__.get(self.__id__)
        self.__dict__["__creator__"] = record["data"]["creator"]["value"]
        self.__dict__["__name__"] = record["data"]["name"]["value"]
        self.__dict__["__record_handler__"] = store.__record_handler__
        self.__dict__["__associated_data__"] = [record["files"]
                                                [f]["kosh_id"] for f in
                                                record["files"]]

    def add_file(self, uri, mime_type, metadata={}):
        """add_file associates a file with dataset

        :param uri: uri to access file
        :type uri: str
        :param mime_type: mime type associated with this file
        :type mime_type: str
        :param metadata: metadata to associate with file, defaults to {}
        :type metadata: dict, optional
        :return: A Kosh Sina File
        :rtype: KoshSinaFile
        """

        rec = self.__record_handler__.get(self.__id__)
        rec.add_file(uri, mime_type)
        kosh_file = KoshSinaObject(Id=None,
                                   koshType="file",
                                   store=self.__store__,
                                   metadata=metadata,
                                   record_handler=self.__record_handler__)
        kosh_file.uri = uri
        kosh_file.mime_type = mime_type
        rec["files"][uri]["kosh_id"] = kosh_file.__id__
        self.__record_handler__.delete(self.__id__)
        self.__record_handler__.insert(rec)
        self.add(kosh_file)
        return kosh_file

    def search(self, *atts, **keys):
        """search associated data matching some metadata
        arguments are the metadata name we are looking for e.g
        search("attr1", "attr2")
        you can further restrict by specifying exact value for a metadata
        via key=value
        you can return ids only by using: ids_only=True
        range can be specified via: sina.utils.DataRange(min, max)

        :return: list of matching objects associated with dataset
        :rtype: list
        """

        if self.__associated_data__ is None:
            return []
        sina_kargs = {}
        ids_only = keys.pop("ids_only", False)
        for att in atts:
            sina_kargs[att] = DataRange(min=-9.e999999)
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
            return [self.__store__._load(rec) for rec in inter_recs]


class KoshSinaLoader(KoshLoader):
    def __init__(self, obj, types={"dataset": []}):
        """KoshSinaLoader generic sina-based loader

        :param types: types the loader can handle and the output format it can produce, defaults to {"dataset": []}
        :type types: dict, optional
        """
        super(KoshSinaLoader, self).__init__(obj, types)

    def open(self, *args, **kargs):
        """open the object
        """
        record = self.obj.__store__.__record_handler__.get(self.obj.__id__)
        if record["type"] == "dataset":
            return KoshSinaDataset(self.obj.__id__, store=self.obj.__store__)
        if record["type"] == "file":
            return KoshSinaFile(self.obj.__id__, store=self.obj.__store__)
        else:
            return KoshSinaObject(self.obj.__id__, record["type"], protected=[
            ], record_handler=self.obj.__store__.__record_handler__)


class KoshSinaStore(KoshStoreClass):
    def __init__(self, username, db='sql', db_uri=None,
                 keyspace=None):
        """__init__ initialize a new Sina-based store

        :param username: user name
        :type username: str
        :param db: type of database, defaults to 'sql', can be 'cass'
        :type db: str, optional
        :param db_uri: uri to sql file or list of cassandra node ips, defaults to None
        :type db_uri: str or list, optional
        :param keyspace: cassandra keyspace, defaults to None
        :type keyspace: str, optional
        :raises ConnectionRefusedError: Could not connect to cassandra
        :raises SystemError: more than one user match.
        """
        KoshStoreClass.__init__(self)
        if db == "sql":
            import sina.datastores.sql as sina
            self.__factory = sina.DAOFactory(db_path=db_uri)
        elif db == 'cass':
            import sina.datastores.cass as sina
            self.__factory = sina.DAOFactory(
                keyspace=keyspace, node_ip_list=db_uri)
        from sina.model import Record
        from sina.utils import DataRange
        global Record, DataRange
        self.__dict__["__record_handler__"] = self.__factory.create_record_dao()
        users_filter = list(self.__record_handler__.get_all_of_type(
            "user", ids_only=True))
        names_filter = list(self.__record_handler__.data_query(username=username))
        inter_recs = set(users_filter).intersection(set(names_filter))
        if len(inter_recs) == 0:
            # raise ConnectionRefusedError("Unknown user: {}".format(username))
            # For now just letting anyone log in as anonymous
            warnings.warn("Unknown user, you will be logged as anonymous user")
            names_filter = self.__record_handler__.data_query(username="anonymous")
            inter_recs = set(users_filter).intersection(set(names_filter))
        elif len(inter_recs) > 1:
            raise SystemError("Internal error, more than one user match!")
        self.__user_id__ = list(inter_recs)[0]
        self.storeLoader = KoshSinaLoader
        self.add_loader(self.storeLoader)

    def create(self, name="Unnamed Dataset", datasetId=None, metadata={}):
        """create a new (possibly named) dataset

        :param name: name for the dataset, defaults to None
        :type name: str, optional
        :param datasetId: unique Id, defaults to None which means use uuid4()
        :type datasetId: str, optional
        :param metadata: dictionary of attribute/value pair for the dataset, defaults to {}
        :type metadata: dict, optional
        :raises RuntimeError: Dataset already exists
        :return: KoshSinaDataset
        :rtype: KoshSinaDataset
        """
        if datasetId is None:
            Id = uuid.uuid4().hex
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

    def _find_loader(self, Id):
        """_find_loader returns a loader that can open Id

        :param Id: Id of the object to load
        :type Id: str
        :return: Kosh object
        """
        record = self.__record_handler__.get(Id)
        obj = self._load(Id)
        # sometime types have subtypes (e.g 'file') let's look if we
        # understand a subtype
        if "mime_type" in record["data"]:
            for ld in self.loaders:
                ld = ld(obj)
                if record["data"]["mime_type"]["value"] in ld.known_types():
                    return ld
        # Ok could not open the actual subtype, looking at generic type
        for ld in self.loaders:
            ld = ld(obj)
            if record["type"] in ld.known_types():
                return ld

    def open(self, Id, loader=None):
        """open loads an object in store based on its Id
        and run its open function

        :param Id: unique id of object to open
        :type Id: str
        :param loader: loader to use, defaults to None which means pick for me
        :return:
        """
        if loader is None:
            loader = self._find_loader(Id)
        else:
            loader = loader(self._load(Id))
        return loader.open()

    def _load(self, Id):
        """_load returns an associated source based on id

        :param Id: unique id in store
        :type Id: str
        :return: loaded object
        """
        record = self.__record_handler__.get(Id)
        if record["type"] == "file":
            return KoshSinaFile(Id, koshType=record["type"],
                                record_handler=self.__record_handler__,
                                store=self)
        else:
            return KoshSinaObject(Id, koshType=record["type"],
                                  record_handler=self.__record_handler__,
                                  store=self)

    def get(self, Id, format=None, loader=None, *args, **kargs):
        """get returns an associated source's data

        :param Id: Id of object to retrieve
        :type Id: str
        :param format: prefered format, defaults to None means pick for me
        :type format: str, optional
        :param loader: loader to use, defaults to None means pick for me
        :return: data in requested format
        """
        if loader is None:
            loader = self._find_loader(Id)

        return loader(self._load(Id)).get(format, *args, **kargs)

    def search(self, *atts, **keys):
        """search store for objects matching some metadata
        arguments are the metadata name we are looking for e.g
        search("attr1", "attr2")
        you can further restrict by specifying exact value for a metadata
        via key=value
        you can return ids only by using: ids_only=True
        range can be specified via: sina.utils.DataRange(min, max)

        :return: list of matching objects in store
        :rtype: list
        """
        sina_kargs = {}
        ids_only = keys.pop("ids_only", False)
        # Until fix in sina
        if len(atts) != 0:
            raise NotImplementedError("Need key/value at the moment")
        for att in atts:
            sina_kargs[att] = DataRange(min=-9.e999999)
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

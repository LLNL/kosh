import uuid
from kosh.core import KoshStoreClass, KoshDataset
from kosh.loaders import KoshLoader
import warnings
import time
import sina.datastores.sql as sina_sql


class KoshSinaObject(object):
    """KoshSinaObject Base class for sina objects
    """
    def get_record(self):
        return self.__store__.get_record(self.__id__)

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
        self.__dict__["__store__"] = store
        self.__dict__["__record_handler__"] = record_handler
        self.__dict__["__protected__"] = [
            "__id__", "__type__", "__protected__",
            "__record_handler__", "__store__", "__id__"] + protected
        self.__dict__["__type__"] = koshType
        if Id is None:
            Id = uuid.uuid4().hex
            record = Record(id=Id, type=koshType)
            if store.__sync__:
                store.__record_handler__.insert(record)
            else:
                record["user_defined"]["last_update_from_db"] = time.time()
                self.__store__.__sync__dict__[Id] = record
            self.__dict__["__id__"] = Id
        else:
            self.__dict__["__id__"] = Id
            try:
                record = self.get_record()
            except BaseException:  # record exists nowhere
                record = Record(id=Id, type=koshType)
                if store.__sync__:
                    store.__record_handler__.insert(record)
                else:
                    self.__store__.__sync__dict__[Id] = record
                    record["user_defined"]["last_update_from_db"] = time.time()

        for att, value in metadata.items():
            setattr(self, att, value)

    def __getattr__(self, name):
        """__getattr__ get an attribute

        :param name: attribute to retrieve
        :type name: str
        :raises AttributeError: could not retrieve attribute
        :return: requested attribute value
        """
        if name in self.__dict__["__protected__"]:
            if name == "_associated_data_":
                record = self.get_record()
                # print("REC IN ASSOVCIATED:", record["files"])
                return [record["files"][f]["kosh_id"] for f in record["files"]]
            else:
                return self.__dict__[name]
        record = self.get_record()
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
            return
        record = self.get_record()
        # Did it change on db since we last read it?
        last_modif_att = f"{name}_last_modified"
        try:
            # Time we last read its value
            last = self.__dict__[last_modif_att]
        except KeyError:
            last = time.time()
        try:
            # Time we last read its value
            last_db = record["user_defined"][last_modif_att]
        except KeyError:
            last_db = last
        # last time attribute was modified in db
        if last_db > last and getattr(self, name) != record["data"][name]["value"]:  # Ooopsie someone touched it!
            raise AttributeError("Attribute {} of object id {} was modified since last sync\n"
                                 "Last modified in db at: {}, value: {}\n"
                                 "You last read it at: {}, with value: {}".format(
                                     name, self.__id__,
                                     last_db, record["data"][name],
                                     last, getattr(self, name)))
        now = time.time()
        if f"{name}_last_modified" not in self.__protected__:
            self.__dict__["__protected__"] += [last_modif_att, ]
        self.__dict__[last_modif_att] = now
        record["user_defined"][last_modif_att] = now
        record["data"][name] = {"value": value}
        if self.__store__.__sync__:
            self.__record_handler__.delete(self.__id__)
            self.__record_handler__.insert(record)

    def __delattr__(self, name):
        """__delattr__ deletes an attribute

        :param name: attribute to delete
        :type name: str
        """
        if name in self.__protected__:
            return
        record = self.get_record()
        last_modif_att = f"{name}_last_modified"
        now = time.time()
        record["user_defined"][last_modif_att] = now
        del(record["data"][name])
        if self.__store__.__sync__:
            self.__record_handler__.delete(self.__id__)
            self.__record_handler__.insert(record)

    def sync(self):
        """sync this object with database"""
        self.__store__.sync([self.__id__, ])

    def listattributes(self):
        """listattributes list all non protected attributes

        :return: list of attributes set on object
        :rtype: list
        """
        record = self.get_record()
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
        record = self.get_record()
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
        super(KoshSinaDataset, self).__init__(datasetId, koshType="dataset",
                                              protected=[
                                                         "__name__", "__creator__", "__store__",
                                                         "_associated_data_"],
                                              record_handler=store.__record_handler__,
                                              store=store)
        self.__dict__["__record_handler__"] = store.__record_handler__
        record = self.get_record()
        self.__dict__["__creator__"] = record["data"]["creator"]["value"]
        self.__dict__["__name__"] = record["data"]["name"]["value"]

    def associate(self, uri, mime_type, metadata={}):
        """associates a uri/mime_type with this dataset

        :param uri: uri to access file
        :type uri: str
        :param mime_type: mime type associated with this file
        :type mime_type: str
        :param metadata: metadata to associate with file, defaults to {}
        :type metadata: dict, optional
        :return: A Kosh Sina File
        :rtype: KoshSinaFile
        """

        rec = self.get_record()
        try:
            rec.add_file(uri, mime_type)
            Id = None
        except Exception:
            # file already in there
            # Let's get the matching id
            rec = self.search(file=uri)
            print("REC FOUND:", rec, uri)
        kosh_file = KoshSinaObject(Id=Id,
                                   koshType="file",
                                   store=self.__store__,
                                   metadata=metadata,
                                   record_handler=self.__record_handler__)
        kosh_file.uri = uri
        kosh_file.mime_type = mime_type
        rec["files"][uri]["kosh_id"] = kosh_file.__id__
        if self.__store__.__sync__:
            self.__record_handler__.delete(self.__id__)
            self.__record_handler__.insert(rec)
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

        if self._associated_data_ is None:
            return []
        sina_kargs = {}
        ids_only = keys.pop("ids_only", False)
        for att in atts:
            sina_kargs[att] = DataRange(min=-9.e999999)
        sina_kargs.update(keys)

        inter_recs = self._associated_data_
        if len(sina_kargs) != 0:
            file_uri = sina_kargs.pop("file", None)
            if len(sina_kargs) == 0:
                match = inter_recs
            else:
                match = list(self.__record_handler__.data_query(**sina_kargs))
            if file_uri is not None:
                file_match = list(self.__record_handler__.get_given_document_uri(file_uri, inter_recs, True))
                match = set(match).intersection(file_match)
            # instantly restrict to associated data
            if not self.__store__.__sync__:
                mem = sina_sql.DAOFactory(db_path=":memory:")
                handler = mem.create_record_dao()
                for rec in self.__store__.__sync__dict__.values():
                    handler.insert(rec)
                if len(sina_kargs) == 0:
                    match_mem = inter_recs
                else:
                    match_mem = list(handler.data_query(**sina_kargs))
                if file_uri is not None:
                    file_match = list(handler.get_given_document_uri(file_uri, inter_recs, True))
                    match_mem = set(matc_mem).intersection(file_match)
                # check that tweaks didn't remove a possible dataset
                yank = []
                for m in match:
                    if m in self.__store__.__sync__dict__ and m not in match_mem:
                        # Ok we chaned something and it's no longer a match
                        yank.append(m)
                for y in yank:
                    match.remove(y)
                match += match_mem
            inter_recs = set(match).intersection(set(self._associated_data_))

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
        record = self.obj.__store__.get_record(self.obj.__id__)
        if record["type"] == "dataset":
            return KoshSinaDataset(self.obj.__id__, store=self.obj.__store__)
        if record["type"] == "file":
            return KoshSinaFile(self.obj.__id__, store=self.obj.__store__)
        else:
            return KoshSinaObject(self.obj.__id__, record["type"], protected=[
            ], record_handler=self.obj.__store__.__record_handler__)


class KoshSinaStore(KoshStoreClass):
    def __init__(self, username, db='sql', db_uri=None,
                 keyspace=None, sync=True):
        """__init__ initialize a new Sina-based store

        :param username: user name
        :type username: str
        :param db: type of database, defaults to 'sql', can be 'cass'
        :type db: str, optional
        :param db_uri: uri to sql file or list of cassandra node ips, defaults to None
        :type db_uri: str or list, optional
        :param keyspace: cassandra keyspace, defaults to None
        :type keyspace: str, optional
        :param sync: Does Kosh sync automatically to the db (True) or on demand (False)
        :type sync: bool
        :raises ConnectionRefusedError: Could not connect to cassandra
        :raises SystemError: more than one user match.
        """
        KoshStoreClass.__init__(self, sync)
        if db == "sql":
            self.__factory = sina_sql.DAOFactory(db_path=db_uri)
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

    def get_record(self, Id):
        if (not self.__sync__) and Id in self.__sync__dict__:
            record = self.__sync__dict__[Id]
        else:
            record = self.__record_handler__.get(Id)
            self.__sync__dict__[Id] = record
            keys = list(record["user_defined"].keys())
            for key in keys:
                if key[-14:] == "_last_modified":
                    del(record["user_defined"][key])
            record["user_defined"]["last_update_from_db"] = time.time()
        return record

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
        rec = Record(id=Id, type="dataset")
        rec.add_data("creator", self.__user_id__)
        rec.add_data("name", name)
        rec.add_data("_associated_data_", None)
        for k in metadata:
            rec.add_data(k, metadata[k])
        if self.__sync__:
            self.__record_handler__.insert(rec)
        else:
            self.__sync__dict__[Id] = rec
        ds = KoshSinaDataset(Id, store=self)
        return ds

    def _find_loader(self, Id):
        """_find_loader returns a loader that can open Id

        :param Id: Id of the object to load
        :type Id: str
        :return: Kosh object
        """
        record = self.get_record(Id)
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
        record = self.get_record(Id)
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

        ds_filter = list(self.__record_handler__.get_all_of_type(
            "dataset", ids_only=True))
        if not self.__sync__:
            mem = sina_sql.DAOFactory(db_path=":memory:")
            handler = mem.create_record_dao()
            for rec in self.__sync__dict__.values():
                handler.insert(rec)
            ds_filter += list(handler.get_all_of_type("dataset", ids_only=True))

        if len(sina_kargs) != 0:  # no restriction, all datsets
            match = list(self.__record_handler__.data_query(**sina_kargs))
            if not self.__sync__:
                match_mem = list(handler.data_query(**sina_kargs))
                # check that tweaks didn't remove a possible dataset
                yank = []
                for m in match:
                    if m in self.__sync__dict__ and m not in match_mem:
                        # Ok we chaned something and it's no longer a match
                        yank.append(m)
                for y in yank:
                    match.remove(y)
                match += match_mem
            inter_recs = set(match).intersection(set(ds_filter))
        else:
            inter_recs = list(ds_filter)

        if ids_only:
            return list(inter_recs)
        else:
            return [self.open(rec) for rec in inter_recs]

    def check_sync_conflicts(self, keys):
        """Checks if their will be sync conflicts
        :param keys: keys of objects to syncs (id/type)
        :type keys: list
        :return: dictionary of objects ids and their failing attributes
        :rtype: dict
        """
        # First pass to make sure we have no conflict
        conflicts = {}
        for key in keys:
            local_record = self.__sync__dict__[key]
            # Dataset created locally on unsynced store do not have this attribute
            last_local = local_record["user_defined"].get("last_update_from_db", -1)
            try:
                db_record = self.__record_handler__.get(key)
                for att in db_record["user_defined"]:
                    conflict = False
                    if att[-14:] != "_last_modified":
                        continue
                    name = att[:-14]
                    last_db = db_record["user_defined"][att]
                    if last_db > last_local and att in local_record["user_defined"]:
                        # Conflict
                        print("Pot conflict")
                        if name not in local_record["data"]:  # deleted locally
                            if name in db_record["data"]:
                                print("deleted locally")
                                conflict = True
                        else:
                            if name not in db_record["data"]:
                                print("Gone from db")
                                conflict = True
                            elif db_record["data"][name]["value"] != local_record["data"][name]["value"]:
                                print("Values differ")
                                conflict = True
                        if conflict:
                            conf = {name: (db_record["data"].get(name, {"value": "deleted"})["value"],
                                        last_db,
                                        local_record["data"].get(name, {"value": "deleted"})["value"],
                                        local_record["user_defined"][att])}
                            if key not in conflicts:
                                conflicts[key] = conf
                            else:
                                conflicts[key].update(conf)
                            conflicts[key]["last_check_from_db"] = last_local
            except BaseException:  # It's a new record no conflict
                pass
        return conflicts

    def sync(self, keys=None):
        """Sync with db
        :param keys: keys of objects to sync (id/type)
        :type keys: list
        :return: None
        :rtype: None
        """
        if self.__sync__:
            return
        if keys is None:
            keys = self.__sync__dict__.keys()
        if len(keys) == 0:
            return
        conflicts = self.check_sync_conflicts(keys)
        if len(conflicts) != 0:  # Conflicts, aborting
            msg = "Conflicts exist objects have been modified in db and locally"
            for key in conflicts:
                msg += "\nObject id:{}".format(key)
                msg += "\n\tLast read from db: {}".format(conflicts[key]["last_check_from_db"])
                for k in conflicts[key]:
                    if k == "last_check_from_db":
                        continue
                    st = "\n\t"+k+" modified to value '{}' at {} in db, modified locally to '{}' at {}"
                    st = st.format(*conflicts[key][k])
                    msg += st
            raise RuntimeError(msg)
        # Ok no conflict we still need to sync
        update_records = []
        del_keys = []
        for key in keys:
            local = self.__sync__dict__[key]
            try:
                db = self.__record_handler__.get(key)
                for att in local["user_defined"]:
                    if att[-14:] == "_last_modified":  # We touched it
                        name = att[:-14]
                        if name not in local["data"]:  # we deleted it
                            if name in db["data"]:
                                del(db["data"][name])
                        elif local["user_defined"][att] > db["user_defined"][att]:
                            db["data"][name] = local["data"][name]
                            db["user_defined"][att] = local["user_defined"][att]
                update_records.append(db)
                del_keys.append(key)
            except Exception as err:
                print("Not in del because:", err)
                update_records.append(local)
        self.__record_handler__.delete(del_keys)
        self.__record_handler__.insert(update_records)
        for key in list(keys):
            del(self.__sync__dict__[key])

    def __del__(self):
        """Delete a Kosh store, we make sure we sync before we go"""
        #self.sync()

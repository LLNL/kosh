import uuid
from kosh.core import KoshStoreClass, KoshDataset
from kosh.schema import KoshSchema
from kosh.loaders import KoshLoader
import warnings
import time
import sina.datastores.sql as sina_sql
import pickle
import os


class KoshSinaObject(object):
    """KoshSinaObject Base class for sina objects
    """
    def get_record(self):
        return self.__store__.get_record(self.__id__)

    def __init__(self, Id, store, koshType,
                 record_handler, protected=[], metadata={}, schema=None,
                 record=None):
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
        :param record: sina record to prevent looking it up again and again in sina
        :type record: Record
        """
        self.__dict__["__store__"] = store
        self.__dict__["__schema__"] = schema
        self.__dict__["__record_handler__"] = record_handler
        self.__dict__["__protected__"] = [
            "__id__", "__type__", "__protected__",
            "__record_handler__", "__store__", "__id__", "__schema__"] + protected
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
            if record is None:
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
                return [record["files"][f]["kosh_id"] for f in record["files"]]
            else:
                return self.__dict__[name]
        record = self.get_record()
        if name == "__attributes__":
            return self.__getattributes__()
        elif name == "schema":
            if self.__dict__["__schema__"] is None and "schema" in record["data"]:
                schema = pickle.loads(record["data"]["schema"]["value"].encode("latin1"))
                self.__dict__["__schema__"] = schema
            return self.__dict__["__schema__"]
        if name not in record["data"]:
            if name == "mime_type":
                return record["type"]
            else:
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
        if name == "schema":
            assert(isinstance(value, KoshSchema))
            value.validate(self)
        elif self.schema is not None:
            self.schema.validate_attribute(name, value)

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
        if name == "schema":
            self.__dict__["__schema__"] = value
            value = pickle.dumps(value).decode("latin1")
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
    def open(self, *args, **kargs):
        """open opens the file
        :return: handle to file in open mode
        """
        return self.__store__.open(self.__id__, *args, **kargs)


class KoshSinaDataset(KoshSinaObject, KoshDataset):
    def __init__(self, datasetId, store, schema=None, record=None):
        """KoshSinaDataset Sina representation of Kosh Dataset

        :param datasetId: dataset's unique Id
        :type datasetId: str
        :param store: store containing the dataset
        :type store: KoshSinaStore
        :param schema: Kosh schema validator
        :type schema: KoshSchema
        :param record: to avoid looking up in sina pass sina record
        :type record: Record
        """
        super(KoshSinaDataset, self).__init__(datasetId, koshType="dataset",
                                              protected=[
                                                         "__name__", "__creator__", "__store__",
                                                         "_associated_data_"],
                                              record_handler=store.__record_handler__,
                                              store=store, schema=schema, record=record)
        self.__dict__["__record_handler__"] = store.__record_handler__
        if record is None:
            record = self.get_record()
        self.__dict__["__creator__"] = record["data"]["creator"]["value"]
        self.__dict__["__name__"] = record["data"]["name"]["value"]
        if schema is not None or "schema" in record["data"]:
            self.validate()

    def validate(self):
        if self.schema is not None:
            self.schema.validate(self)

    def deassociate(self, uri):
        """deassociates a uri/mime_type with this dataset

        :param uri: uri to access file
        :type uri: str
        :return: None
        :rtype: None
        """
        rec = self.get_record()
        if uri not in rec["files"]:
            # Not associated with this uri anyway
            return
        kosh_id = rec["files"][uri]["kosh_id"]
        del(rec["files"][uri])
        now = time.time()
        rec["user_defined"][f"{uri}___associated_last_modified"] = now
        if self.__store__.__sync__:
            self.__record_handler__.delete(rec.id)
            self.__record_handler__.insert(rec)
        # Get all object that have been associated with this uri
        rec = self.__store__.get_record(kosh_id)
        if (not hasattr(rec, "associated")) or len(rec.associated) == 0:  # ok no other object is associated
            self.__store__.delete(kosh_id)

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
            existing_mime = rec["files"][uri]["mimetype"]
            if existing_mime != mime_type:
                raise ValueError("file {} is already associated with this dataset with mimetype"
                                 " '{}' you specified mime_type '{}'".format(uri, existing_mime, mime_type))
            else:
                Id = rec["files"][uri]["kosh_id"]

        kosh_file = KoshSinaObject(Id=Id,
                                   koshType="file",
                                   store=self.__store__,
                                   metadata=metadata,
                                   record_handler=self.__record_handler__,
                                   record=rec)
        kosh_file.uri = uri
        kosh_file.mime_type = mime_type
        rec["files"][uri]["kosh_id"] = kosh_file.__id__
        # Need to remember we touched associated files
        now = time.time()
        rec["user_defined"][f"{uri}___associated_last_modified"] = now
        if hasattr(kosh_file, "associated"):
            st = set(kosh_file.associated)
            st.add(self.__id__)
            kosh_file.associated = list(st)
        else:
            kosh_file.associated = [self.__id__, ]
        if self.__store__.__sync__:
            self.__record_handler__.delete(self.__id__)
            self.__record_handler__.insert(rec)
        else:
            self.__store__._added_unsync_handler.delete(self.__id__)
            self.__store__._added_unsync_handler.insert(rec)

        return kosh_file

    def search(self, *atts, **keys):
        """search associated data matching some metadata
        arguments are the metadata name we are looking for e.g
        search("attr1", "attr2")
        you can further restrict by specifying exact value for a metadata
        via key=value
        you can return ids only by using: ids_only=True
        range can be specified via: sina.utils.DataRange(min, max)

        "file" is a special key that will return the kosh object associated
        with this dataset for the given uri.  e.g store.search(file=uri)

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
            # instantly restrict to associated data
            if not self.__store__.__sync__:
                if len(sina_kargs) == 0:
                    match_mem = inter_recs
                else:
                    match_mem = list(self.__store__._added_unsync_handler.data_query(**sina_kargs))
                # if file_uri is not None:
                #     match_mem = set(match_mem).intersection(file_match)
                # check that tweaks didn't remove a possible dataset
                yank = []
                for m in match:
                    if m in self.__store__.__sync__dict__ and m not in match_mem:
                        # Ok we chaned something and it's no longer a match
                        yank.append(m)
                for y in yank:
                    match.remove(y)
                match += match_mem
            if file_uri is not None:
                rec = self.get_record()
                files = rec["files"].keys()
                if file_uri in files:
                    match = [rec["files"][file_uri]["kosh_id"], ]
                else:
                    match = []
            inter_recs = set(match).intersection(set(self._associated_data_))

        if ids_only:
            return list(inter_recs)
        else:
            return [self.__store__._load(rec) for rec in inter_recs]


class KoshSinaLoader(KoshLoader):
    types = {"dataset": []}

    def __init__(self, obj):
        """KoshSinaLoader generic sina-based loader
        """

        super(KoshSinaLoader, self).__init__(obj)

    def open(self, *args, **kargs):
        """open the object
        """
        record = self.obj.__store__.get_record(self.obj.__id__)
        if record["type"] == "dataset":
            return KoshSinaDataset(self.obj.__id__, store=self.obj.__store__, record=record)
        if record["type"] == "file":
            return KoshSinaFile(self.obj.__id__, store=self.obj.__store__, record=record)
        else:
            return KoshSinaObject(self.obj.__id__, record["type"], protected=[
            ], record_handler=self.obj.__store__.__record_handler__, record=record)


class KoshSinaStore(KoshStoreClass):
    def __init__(self, username=os.environ["USER"], db='sql', db_uri=None,
                 keyspace=None, sync=True):
        """__init__ initialize a new Sina-based store

        :param username: user name defautl to user id
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
            self.__factory = sina_sql.DAOFactory(db_path=os.path.abspath(db_uri))
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
        mem = sina_sql.DAOFactory(db_path=":memory:")
        self._added_unsync_handler = mem.create_record_dao()

    def get_record(self, Id):
        if (not self.__sync__) and Id in self.__sync__dict__:
            record = self.__sync__dict__[Id]
        else:
            record = self.__record_handler__.get(Id)
            self.__sync__dict__[Id] = record
            if not self.__sync__:  # we are not autosyncing
                keys = list(record["user_defined"].keys())
                for key in keys:
                    if key[-14:] == "_last_modified":
                        del(record["user_defined"][key])
            record["user_defined"]["last_update_from_db"] = time.time()
        return record

    def delete(self, Id):
        """remove a record from store.
        for datasets deassociate all associated data first.

        :param Id: unique Id or kosh_obj
        :type Id: str
        """
        if not isinstance(Id, str):
            Id = Id.__id__

        rec = self.get_record(Id)
        if rec.type == "dataset":
            kosh_obj = self.open(Id)
            for uri in list(rec["files"].keys()):
                # Let's deassociate to remove unused kosh objects as well
                kosh_obj.deassociate(uri)
        if not self.__sync__:
            if Id in self.__sync__dict__:
                del(self.__sync__dict__[Id])
                self.__sync__deleted__[Id] = rec
                rec["user_defined"]["deleted_time"] = time.time()
        else:
            self.__record_handler__.delete(Id)

    def create(self, name="Unnamed Dataset", datasetId=None, metadata={}, schema=None):
        """create a new (possibly named) dataset

        :param name: name for the dataset, defaults to None
        :type name: str, optional
        :param datasetId: unique Id, defaults to None which means use uuid4()
        :type datasetId: str, optional
        :param metadata: dictionary of attribute/value pair for the dataset, defaults to {}
        :type metadata: dict, optional
        :param schema: a KoshSchema object to validate datasets and when setting attributes
        :type schema: KoshSchema
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

        metadata = metadata.copy()
        metadata["creator"] = self.__user_id__
        metadata["name"] = name
        metadata["_associated_data_"] = None
        for k in metadata:
            metadata[k] = {'value': metadata[k]}
        rec = Record(id=Id, type="dataset", data=metadata)
        if self.__sync__:
            self.__record_handler__.insert(rec)
        else:
            self.__sync__dict__[Id] = rec
            self._added_unsync_handler.insert(rec)
        try:
            ds = KoshSinaDataset(Id, store=self, schema=schema, record=rec)
        except Exception as err:  # probably schema validation error
            if self.__sync__:
                self.__record_handler__.delete(Id)
            else:
                del(self.__sync__dict__[Id])
                self._added_unsync_handler.delete(rec)
            raise err
        return ds

    def _find_loader(self, Id):
        """_find_loader returns a loader that can open Id

        :param Id: Id of the object to load
        :type Id: str
        :return: Kosh object
        """
        record = self.get_record(Id)
        obj = self._load(Id)
        if record["type"] == "dataset":
            return KoshSinaLoader(obj)
        loader = None
        if "mime_type" in record["data"]:
            if record["data"]["mime_type"]["value"] in self.loaders:
                return self.loaders[record["data"]["mime_type"]["value"]][0](obj)
        # sometime types have subtypes (e.g 'file') let's look if we
        # understand a subtype since we can't figure it out from mime_type
        if record["type"] in self.loaders:  # ok not a generic loader let's use it
            return self.loaders[record["type"]][0](obj)
        return loader

    def open(self, Id, loader=None, *args, **kargs):
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
        return loader.open(*args, **kargs)

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
                                store=self, record=record)
        else:
            return KoshSinaObject(Id, koshType=record["type"],
                                  record_handler=self.__record_handler__,
                                  store=self, record=record)

    def get(self, Id, feature, format=None, loader=None, *args, **kargs):
        """get returns an associated source's data

        :param Id: Id of object to retrieve
        :type Id: str
        :param feature: feature to retrieve
        :type feature: str
        :param format: prefered format, defaults to None means pick for me
        :type format: str, optional
        :param loader: loader to use, defaults to None means pick for me
        :return: data in requested format
        """
        if loader is None:
            loader = self._find_loader(Id)
        else:
            loader = loader(self._load(Id))

        return loader.get(feature, format, *args, **kargs)

    def search(self, *atts, **keys):
        """search store for objects matching some metadata
        arguments are the metadata name we are looking for e.g
        search("attr1", "attr2")
        you can further restrict by specifying exact value for a metadata
        via key=value
        you can return ids only by using: ids_only=True
        range can be specified via: sina.utils.DataRange(min, max)

        "file" is a special key that will return all records being associated
        with the given "uri", e.g store.search(file=uri)

        :return: list of matching objects in store
        :rtype: list
        """
        mode = self.__sync__
        if mode:
            # we will not update any rec in here, turnin off sync
            # it makes things much d=faster
            backup = self.__sync__dict__
            self.__sync__dict__ = {}
            self.synchronous()
        sina_kargs = {}
        ids_only = keys.pop("ids_only", False)
        # Until fix in sina
        if len(atts) != 0:
            raise NotImplementedError("Need key/value at the moment")
        # for att in atts:
        #     sina_kargs[att] = DataRange(min=-9.e999999)
        sina_kargs.update(keys)

        ds_filter = list(self.__record_handler__.get_all_of_type(
            "dataset", ids_only=True))
        if not self.__sync__:
            ds_filter += list(self._added_unsync_handler.get_all_of_type("dataset", ids_only=True))

        file_uri = sina_kargs.pop("file", None)
        if len(sina_kargs) != 0:  # no restriction, all datasets
            match = set(self.__record_handler__.data_query(**sina_kargs))
            if not self.__sync__:
                match_mem = set(self._added_unsync_handler.data_query(**sina_kargs))
                # check that tweaks didn't remove a possible dataset
                # print(f"sync: {set(self.__sync__dict__.keys())}")
                # print(f"mem: {match_mem}")
                # yank = set(self.__sync__dict__.keys()).difference(match_mem).intersection(match)
                # print(f"ynk: {yank}")
                # for m in match:
                #    if m in self.__sync__dict__ and m not in match_mem:
                #        # Ok we chaned something and it's no longer a match
                #        yank.append(m)
                # print(f"Match: {match}")
                # for y in yank:
                #    match.remove(y)
                match = match.union(match_mem)
            inter_recs = match.intersection(set(ds_filter))
            # inter_recs = set(ds_filter)
        else:
            inter_recs = set(ds_filter)

        if file_uri is not None:
            # print("INTRERE SRESC:", inter_recs)
            file_match = list(self.__record_handler__.get_given_document_uri(file_uri, inter_recs, True))
            # print("FILE MATCH:", file_match)
            if not self.__sync__:
                file_match += list(self._added_unsync_handler.get_given_document_uri(file_uri, inter_recs, True))
                # print("FILE MATCH 2:", file_match)
            inter_recs = set(inter_recs).intersection(file_match)

        if ids_only:
            out = list(inter_recs)
        else:
            out = [self.open(rec) for rec in inter_recs]
        if mode:
            # we need to restore sync mode
            self.__sync__dict__ = backup
            self.synchronous()
        return out

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
            try:
                db_record = self.__record_handler__.get(key)
                try:
                    local_record = self.__sync__dict__[key]
                    # Dataset created locally on unsynced store do not have this attribute
                    last_local = local_record["user_defined"].get("last_update_from_db", -1)
                    for att in db_record["user_defined"]:
                        conflict = False
                        if att[-14:] != "_last_modified":
                            continue
                        last_db = db_record["user_defined"][att]
                        if last_db > last_local and att in local_record["user_defined"]:
                            # Conflict
                            if att[-27:-14] == "___associated":
                                # ok dealing with associated data
                                uri = att[:-27]
                                if uri not in local_record["files"]:  # deleted locally
                                    if uri in db_record["files"]:
                                        conflict = True
                                else:
                                    if uri not in db_record["files"]:
                                        conflict = True
                                    elif db_record["files"][uri]["mimetype"] != local_record["files"][uri]["mimetype"]:
                                        conflict = True
                                if conflict:
                                    conf = {uri: (db_record["files"].get(uri, {"mimetype": "deleted"})["mimetype"],
                                                  last_db,
                                                  local_record["files"].get(uri, {"mimetype": "deleted"})[
                                        "mimetype"],
                                        local_record["user_defined"][att])}
                                    if key not in conflicts:
                                        conflicts[key] = conf
                                    else:
                                        conflicts[key].update(conf)
                                    conflicts[key]["last_check_from_db"] = last_local
                                    conflicts[key]["type"] = "associated"
                            else:
                                name = att[:-14]
                                if name not in local_record["data"]:  # deleted locally
                                    if name in db_record["data"]:
                                        conflict = True
                                else:
                                    if name not in db_record["data"]:
                                        conflict = True
                                    elif db_record["data"][name]["value"] != local_record["data"][name]["value"]:
                                        conflict = True
                                if conflict:
                                    conf = {name: (db_record["data"].get(name, {"value": "deleted"})["value"],
                                                   last_db,
                                                   local_record["data"].get(
                                        name, {"value": "deleted"})["value"],
                                        local_record["user_defined"][att])}
                                    if key not in conflicts:
                                        conflicts[key] = conf
                                    else:
                                        conflicts[key].update(conf)
                                    conflicts[key]["last_check_from_db"] = last_local
                                    conflicts[key]["type"] = "attribute"
                except Exception:  # ok let's see if it was a delete ones
                    local_record = self.__sync__deleted[key]
                    last_local = local_record["user_defined"].get("last_update_from_db", -1)
                    for att in db_record["user_defined"]:
                        conflict = False
                        if att[-14:] != "_last_modified":
                            continue
                        last_db = db_record["user_defined"][att]
                        if last_db > last_local:
                            conf = {att[:14]: (
                                "modified in db", "ds deleted here", "")}
                            if key not in conflicts:
                                conflicts[key] = conf
                            else:
                                conflicts[key].update(conf)
                            conflicts[key]["last_check_from_db"] = last_local
                            conflicts[key]["type"] = "delete"
            except BaseException:  # It's a new record no conflict
                # It could be it was deleted in store while we touched it here
                try:
                    local_record = self.__sync__dict__[key]
                    # Dataset created locally on unsynced store do not have this attribute
                    last_local = local_record["user_defined"].get("last_update_from_db", -1)
                    if last_local != -1:  # yep we read it from store
                        conf = {local_record["data"]["name"]["value"]: ("deleted in store", "", "")}
                        conf["last_check_from_db"] = last_local
                        conf["type"] = "delete"
                        if key not in conflicts:
                            conflicts[key] = conf
                        else:
                            conflicts[key].update(conf)
                except Exception:  # deleted too so no issue
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
            keys = list(self.__sync__dict__.keys()) + \
                list(self.__sync__deleted__.keys())
        if len(keys) == 0:
            return
        conflicts = self.check_sync_conflicts(keys)
        if len(conflicts) != 0:  # Conflicts, aborting
            msg = "Conflicts exist objects have been modified in db and locally"
            for key in conflicts:
                msg += "\nObject id:{}".format(key)
                msg += "\n\tLast read from db: {}".format(conflicts[key]["last_check_from_db"])
                for k in conflicts[key]:
                    if k in ["last_check_from_db", "type"]:
                        continue
                    if conflicts[key]["type"] == "attribute":
                        st = "\n\t"+k+" modified to value '{}' at {} in db, modified locally to '{}' at {}"
                    elif conflicts[key]["type"] == "delete":
                        st = "\n\t"+k+"{} {} {}"
                    else:
                        st = "\n\tfile '"+k+"' mimetype modified to'{}' at {} in db, modified locally to '{}' at {}"
                    st = st.format(*conflicts[key][k])
                    msg += st
            raise RuntimeError(msg)
        # Ok no conflict we still need to sync
        update_records = []
        del_keys = []
        for key in keys:
            try:
                local = self.__sync__dict__[key]
            except Exception:
                # Ok it comes from the deleted datasets
                del_keys.append(key)
                continue
            try:
                db = self.__record_handler__.get(key)
                for att in local["user_defined"]:
                    if att[-14:] == "_last_modified":  # We touched it
                        if att[-27:-14] == "___associated":
                            # ok it's an associated thing
                            uri = att[:-27]
                            if uri not in local["files"]:  # deassociated
                                del(db["files"][uri])
                            elif att not in db["user_defined"]:  # newly associated
                                db["files"][uri] = local["files"][uri]
                                db["user_defined"][att] = local["user_defined"][att]
                            elif local["user_defined"][att] > db["user_defined"][att]:
                                # last changed locally
                                db["files"][uri] = local["files"][uri]
                                db["user_defined"][att] = local["user_defined"][att]
                        else:
                            name = att[:-14]
                            if name not in local["data"]:  # we deleted it
                                if name in db["data"]:
                                    del(db["data"][name])
                            elif local["user_defined"][att] > db["user_defined"][att]:
                                db["data"][name] = local["data"][name]
                                db["user_defined"][att] = local["user_defined"][att]
                if db is not None:
                    update_records.append(db)
                else:  # db did not have that key and returned None (no error)
                    update_records.append(local)
                del_keys.append(key)
            except Exception:
                update_records.append(local)
        self.__record_handler__.delete(del_keys)
        self.__record_handler__.insert(update_records)
        for key in list(keys):
            try:
                self._added_unsync_handler.delete(key)
            except Exception:
                pass
            try:
                del(self.__sync__dict__[key])
            except Exception:
                # probably coming from del then
                del(self.__sync__deleted__[key])

import uuid
from kosh.core import KoshStoreClass, KoshDataset
from kosh.schema import KoshSchema
from kosh.loaders import KoshLoader
from kosh.utils import compute_fast_sha, compute_long_sha
import warnings
import time
import sina.datastores.sql as sina_sql
import sina.utils
import pickle
import os
import grp
try:
    basestring
except NameError:
    basestring = str
from sina import get_version


sina_version = float(".".join(get_version().split(".")[:2]))


class KoshSinaObject(object):
    """KoshSinaObject Base class for sina objects
    """
    def get_record(self):
        return self.__store__.get_record(self.__id__)

    def __init__(self, Id, store, koshType,
                 record_handler, protected=[], metadata={}, schema=None,
                 record=None):
        """__init__ sina object base class

        :param Id: id to use for unique identification, if None is passed set for you via uui4()
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
                store.lock()
                store.__record_handler__.insert(record)
                store.unlock()
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
                        store.lock()
                        store.__record_handler__.insert(record)
                        store.unlock()
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
        value = record["data"][name]["value"]
        if name == "creator":
            # old records have user id let's fix this
            if value in self.__store__.__record_handler__.get_all_of_type("user", ids_only=True):
                value = self.__store__.get_record(value)["data"]["username"]["value"]
        return value

    def update(self, attributes):
        """update many attributes at once to limit db writes
        :param: attributes: dictionary with attributes to update
        :type attributes: dict
        """
        rec = None
        N = len(attributes)
        n = 0
        for name, value in attributes.items():
            n += 1
            if n == N:
                update_db = True
            else:
                update_db = False
            rec = self.___setattr___(name, value, rec, update_db=update_db)

    def __setattr__(self, name, value):
        """set an attribute
        We are calling the ___setattr___
        because of special case that needs extra args and return values
        """
        self.___setattr___(name, value)

    def ___setattr___(self, name, value, record=None, update_db=True):
        """__setattr__ set an attribute on an object

        :param name: name of attribute
        :type name: str
        :param value: value to set attribute to
        :type value: object
        :param record: sina record if already extracted before, save db access
        :type record: sina.model.Record
        :return: sina record updated
        :rtype: sina.model.Record
        """
        if name in self.__protected__:  # Cannot set protected attributes
            return
        if record is None:
            record = self.get_record()
        if name == "schema":
            assert(isinstance(value, KoshSchema))
            value.validate(self)
        elif self.schema is not None:
            self.schema.validate_attribute(name, value)

        # Did it change on db since we last read it?
        last_modif_att = "{name}_last_modified".format(name=name)
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
        if "{name}_last_modified".format(name=name) not in self.__protected__:
            self.__dict__["__protected__"] += [last_modif_att, ]
        self.__dict__[last_modif_att] = now
        record["user_defined"][last_modif_att] = now
        if name == "schema":
            self.__dict__["__schema__"] = value
            value = pickle.dumps(value).decode("latin1")
        record["data"][name] = {"value": value}
        if update_db and self.__store__.__sync__:
            self.__store__.lock()
            self.__record_handler__.delete(self.__id__)
            self.__record_handler__.insert(record)
            self.__store__.unlock()
        return record

    def __delattr__(self, name):
        """__delattr__ deletes an attribute

        :param name: attribute to delete
        :type name: str
        """
        if name in self.__protected__:
            return
        record = self.get_record()
        last_modif_att = "{name}_last_modified".format(name=name)
        now = time.time()
        record["user_defined"][last_modif_att] = now
        del(record["data"][name])
        if self.__store__.__sync__:
            self.__store__.lock()
            self.__record_handler__.delete(self.__id__)
            self.__record_handler__.insert(record)
            self.__store__.unlock()

    def sync(self):
        """sync this object with database"""
        self.__store__.sync([self.__id__, ])

    def list_attributes(self, dictionary=False):
        __doc__ = self.listattributes.__doc__.replace("listattributes", "list_attributes")  # noqa
        return self.listattributes(dictionary=dictionary)

    def listattributes(self, dictionary=False):
        """listattributes list all non protected attributes

        :parm dictionary: return a dictionary of value/pair rather than just attributes names
        :type dictionary: bool

        :return: list of attributes set on object
        :rtype: list
        """
        record = self.get_record()
        attributes = list(record["data"].keys())
        for att in self.__protected__:
            if att in attributes:
                attributes.remove(att)
        if dictionary:
            out = {}
            for att in attributes:
                out[att] = getattr(self, att)
            return out
        else:
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
            if a == "creator":
                # old records have user id let's fix this
                if attributes[a] in self.__store__.__record_handler__.get_all_of_type("user", ids_only=True):
                    attributes[a] = self.__store__.get_record(attributes[a])["data"]["username"]["value"]
        return attributes

    def __str__(self):
        """String for printing"""
        st = "Id: {}".format(self.__id__)
        for att in sorted(self.listattributes()):
            st += "\n\t{}: {}".format(att, getattr(self, att))
        return st


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
        super(KoshSinaDataset, self).__init__(datasetId, koshType=store._dataset_record_type,
                                              protected=[
                                                         "__name__", "__creator__", "__store__",
                                                         "_associated_data_", "__features__"],
                                              record_handler=store.__record_handler__,
                                              store=store, schema=schema, record=record)
        self.__dict__["__record_handler__"] = store.__record_handler__
        self.__dict__["__features__"] = {None: {}}
        if record is None:
            record = self.get_record()
        try:
            self.__dict__["__creator__"] = record["data"]["creator"]["value"]
        except Exception:
            pass
        try:
            self.__dict__["__name__"] = record["data"]["name"]["value"]
        except Exception:
            pass
        if schema is not None or "schema" in record["data"]:
            self.validate()

    __str__ = KoshDataset.__str__

    def dissociate(self, uri, absolute_path=True):
        """dissociates a uri/mime_type with this dataset

        :param uri: uri to access file
        :type uri: str
        :param absolute_path: if file exists should we store its absolute_path
        :type absolute_path: bool
        :return: None
        :rtype: None
        """

        if absolute_path and os.path.exists(uri):
            uri = os.path.abspath(uri)
        rec = self.get_record()
        if uri not in rec["files"]:
            # Not associated with this uri anyway
            return
        kosh_id = str(rec["files"][uri]["kosh_id"])
        del(rec["files"][uri])
        now = time.time()
        rec["user_defined"]["{uri}___associated_last_modified".format(uri=uri)] = now
        if self.__store__.__sync__:
            self.__store__.lock()
            self.__record_handler__.delete(rec.id)
            self.__record_handler__.insert(rec)
            self.__store__.unlock()
        # Get all object that have been associated with this uri
        rec = self.__store__.get_record(kosh_id)
        if (not hasattr(rec, "associated")) or len(rec.associated) == 0:  # ok no other object is associated
            self.__store__.delete(kosh_id)
            if kosh_id in self.__store__._cached_loaders:
                del(self.__store__._cached_loaders[kosh_id])

        # Since we changed the associated, we need to cleanup
        # the features cache
        self.__dict__["__features__"][None] = {}
        self.__dict__["__features__"][kosh_id] = {}

    def associate(self, uri, mime_type, metadata={}, id_only=True, long_sha=False, absolute_path=True):
        """associates a uri/mime_type with this dataset

        :param uri: uri(s) to access content
        :type uri: str or list of str
        :param mime_type: mime type associated with this file
        :type mime_type: str or list of str
        :param metadata: metadata to associate with file, defaults to {}
        :type metadata: dict, optional
        :param id_only: do not return kosh file object, just its id
        :type id_only: bool
        :param long_sha: Do we compute the long sha on this or not?
        :type long_sha: bool
        :param absolute_path: if file exists should we store its absolute_path
        :type absolute_path: bool
        :return: A (list) Kosh Sina File(s)
        :rtype: list of KoshSinaFile or KoshSinaFile
        """

        rec = self.get_record()
        # Need to remember we touched associated files
        now = time.time()

        if isinstance(uri, basestring):
            uris = [uri, ]
            metadatas = [metadata, ]
            mime_types = [mime_type, ]
            single_element = True
        else:
            uris = uri
            if isinstance(metadata, dict):
                metadatas = [metadata, ] * len(uris)
            else:
                metadatas = metadata
            if isinstance(mime_type, basestring):
                mime_types = [mime_type, ] * len(uris)
            else:
                mime_types = mime_type
            single_element = False

        new_recs = []
        kosh_file_ids = []

        for i, uri in enumerate(uris):
            try:
                meta = metadatas[i].copy()
                if os.path.exists(uri):
                    if long_sha:
                        meta["long_sha"] = compute_long_sha(uri)
                    if absolute_path:
                        uri = os.path.abspath(uri)
                    if not os.path.isdir(uri) and "fast_sha" not in meta:
                        meta["fast_sha"] = compute_fast_sha(uri)
                rec["user_defined"]["{uri}___associated_last_modified".format(uri=uri)] = now
                # We need to check if the uri was already associated somewhere
                tmp_uris = self.__store__.search(kosh_type="file", uri=uri, ids_only=True)
                if len(tmp_uris) == 0:
                    Id = uuid.uuid4().hex
                    rec_obj = Record(id=Id, type="file")
                else:
                    rec_obj = self.__store__.get_record(tmp_uris[0])
                    Id = rec_obj.id
                    existing_mime = rec_obj["data"]["mime_type"]["value"]
                    mime_type = mime_types[i]
                    if existing_mime != mime_types[i]:
                        rec["files"][uri]["mime_type"] = existing_mime
                        raise TypeError("file {} is already associated with another dataset with mimetype"
                                        " '{}' you specified mime_type '{}'".format(uri, existing_mime, mime_types[i]))
                rec.add_file(uri, mime_types[i])

                rec["files"][uri]["kosh_id"] = Id
                meta["uri"] = uri
                meta["mime_type"] = mime_types[i]
                meta["associated"] = [self.__id__, ]
                for key in meta:
                    rec_obj.add_data(key, meta[key])
                    last_modif_att = "{name}_last_modified".format(name=key)
                    rec_obj["user_defined"][last_modif_att] = time.time()
                if not self.__store__.__sync__:
                    rec_obj["user_defined"]["last_update_from_db"] = time.time()
                    self.__store__.__sync__dict__[Id] = rec_obj
                new_recs.append(rec_obj)
            except TypeError as err:
                raise(err)
            except Exception:
                # file already in there
                # Let's get the matching id
                if rec_obj["data"]["mime_type"]["value"] != mime_types[i]:
                    raise TypeError("file {} is already associated with this dataset with mimetype"
                                    " '{}' you specified mime_type '{}'".format(uri, existing_mime, mime_type))
                else:
                    Id = rec["files"][uri]["kosh_id"]
                    if len(metadatas[i]) != 0:
                        warnings.warn(
                            "uri {} was already associated, metadata will "
                            "stay unchanged\nEdit object (id={}) directly to update attributes.".format(uri, Id))
            kosh_file_ids.append(Id)

        if self.__store__.__sync__:
            self.__store__.lock()
            self.__store__.__record_handler__.insert(new_recs)
            self.__store__.__record_handler__.delete(self.__id__)
            self.__store__.__record_handler__.insert(rec)
            self.__store__.unlock()
        else:
            self.__store__._added_unsync_handler.delete(self.__id__)
            self.__store__._added_unsync_handler.insert(rec)

        # Since we changed the associated, we need to cleanup
        # the features cache
        self.__dict__["__features__"][None] = {}

        if id_only:
            if single_element:
                return kosh_file_ids[0]
            else:
                return kosh_file_ids

        kosh_files = []
        for Id in kosh_file_ids:
            self.__dict__["__features__"][Id] = {}
            kosh_file = KoshSinaObject(Id=Id,
                                       koshType="file",
                                       store=self.__store__,
                                       metadata=metadata,
                                       record_handler=self.__record_handler__)
            kosh_files.append(kosh_file)

        if single_element:
            return kosh_files[0]
        else:
            return kosh_files

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

        warnings.warn(
            "\nIn the next version the search function will return a generator.\n"
            "You might need to wrap the result in a list.")

        if self._associated_data_ is None:
            return []
        sina_kargs = {}
        ids_only = keys.pop("ids_only", False)
        for att in atts:
            sina_kargs[att] = sina.utils.exists()
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
            return [self.__store__._load(record) for record in inter_recs]


class KoshSinaLoader(KoshLoader):
    """Sina base class for loaders"""
    types = {"dataset": []}

    def __init__(self, obj):
        """KoshSinaLoader generic sina-based loader
        """

        super(KoshSinaLoader, self).__init__(obj)

    def open(self, *args, **kargs):
        """open the object
        """
        record = self.obj.__store__.get_record(self.obj.__id__)
        if record["type"] == self.obj.__store__._dataset_record_type:
            return KoshSinaDataset(self.obj.__id__, store=self.obj.__store__, record=record)
        if record["type"] == "file":
            return KoshSinaFile(self.obj.__id__, store=self.obj.__store__, record=record)
        else:
            return KoshSinaObject(self.obj.__id__, record["type"], protected=[
            ], record_handler=self.obj.__store__.__record_handler__, record=record)


class KoshSinaStore(KoshStoreClass):
    """Sina-based implementation of Kosh store"""
    def __init__(self, username=os.environ["USER"], db='sql', db_uri=None,
                 keyspace=None, sync=True, dataset_record_type="dataset",
                 verbose=True, use_lock_file=False):
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
        :param dataset_record_type: Kosh element type is "dataset" this can change the default
                                    This is usefull if reading in other sina db
        :type dataset_record_type: str
        :param verbose: verbose message
        :type verbose: bool
        :param use_lock_file: If you receive sqlite threads access error, turning this on might help
        :type use_lock_file: bool
        :raises ConnectionRefusedError: Could not connect to cassandra
        :raises SystemError: more than one user match.
        """
        KoshStoreClass.__init__(self, sync, verbose, use_lock_file)
        self._dataset_record_type = dataset_record_type
        self.db_uri = db_uri
        if db == "sql":
            if not os.path.exists(db_uri):
                raise ValueError("Kosh store could not be found at: {}".format(db_uri))
            self.lock()
            self.__factory = sina_sql.DAOFactory(db_path=os.path.abspath(db_uri))
            self.unlock()
        elif db == 'cass':
            import sina.datastores.cass as sina
            self.__factory = sina.DAOFactory(
                keyspace=keyspace, node_ip_list=db_uri)
        from sina.model import Record
        from sina.utils import DataRange
        global Record, DataRange
        self.lock()
        self.__dict__["__record_handler__"] = self.__factory.create_record_dao()
        self.unlock()
        users_filter = list(self.__record_handler__.get_all_of_type(
            "user", ids_only=True))
        names_filter = list(self.__record_handler__.data_query(username=username))
        inter_recs = set(users_filter).intersection(set(names_filter))
        if len(inter_recs) == 0:
            # raise ConnectionRefusedError("Unknown user: {}".format(username))
            # For now just letting anyone log in as anonymous
            warnings.warn("Unknown user, you will be logged as anonymous user")
            names_filter = self.__record_handler__.data_query(username="anonymous")
            self.__user_id__ = "anonymous"
        elif len(inter_recs) > 1:
            raise SystemError("Internal error, more than one user match!")
        else:
            self.__user_id__ = list(inter_recs)[0]
        self.storeLoader = KoshSinaLoader
        self.add_loader(self.storeLoader)

        # Now let's add the loaders in the store
        for rec_loader in self.__record_handler__.get_all_of_type("koshloader"):
            pickled_code = rec_loader.data["code"]["value"].encode("latin1")
            loader = pickle.loads(pickled_code)
            self.add_loader(loader)
        if sina_version < 1.9:
            mem = sina_sql.DAOFactory(db_path=":memory:")
        else:
            mem = sina_sql.DAOFactory(db_path=None)
        self._added_unsync_handler = mem.create_record_dao()
        self._cached_loaders = {}

    def close(self):
        """closes store and sina related things"""
        self.__factory.close()

    def save_loader(self, loader):
        """Save a loader to the store
        Executed immediately even in async mode

        :param loader: Loader to save
        :type loader: KoshLoader
        """

        pickled = pickle.dumps(loader).decode("latin1")
        rec = Record(id=uuid.uuid4().hex, type="koshloader")
        rec.add_data("code", pickled)
        self.lock()
        self.__record_handler__.insert(rec)
        self.unlock()

    def get_record(self, Id):
        """Gets the sina record tied to an id
        tags record with time of last access to db
        :param Id: record id
        :type Id: str
        :return: sina record
        :rtpye: sina.model.Record
        """
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
        for datasets dissociate all associated data first.

        :param Id: unique Id or kosh_obj
        :type Id: str
        """
        if not isinstance(Id, basestring):
            Id = Id.__id__

        rec = self.get_record(Id)
        if rec.type == self._dataset_record_type:
            kosh_obj = self.open(Id)
            for uri in list(rec["files"].keys()):
                # Let's dissociate to remove unused kosh objects as well
                kosh_obj.dissociate(uri)
        if not self.__sync__:
            self._added_unsync_handler.delete(Id)
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
                    self._dataset_record_type, ids_only=True):
                raise RuntimeError(
                    "Dataset id {} already exists".format(datasetId))
            Id = datasetId

        metadata = metadata.copy()
        metadata["creator"] = self.__user_id__
        if "name" not in metadata:
            metadata["name"] = name
        metadata["_associated_data_"] = None
        for k in metadata:
            metadata[k] = {'value': metadata[k]}
        rec = Record(id=Id, type=self._dataset_record_type, data=metadata)
        if self.__sync__:
            self.lock()
            self.__record_handler__.insert(rec)
            self.unlock()
        else:
            self.__sync__dict__[Id] = rec
            self._added_unsync_handler.insert(rec)
        try:
            ds = KoshSinaDataset(Id, store=self, schema=schema, record=rec)
        except Exception as err:  # probably schema validation error
            if self.__sync__:
                self.lock()
                self.__record_handler__.delete(Id)
                self.unlock()
            else:
                del(self.__sync__dict__[Id])
                self._added_unsync_handler.delete(rec)
            raise err
        return ds

    def _find_loader(self, Id, format=None, transformers=[]):
        """_find_loader returns a loader that can open Id

        :param Id: Id of the object to load
        :type Id: str
        :return: Kosh loader object and mime_type
        """
        if Id in self._cached_loaders:
            return self._cached_loaders[Id]
        record = self.get_record(Id)
        obj = self._load(Id)
        if record["type"] == self._dataset_record_type:
            return KoshSinaLoader(obj), self._dataset_record_type
        if "mime_type" in record["data"]:
            if record["data"]["mime_type"]["value"] in self.loaders:
                self._cached_loaders[Id] = self.loaders[record["data"]["mime_type"]["value"]][0](
                    obj), record["data"]["mime_type"]["value"]
                return self._cached_loaders[Id]
        # sometime types have subtypes (e.g 'file') let's look if we
        # understand a subtype since we can't figure it out from mime_type
        if record["type"] in self.loaders:  # ok not a generic loader let's use it
            self._cached_loaders[Id] = self.loaders[record["type"]][0](obj), record["type"]
            return self._cached_loaders[Id]
        return

    def open(self, Id, loader=None, *args, **kargs):
        """open loads an object in store based on its Id
        and run its open function

        :param Id: unique id of object to open
        :type Id: str
        :param loader: loader to use, defaults to None which means pick for me
        :return:
        """
        if loader is None:
            loader, _ = self._find_loader(Id)
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

    def get(self, Id, feature, format=None, loader=None, transformers=[], *args, **kargs):
        """get returns an associated source's data

        :param Id: Id of object to retrieve
        :type Id: str
        :param feature: feature to retrieve
        :type feature: str
        :param format: preferred format, defaults to None means pick for me
        :type format: str, optional
        :param loader: loader to use, defaults to None means pick for me
        :return: data in requested format
        :param transformers: A list of transformers to use after the data is loaded
        :type transformers: kosh.operator.KoshTransformer
        """
        if loader is None:
            loader, _ = self._find_loader(Id)
        else:
            loader = loader(self._load(Id))

        return loader.get(feature, format, transformers=[], *args, **kargs)

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

        warnings.warn(
            "\nIn the next version the search function will return a generator.\n"
            "You might need to wrap the result in a list.")

        mode = self.__sync__
        if mode:
            # we will not update any rec in here, turnin off sync
            # it makes things much d=faster
            backup = self.__sync__dict__
            self.__sync__dict__ = {}
            self.synchronous()
        sina_kargs = {}
        ids_only = keys.pop("ids_only", False)
        for att in atts:
            sina_kargs[att] = sina.utils.exists()
        search_type = keys.pop("kosh_type", self._dataset_record_type)
        sina_kargs.update(keys)
        ds_filter = list(self.__record_handler__.get_all_of_type(
            search_type, ids_only=True))

        if not self.__sync__:
            ds_filter += list(self._added_unsync_handler.get_all_of_type(search_type, ids_only=True))

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
            file_match = list(self.__record_handler__.get_given_document_uri(file_uri, inter_recs, True))
            if not self.__sync__:
                file_match += list(self._added_unsync_handler.get_given_document_uri(file_uri, inter_recs, True))
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
                            if uri not in local["files"]:  # dissociated
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

    def add_user(self, username, groups=[]):
        """add_user adds a user to the Kosh store

        :param username: username to add
        :type username: str
        :param groups: kosh specific groups to add to this user
        :type groups: list
        """

        existing_users = self.__record_handler__.get_all_of_type("user")
        users = [rec["data"]["username"]["value"] for rec in existing_users]
        if username not in users:
            # Create user
            uid = uuid.uuid4().hex
            user = Record(id=uid, type="user")
            user.add_data("username", username)
            self.__record_handler__.insert(user)
            self.add_user_to_group(username, groups)
        else:
            raise ValueError("User {} already exists".format(username))

    def add_group(self, group):
        """Add a kosh specific group, cannot match existing group on unix system

        :param group: ugroup to add
        :type group: str
        """

        existing_groups = self.__record_handler__.get_all_of_type("group")
        groups_names = [rec["data"]["name"]["value"] for rec in existing_groups]
        if group in groups_names:
            raise ValueError("group {} already exist".format(group))

        # now get unix groups
        unix_groups = [g[0] for g in grp.getgrall()]
        if group in unix_groups:
            raise ValueError("{} is a unix group on this system.format(group)")

        # Create group
        uid = uuid.uuid4().hex
        group_rec = Record(id=uid, type="group")
        group_rec.add_data("name", group)
        self.__record_handler__.insert(group_rec)

    def add_user_to_group(self, username, groups):
        """Add a user to some group(s)

        :param username: username to add
        :type username: str
        :param groups: kosh specific groups to add to this user
        :type groups: list
        """

        users_filter = self.__record_handler__.get_all_of_type("user", ids_only=True)
        names_filter = list(self.__record_handler__.data_query(username=username))
        inter_recs = set(users_filter).intersection(set(names_filter))
        if len(inter_recs) == 0:
            raise ValueError("User {} does not exists".format(username))
        user = self.get_record(names_filter[0])
        user_groups = user["data"].get("groups", {"value": []})["value"]

        existing_groups = self.__record_handler__.get_all_of_type("group")
        groups_names = [rec["data"]["name"]["value"] for rec in existing_groups]
        for group in groups:
            if group not in groups_names:
                warnings.warn("Group {} is not a Kosh group, skipping".format(group))
                continue
            user_groups.append(group)
        if len(user_groups) == 0:
            user.add_data("groups", None)
        else:
            user.add_data("groups", list(set(user_groups)))
        self.__record_handler__.delete(names_filter[0])
        self.__record_handler__.insert(user)

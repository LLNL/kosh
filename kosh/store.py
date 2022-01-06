import os
import uuid
import sina.utils
from sina.model import Relationship
import fcntl
import collections
import grp
import hashlib
import warnings
import time
import pickle
from .loaders import KoshLoader, KoshFileLoader, PGMLoader, KoshSinaLoader
from .utils import compute_fast_sha, merge_datasets_handler
from .loaders import JSONLoader
from .loaders import NpyLoader
from .dataset import KoshDataset
from .ensemble import KoshEnsemble
from .core_sina import KoshSinaFile, KoshSinaObject
from .utils import create_kosh_users
from .utils import update_store_and_get_info_record
from sina.datastore import connect as sina_connect
from inspect import isfunction, ismethod
import kosh
try:
    basestring
except NameError:
    basestring = str
import types

try:
    from .loaders import HDF5Loader
except ImportError:
    pass
try:
    from .loaders import PILLoader
except ImportError:
    pass
try:
    from .loaders import UltraLoader
except ImportError:
    pass
try:
    from .loaders import SidreMeshBlueprintFieldLoader
except ImportError:
    pass
try:
    import orjson
except ImportError:
    import json as orjson  # noqa


def connect(database, keyspace=None, database_type=None,
            allow_connection_pooling=False, read_only=False,
            delete_all_contents=False, **kargs):
    """Connect to a Sina store.

Given a uri/path (and, if required, the name of a keyspace),
figures out which backend is required.

:param database: The URI of the store to connect to.
:type database: str
:param keyspace: The keyspace to connect to (Cassandra only).
:type keyspace: str
:param database_type: Type of backend to connect to. If not provided, Sina
                      will infer this from <database>. One of "sql" or
                      "cassandra".
:type database_type: str
:param allow_connection_pooling: Allow "pooling" behavior that recycles connections,
                                 which may prevent them from closing fully on .close().
                                 Only used for the sql backend.
:type allow_connection_pooling: bool
:param read_only: whether to create a read-only store
:type read_only: bool
:param kargs: Any extra arguments you wish to pass to the KoshStore function
:type kargs: dict, key=value
:param delete_all_contents: Deletes all data after opening the db
:type delete_all_contents: bool
:return: a KoshStore object connected to the specified database
:rtype: KoshStoreClass
"""
    db = kargs.pop("db", None)
    if db is not None:
        if database_type is not None and db != database_type:
            raise ValueError(
                "You cannot specify `db` and `database_type` with different values")
        database_type = db
    sina_store = sina_connect(database=database,
                              keyspace=keyspace,
                              database_type=database_type,
                              allow_connection_pooling=allow_connection_pooling,
                              read_only=read_only)
    if not read_only:
        if delete_all_contents:
            sina_store.delete_all_contents(force="SKIP PROMPT")
        update_store_and_get_info_record(sina_store.records)
        create_kosh_users(sina_store.records)
    sina_store.close()
    sync = kargs.pop("sync", True)
    if read_only:
        sync = False
    store = KoshStore(database, sync=sync, keyspace=keyspace, read_only=read_only,
                      db=database_type,
                      allow_connection_pooling=allow_connection_pooling,
                      **kargs)
    return store


class KoshStore(object):
    """Kosh store, relies on Sina"""

    def __init__(self, db_uri=None, username=os.environ.get("USER", "default"), db=None,
                 keyspace=None, sync=True, dataset_record_type="dataset",
                 verbose=True, use_lock_file=False, kosh_reserved_record_types=[],
                 read_only=False, allow_connection_pooling=False, ensemble_predicate=None):
        """__init__ initialize a new Sina-based store

        :param db: type of database, defaults to 'sql', can be 'cass'
        :type db: str, optional
        :param username: user name defaults to user id
        :type username: str
        :param db_uri: uri to sql file or list of cassandra node ips, defaults to None
        :type db_uri: str or list, optional
        :param keyspace: cassandra keyspace, defaults to None
        :type keyspace: str, optional
        :param sync: Does Kosh sync automatically to the db (True) or on demand (False)
        :type sync: bool
        :param dataset_record_type: Kosh element type is "dataset" this can change the default
                                    This is useful if reading in other sina db
        :type dataset_record_type: str
        :param verbose: verbose message
        :type verbose: bool
        :param use_lock_file: If you receive sqlite threads access error, turning this on might help
        :type use_lock_file: bool
        :param kosh_reserved_record_types: list of record types that are reserved for Kosh internal
                                           use, will be ignored when searching store
        :type kosh_reserved_record_types: list of strings
        :param read_only: Can we modify the database source?
        :type read_only: bool
        :param allow_connection_pooling: Allow "pooling" behavior that recycles connections,
                                        which may prevent them from closing fully on .close().
                                        Only used for the sql backend.
        :type allow_connection_pooling: bool
        :param ensemble_predicate: The predicate for the relationship to an ensemble
        :type ensemble_predicate: str
        :raises ConnectionRefusedError: Could not connect to cassandra
        :raises SystemError: more than one user match.
        """
        self.use_lock_file = use_lock_file
        self.loaders = {}
        self.storeLoader = KoshLoader
        self.add_loader(KoshFileLoader)
        self.add_loader(JSONLoader)
        self.add_loader(NpyLoader)
        try:
            self.add_loader(HDF5Loader)
        except Exception:  # no h5py module?
            if verbose:
                warnings.warn("Could not add hdf5 loader, check if you have h5py installed."
                              " Pass verbose=False when creating the store to turn this message off")
        try:
            self.add_loader(PILLoader)
        except Exception:  # no PIL?
            if verbose:
                warnings.warn("Could not add pil loader, check if you have pillow installed."
                              " Pass verbose=False when creating the store to turn this message off")
        self.add_loader(PGMLoader)
        try:
            self.add_loader(UltraLoader)
        except Exception:  # no pydv?
            if verbose:
                warnings.warn("Could not add ultra files loader, check if you have pydv installed."
                              " Pass verbose=False when creating the store to turn this message off")
        try:
            self.add_loader(SidreMeshBlueprintFieldLoader)
        except Exception:  # no conduit?
            if verbose:
                warnings.warn("Could not add sidre blueprint meshfield loader, check if you have conduit installed."
                              " Pass verbose=False when creating the store to turn this message off")
        if read_only:
            sync = False
        self.__read_only__ = read_only
        self.__sync__ = sync
        self.__sync__dict__ = {}
        self.__sync__deleted__ = {}

        if db is None:
            db = 'sql'
        self._dataset_record_type = dataset_record_type
        self.db_uri = db_uri
        if db == "sql":
            if not os.path.exists(db_uri):
                if ("://" in db_uri and "@" in db_uri):
                    self.__sina_store = sina_connect(
                        db_uri, read_only=read_only)
                else:
                    raise ValueError(
                        "Kosh store could not be found at: {}".format(db_uri))
            else:
                self.lock()
                self.__sina_store = sina_connect(database=os.path.abspath(db_uri),
                                                 read_only=read_only,
                                                 database_type=db,
                                                 allow_connection_pooling=allow_connection_pooling)
                self.unlock()
        elif db.lower().startswith('cass'):
            self.__sina_store = sina_connect(
                keyspace=keyspace, database=db_uri,
                database_type='cassandra', read_only=read_only,
                allow_connection_pooling=allow_connection_pooling)
        from sina.model import Record
        from sina.utils import DataRange
        global Record, DataRange

        rec = update_store_and_get_info_record(self.__sina_store.records, ensemble_predicate)

        self._sources_type = rec["data"]["sources_type"]["value"]
        self._users_type = rec["data"]["users_type"]["value"]
        self._groups_type = rec["data"]["groups_type"]["value"]
        self._loaders_type = rec["data"]["loaders_type"]["value"]
        self._ensembles_type = rec["data"]["ensembles_type"]["value"]
        self._ensemble_predicate = rec["data"]["ensemble_predicate"]["value"]
        self._kosh_reserved_record_types = kosh_reserved_record_types + \
            rec["data"]["reserved_types"]["value"]

        # Associated stores
        self._associated_stores_ = []
        if "associated_stores" in rec["data"]:
            for store in rec["data"]["associated_stores"]["value"]:
                try:
                    self._associated_stores_.append(kosh.connect(store, read_only=read_only, sync=sync))
                except Exception:  # most likely a sqlalchemy.exc.DatabaseError
                    warnings.warn("Could not open associated store: {}".format(store))

        self.lock()
        self.__dict__["__record_handler__"] = self.__sina_store.records
        self.unlock()
        users_filter = list(self.__record_handler__.find_with_type(
            self._users_type, ids_only=True))
        names_filter = list(
            self.__record_handler__.find_with_data(
                username=username))
        inter_recs = set(users_filter).intersection(set(names_filter))
        if len(inter_recs) == 0:
            # raise ConnectionRefusedError("Unknown user: {}".format(username))
            # For now just letting anyone log in as anonymous
            warnings.warn("Unknown user, you will be logged as anonymous user")
            names_filter = self.__record_handler__.find_with_data(
                username="anonymous")
            self.__user_id__ = "anonymous"
        elif len(inter_recs) > 1:
            raise SystemError("Internal error, more than one user match!")
        else:
            self.__user_id__ = list(inter_recs)[0]
        self.storeLoader = KoshSinaLoader
        self.add_loader(self.storeLoader)

        # Now let's add the loaders in the store
        for rec_loader in self.__record_handler__.find_with_type("koshloader"):
            pickled_code = rec_loader.data["code"]["value"].encode("latin1")
            loader = pickle.loads(pickled_code)
            self.add_loader(loader)
        self._added_unsync_mem_store = sina_connect(None)
        self._cached_loaders = {}

        # Ok we need to map the KoshFileLoader back to whatever the source_type is
        # in this store
        ks = self.loaders["file"]
        for loader in ks:
            loader.types[self._sources_type] = loader.types["file"]
        self.loaders[self._sources_type] = self.loaders["file"]

    def add_loader(self, loader, save=False):
        """Adds a loader to the store

        :param loader: The Kosh loader you want to add to the store
        :type loader: KoshLoader
        :param save: Do we also save it in store for later re-use
        :type save: bool

        :return: None
        :rtype: None
        """
        # We add a loader we need to clear the cache
        self._cached_loaders = collections.OrderedDict()
        for k in loader.types:
            if k in self.loaders:
                self.loaders[k].append(loader)
            else:
                self.loaders[k] = [loader, ]

        if save:  # do we save it in store
            self.save_loader(loader)

    def lock(self):
        """Attempts to lock the store, helps when many concurrent requests are made to the store"""
        if not self.use_lock_file:
            return
        locked = False
        while not locked:
            try:
                self.lock_file = open(self.db_uri + ".handle", "w")
                fcntl.lockf(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except Exception:
                time.sleep(0.1)

    def unlock(self):
        """Unlocks the store so other can access it"""
        if not self.use_lock_file:
            return
        fcntl.lockf(self.lock_file, fcntl.LOCK_UN)
        self.lock_file.close()
        # Wrapping this in a try/except
        # In case concurrency by same user
        # already removed the file
        try:
            os.remove(self.lock_file.name)
        except Exception:
            pass

    def __del__(self):
        """delete the KoshStore object"""
        if not self.use_lock_file:
            return
        name = self.lock_file.name
        self.lock_file.close()
        if os.path.exists(name):
            os.remove(name)

    def get_sina_store(self):
        """Returns the sina store object"""
        return self.__sina_store

    def get_sina_records(self):
        """Returns sina store's records"""
        return self.__record_handler__

    def close(self):
        """closes store and sina related things"""
        self.__sina_store.close()

    def delete_all_contents(self, force=""):
        """
        Delete EVERYTHING in a datastore; this cannot be undone.

        :param force: This function is meant to raise a confirmation prompt. If you
                      want to use it in an automated script (and you're sure of
                      what you're doing), set this to "SKIP PROMPT".
        :type force: str
        :returns: whether the deletion happened.
        """
        ret = self.__sina_store.delete_all_contents(force=force)
        update_store_and_get_info_record(self.__sina_store.records)
        create_kosh_users(self.__sina_store.records)
        return ret

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
            Id = Id.id

        rec = self.get_record(Id)
        if rec.type not in self._kosh_reserved_record_types:
            kosh_obj = self.open(Id)
            for uri in list(rec["files"].keys()):
                # Let's dissociate to remove unused kosh objects as well
                kosh_obj.dissociate(uri)
        if not self.__sync__:
            self._added_unsync_mem_store.records.delete(Id)
            if Id in self.__sync__dict__:
                del(self.__sync__dict__[Id])
                self.__sync__deleted__[Id] = rec
                rec["user_defined"]["deleted_time"] = time.time()
        else:
            self.__record_handler__.delete(Id)

    def create_ensemble(self, name="Unnamed Ensemble", id=None, metadata={}, schema=None, **kargs):
        """Create a Kosh ensemble object
        :param name: name for the dataset, defaults to None
        :type name: str, optional
        :param id: unique Id, defaults to None which means use uuid4()
        :type id: str, optional
        :param metadata: dictionary of attribute/value pair for the dataset, defaults to {}
        :type metadata: dict, optional
        :param schema: a KoshSchema object to validate datasets and when setting attributes
        :type schema: KoshSchema
        :param kargs: extra keyword arguments (ignored)
        :type kargs: dict
        :raises RuntimeError: Dataset already exists
        :return: KoshEnsemble
        :rtype: KoshEnsemble
        """
        return self.create(name=name, id=id, metadata=metadata, schema=schema, sina_type=self._ensembles_type, **kargs)

    def create(self, name="Unnamed Dataset", id=None,
               metadata={}, schema=None, sina_type=None, **kargs):
        """create a new (possibly named) dataset

        :param name: name for the dataset, defaults to None
        :type name: str, optional
        :param id: unique Id, defaults to None which means use uuid4()
        :type id: str, optional
        :param metadata: dictionary of attribute/value pair for the dataset, defaults to {}
        :type metadata: dict, optional
        :param schema: a KoshSchema object to validate datasets and when setting attributes
        :type schema: KoshSchema
        :param sina_type: If you want to query the store for a specific sina record type, not just a dataset
        :type sina_type: str
        :param kargs: extra keyword arguments (ignored)
        :type kargs: dict
        :raises RuntimeError: Dataset already exists
        :return: KoshDataset
        :rtype: KoshDataset
        """
        if "datasetId" in kargs:
            if id is None:
                warnings.warn(
                    "'datasetId' has been deprecated in favor of 'id'",
                    DeprecationWarning)
                id = kargs["datasetId"]
            else:
                raise ValueError(
                    "'datasetId' is deprecated in favor of 'id' which you already set here")

        if sina_type is None:
            sina_type = self._dataset_record_type
        if id is None:
            Id = uuid.uuid4().hex
        else:
            if id in self.__record_handler__.find_with_type(
                    sina_type, ids_only=True):
                raise RuntimeError(
                    "Dataset id {} already exists".format(id))
            Id = id

        metadata = metadata.copy()
        metadata["creator"] = self.__user_id__
        if "name" not in metadata:
            metadata["name"] = name
        metadata["_associated_data_"] = None
        for k in metadata:
            metadata[k] = {'value': metadata[k]}
        rec = Record(id=Id, type=sina_type, data=metadata)
        if self.__sync__:
            self.lock()
            self.__record_handler__.insert(rec)
            self.unlock()
        else:
            self.__sync__dict__[Id] = rec
            self._added_unsync_mem_store.records.insert(rec)
        try:
            if sina_type == self._ensembles_type:
                out = KoshEnsemble(Id, store=self, schema=schema, record=rec)
            else:
                out = KoshDataset(Id, store=self, schema=schema, record=rec)
        except Exception as err:  # probably schema validation error
            if self.__sync__:
                self.lock()
                self.__record_handler__.delete(Id)
                self.unlock()
            else:
                del(self.__sync__dict__[Id])
                self._added_unsync_mem_store.records.delete(rec)
            raise err
        return out

    def _find_loader(self, Id, format=None, transformers=[]):
        """_find_loader returns a loader that can open Id

        :param Id: Id of the object to load
        :type Id: str
        :return: Kosh loader object
        """
        Id_original = str(Id)
        if "__uri__" in Id:
            # Ok this is a pure sina file with mime_type
            Id, uri = Id.split("__uri__")
        else:
            uri = None
        if Id_original in self._cached_loaders:
            try:
                feats = self._cached_loaders[Id_original][0].list_features() != [
                ]
            except Exception:
                feats = []
            if feats != []:
                return self._cached_loaders[Id_original]
        record = self.get_record(Id)
        obj = self._load(Id)
        # uri not none means it is pure sina record with file and mime_type
        if (record["type"] not in self._kosh_reserved_record_types and uri is None)\
                or record["type"] in [self._ensembles_type, "__kosh_storeinfo__"]:
            # Not reserved means dataset
            return KoshSinaLoader(obj), record["type"]
        # Ok special type
        if uri is None:
            if "mime_type" in record["data"]:
                mime_type = record["data"]["mime_type"]["value"]
            else:
                mime_type = None
            mime_type_passed = mime_type
        else:  # Pure sina with file/mime_type
            mime_type = mime_type_passed = record["files"][uri]["mimetype"]
        if mime_type in self.loaders:
            for ld in self.loaders[mime_type]:
                try:
                    feats = ld(obj, mime_type=mime_type_passed, uri=uri).list_features()
                except Exception:
                    # Something happened can't list features
                    feats = []
                if feats != []:
                    break
            self._cached_loaders[Id_original] = ld(
                obj, mime_type=mime_type_passed, uri=uri), record["type"]
            return self._cached_loaders[Id_original]
        # sometime types have subtypes (e.g 'file') let's look if we
        # understand a subtype since we can't figure it out from mime_type
        if record["type"] in self.loaders:  # ok not a generic loader let's use it
            for ld in self.loaders[record["type"]]:
                try:
                    feats = ld(obj, mime_type=mime_type_passed, uri=uri).list_features()
                except Exception:
                    # Something happened can't list features
                    feats = []
                if feats != []:
                    break
            self._cached_loaders[Id_original] = ld(
                obj, mime_type=mime_type_passed, uri=uri), record["type"]
            return self._cached_loaders[Id_original]
        return None, None

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
        if record["type"] == self._sources_type:
            return KoshSinaFile(Id, kosh_type=record["type"],
                                record_handler=self.__record_handler__,
                                store=self, record=record)
        else:
            return KoshSinaObject(Id, kosh_type=record["type"],
                                  record_handler=self.__record_handler__,
                                  store=self, record=record)

    def get(self, Id, feature, format=None, loader=None,
            transformers=[], *args, **kargs):
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
        """
        Deprecated use find
        """
        warnings.warn("The 'search' function is deprecated and now called `find`.\n"
                      "Please update your code to use `find` as `search` might disappear in the future",
                      DeprecationWarning)
        return self.find(*atts, **keys)

    def find_ensembles(self, *atts, **keys):
        """Find ensembles matching some metadata in the store
        arguments are the metadata name we are looking for e.g
        find("attr1", "attr2")
        you can further restrict by specifying exact value for a metadata
        via key=value
        you can return ids only by using: ids_only=True
        range can be specified via: sina.utils.DataRange(min, max)

        :return: generator of matching ensembles in store
        :rtype: generator
        """
        return self.find(types=self._ensembles_type, *atts, **keys)

    def find(self, *atts, **keys):
        """Find objects matching some metadata in the store
        and its associated stores.

        Arguments are the metadata name we are looking for e.g
        find("attr1", "attr2")
        you can further restrict by specifying exact value for a metadata
        via key=value
        you can return ids only by using: ids_only=True
        range can be specified via: sina.utils.DataRange(min, max)

        "file_uri" is a reserved key that will return all records being associated
                   with the given "uri", e.g store.find(file_uri=uri)
        "types" let you search over specific sina record types only.

        :return: generator of matching objects in store
        :rtype: generator
        """

        for result in self._find(*atts, **keys):
            yield result

        searched_stores = [self.db_uri]
        if hasattr(self, "searched_stores"):
            self.searched_stores += list(searched_stores)
        else:
            self.searched_stores = list(searched_stores)
        for store in self._associated_stores_:
            if hasattr(store, "searched_stores"):
                if store.db_uri in store.searched_stores:
                    continue
                else:
                    store.searched_stores += list(searched_stores)
            else:
                store.searched_stores = list(searched_stores)
            searched_stores += [store.db_uri, ]
            for result in store.find(*atts, **keys):
                yield result
        # cleanup searched store uris
        for store in self._associated_stores_ + [self, ]:
            for id_ in list(searched_stores):
                if id_ in store.searched_stores:
                    store.searched_stores.remove(id_)

    def _find(self, *atts, **keys):
        """Find objects matching some metadata in the store
        arguments are the metadata name we are looking for e.g
        find("attr1", "attr2")
        you can further restrict by specifying exact value for a metadata
        via key=value
        you can return ids only by using: ids_only=True
        range can be specified via: sina.utils.DataRange(min, max)

        "file_uri" is a reserved key that will return all records being associated
                   with the given "uri", e.g store.find(file_uri=uri)
        "types" let you search over specific sina record types only.

        :return: generator of matching objects in store
        :rtype: generator
        """

        mode = self.__sync__
        if mode:
            # we will not update any rec in here, turnin off sync
            # it makes things much faster
            backup = self.__sync__dict__
            self.__sync__dict__ = {}
            self.synchronous()
        sina_kargs = {}
        ids_only = keys.pop("ids_only", False)
        # We only want to search sina for ids not records
        sina_kargs["ids_only"] = True

        if "kosh_type" in keys and "types" in keys:
            raise ValueError(
                "`kosh_type` has been replaced with `types` you cannot use both at same time")
        if "kosh_type" in keys:
            warnings.warn(
                "`kosh_type` is being deprecated in favor of `types` and will not work in a future version",
                DeprecationWarning)
            record_types = keys.pop("kosh_type")
        else:
            record_types = keys.pop("types", None)

        if isinstance(record_types, basestring):
            record_types = [record_types, ]
        if record_types is not None and not isinstance(
                record_types, (list, tuple)):
            raise ValueError("`types` must be str or list")

        if record_types is None:
            # Ok we want anything, but we need to exclude Kosh reserved
            record_types = sina.utils.not_(self._kosh_reserved_record_types)

        sina_kargs["types"] = record_types

        if 'file_uri' in keys and 'file' in keys:
            raise ValueError(
                "`file` has been deprecated for `file_uri` but you cannot use both at same time")
        if 'file' in keys:
            file_uri = keys.pop("file")
        else:
            file_uri = keys.pop("file_uri", None)

        sina_kargs["file_uri"] = file_uri
        sina_kargs["id_pool"] = keys.pop("id_pool", None)
        sina_kargs["query_order"] = keys.pop(
            "query_order", ("data", "file_uri", "types"))

        # The data dict for sina
        sina_data = keys.pop("data", {})
        if not isinstance(sina_data, dict):
            keys["data"] = sina_data
            sina_data = {}
        elif len(sina_data) != 0 and (len(keys) != 0 or len(atts) != 0):
            warnings.warn(
                "It is not recommended to use the find function by mixing keys and the reserved key `data`")

        # Maybe user is trying to get an attribute data
        for att in atts:
            sina_data[att] = sina.utils.exists()

        sina_data.update(keys)
        sina_kargs["data"] = sina_data

        # is it a blank search, e.g get me everything?
        get_all = sina_kargs.get("data", {}) == {} and \
            sina_kargs.get("file_uri", None) is None and \
            sina_kargs.get("id_pool", None) is None and \
            sina_kargs.get("types", []) == []

        if get_all:
            match = set(self.__record_handler__.get_all(ids_only=True))
        else:
            match = set(self.__record_handler__.find(**sina_kargs))

        if not self.__sync__:
            # We need to check or in memory records as well
            if get_all:
                match_mem = set(
                    self._added_unsync_mem_store.records.get_all(
                        ids_only=True))
            else:
                match_mem = set(self._added_unsync_mem_store.records.find(**sina_kargs))
            match = match.union(match_mem)

        if mode:
            # we need to restore sync mode
            self.__sync__dict__ = backup
            self.synchronous()

        for rec_id in match:
            yield rec_id if ids_only else self.open(rec_id)

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
                    # Dataset created locally on unsynced store do not have
                    # this attribute
                    last_local = local_record["user_defined"].get(
                        "last_update_from_db", -1)
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
                                # deleted locally
                                if uri not in local_record["files"]:
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
                                # deleted locally
                                if name not in local_record["data"]:
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
                    last_local = local_record["user_defined"].get(
                        "last_update_from_db", -1)
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
                    # Dataset created locally on unsynced store do not have
                    # this attribute
                    last_local = local_record["user_defined"].get(
                        "last_update_from_db", -1)
                    if last_local != -1:  # yep we read it from store
                        conf = {
                            local_record["data"]["name"]["value"]: (
                                "deleted in store", "", "")}
                        conf["last_check_from_db"] = last_local
                        conf["type"] = "delete"
                        if key not in conflicts:
                            conflicts[key] = conf
                        else:
                            conflicts[key].update(conf)
                except Exception:  # deleted too so no issue
                    pass
        return conflicts

    def is_synchronous(self):
        """is_synchronous is store is synchronous mode

        :return: synchronous or not
        :rtype: bool
        """
        return self.__sync__

    def synchronous(self, mode=None):
        """Change sync mode for the store

        :param mode: The mode to True means synchronous mode, False means asynchronous, None  means switch
                     anything else is ignored and it simply returns the mode
        :type mode: bool
        :return: current synchronization mode
        :rtype: bool
        """

        if mode is None:
            self.__sync__ = not self.__sync__
        elif mode in [True, False]:
            if mode and not self.__sync__:  # Going to go to always sync on need to sync first
                self.sync()
            self.__sync__ = mode
        return self.__sync__

    def sync(self, keys=None):
        """Sync with db
        :param keys: keys of objects to sync (id/type)
        :type keys: list
        :return: None
        :rtype: None
        """
        if self.__sync__:
            return

        if not hasattr(self.__record_handler__, "insert"):
            raise RuntimeError("Kosh store is read_only, cannot sync with it")

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
                msg += "\n\tLast read from db: {}".format(
                    conflicts[key]["last_check_from_db"])
                for k in conflicts[key]:
                    if k in ["last_check_from_db", "type"]:
                        continue
                    if conflicts[key]["type"] == "attribute":
                        st = "\n\t" + k + " modified to value '{}' at {} in db, modified locally to '{}' at {}"
                    elif conflicts[key]["type"] == "delete":
                        st = "\n\t" + k + "{} {} {}"
                    else:
                        st = "\n\tfile '" + k + \
                            "' mimetype modified to'{}' at {} in db, modified locally to '{}' at {}"
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

        rels = []
        relationships = self.get_sina_store().relationships
        for id_ in update_records:
            rels += relationships.find(id_.id, None, None)
            rels += relationships.find(None, None, id_.id)
        self.__record_handler__.delete(del_keys)
        self.__record_handler__.insert(update_records)
        relationships.insert(rels)
        for key in list(keys):
            try:
                self._added_unsync_mem_store.records.delete(key)
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

        existing_users = self.__record_handler__.find_with_type(
            self._users_type)
        users = [rec["data"]["username"]["value"] for rec in existing_users]
        if username not in users:
            # Create user
            uid = hashlib.md5(username.encode()).hexdigest()
            user = Record(id=uid, type=self._users_type)
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

        existing_groups = self.__record_handler__.find_with_type(
            self._groups_type)
        groups_names = [rec["data"]["name"]["value"]
                        for rec in existing_groups]
        if group in groups_names:
            raise ValueError("group {} already exist".format(group))

        # now get unix groups
        unix_groups = [g[0] for g in grp.getgrall()]
        if group in unix_groups:
            raise ValueError("{} is a unix group on this system.format(group)")

        # Create group
        uid = uuid.uuid4().hex
        group_rec = Record(id=uid, type=self._groups_type)
        group_rec.add_data("name", group)
        self.__record_handler__.insert(group_rec)

    def add_user_to_group(self, username, groups):
        """Add a user to some group(s)

        :param username: username to add
        :type username: str
        :param groups: kosh specific groups to add to this user
        :type groups: list
        """

        users_filter = self.__record_handler__.find_with_type(
            self._users_type, ids_only=True)
        names_filter = list(
            self.__record_handler__.find_with_data(
                username=username))
        inter_recs = set(users_filter).intersection(set(names_filter))
        if len(inter_recs) == 0:
            raise ValueError("User {} does not exists".format(username))
        user = self.get_record(names_filter[0])
        user_groups = user["data"].get(
            self._groups_type, {
                "value": []})["value"]

        existing_groups = self.__record_handler__.find_with_type(
            self._groups_type)
        groups_names = [rec["data"]["name"]["value"]
                        for rec in existing_groups]
        for group in groups:
            if group not in groups_names:
                warnings.warn(
                    "Group {} is not a Kosh group, skipping".format(group))
                continue
            user_groups.append(group)
        if len(user_groups) == 0:
            user.add_data("groups", None)
        else:
            user.add_data("groups", list(set(user_groups)))
        self.__record_handler__.delete(names_filter[0])
        self.__record_handler__.insert(user)

    def export_dataset(self, datasets, file=None):
        """exports a dataset

        :param datasets: dataset (or their ids) to export
        :type datasets: list or str
        :param file: optional file to dump dataset to
        :type file: None or str
        """
        if not isinstance(datasets, (list, tuple, types.GeneratorType)):
            datasets = [datasets, ]
        for dataset in datasets:
            if isinstance(dataset, basestring):
                return self.open(dataset).export(file)
            else:
                return dataset.export(file)

    def import_dataset(self, datasets, match_attributes=[
                       "name", ], merge_handler=None, merge_handler_kargs={}):
        """import datasets and ensembles that were exported from another store, or load them from a json file
        :param datasets: Dataset/Ensemble object exported by another store, a dataset/ensemble
                         or a json file containing these.
        :type datasets: json file, json loaded object, KoshDataset or KoshEnsemble
        :param match_attributes: parameters on a dataset to use if this it is already in the store
                                 in general we can't use 'id' since it is randomly generated at creation
                                 If the "same" dataset was created in two different stores
                                 (e.g running the same code twice but with different Kosh store)
                                 the dataset would be identical in both store but with different ids.
                                 This helps you make sure you do not end up with duplicate entries.
                                 Warning, if this parameter is too lose too many datasets will match
                                 and the import will abort, if it's too tight duplicates will not be identified.
        :type match_attributes: list of str
        :param merge_handler: If found dataset has attributes with different values from imported dataset
                                 how do we handle this? Accept values are: None, "conservative", "overwrite",
                                 "preserve", or a function.
                              The function decalartion should be:
                                        foo(store_dataset,
                                            imported_dataset_attributes_dict,
                                            section,
                                            **merge_handler_kargs)
                              Where `store_dataset` is the destination kosh dataset or its non-data dictionary section
                                    `imported_dataset_attributes_dict` is a dictionary of attributes/values
                                                                       of the dataset being imported
                                    `section` is the section of the record being updated
                                    `merge_handler_kargs` is a dict of passed for this function
                              And return a dictionary of attributes/values the target_dataset should have.
        :type merge_handler: None, str, func
        :param merge_handler_kargs: If a function is passed to merge_handler these keywords arguments
                                    will be passed in addition to this store dataset and the imported dataset.
        :type merge_handler_kargs: dict
        :return: list of datasets
        :rtype: list of KoshSinaDataset
        """
        out = []
        if not isinstance(datasets, (list, tuple, types.GeneratorType)):
            return self._import_dataset(datasets, match_attributes=match_attributes,
                                        merge_handler=merge_handler,
                                        merge_handler_kargs=merge_handler_kargs)
        else:
            for dataset in datasets:
                out.append(self._import_dataset(dataset, match_attributes=match_attributes,
                                                merge_handler=merge_handler,
                                                merge_handler_kargs=merge_handler_kargs))
        return out

    def _import_dataset(self, datasets, match_attributes=[
            "name", ], merge_handler=None, merge_handler_kargs={}):
        """import dataset that was exported from another store, or load them from a json file
        :param datasets: Dataset object exported by another store, a dataset or a json file containing the dataset
        :type datasets: json file, json loaded object or kosh.KoshDataset
        :param match_attributes: parameters on a dataset to use if this it is already in the store
                                 in general we can't use 'id' since it is randomly generated at creation
                                 If the "same" dataset was created in two different stores
                                 (e.g running the same code twice but with different Kosh store)
                                 the dataset would be identical in both store but with different ids.
                                 This helps you make sure you do not end up with duplicate entries.
                                 Warning, if this parameter is too lose too many datasets will match
                                 and the import will abort, if it's too tight duplicates will not be identified.
        :type match_attributes: list of str
        :param merge_handler: If found dataset has attributes with different values from imported dataset
                                 how do we handle this? Accept values are: None, "conservative", "overwrite",
                                 "preserve", or a function.
                                 A function should take in foo(store_dataset, imported_dataset, **merge_handler_kargs)
        :type merge_handler: None, str, func
        :param merge_handler_kargs: If a function is passed to merge_handler these keywords arguments
                                    will be passed in addition to this store dataset and the imported dataset.
        :type merge_handler_kargs: dict
        :return: list of datasets
        :rtype: list of KoshSinaDataset
        """
        if isinstance(datasets, str):
            with open(datasets) as f:
                from_file = orjson.loads(f.read())
                records_in = from_file.get("records", [])
                relationships_in = from_file.get("relationships", [])
        elif isinstance(datasets, dict):
            from_file = datasets
            records_in = from_file["records"]
            relationships_in = from_file.get("relationships", [])
        elif isinstance(datasets, (KoshDataset, KoshEnsemble)):
            from_file = datasets.export()
            records_in = from_file["records"]
            relationships_in = []
        else:
            raise ValueError(
                "`datasets` must be a Kosh importable object or a file or dict containing json-ized datasets")

        # setup merge handler
        ok_merge_handler_values = [
            None, "conservative", "preserve", "overwrite"]
        if merge_handler in ok_merge_handler_values:
            merge_handler_kargs = {"handling_method": merge_handler}
            merge_handler = merge_datasets_handler
        elif not (isfunction(merge_handler) or ismethod(merge_handler)):
            raise ValueError(
                "'merge_handler' must be one {} or a function/method".format(ok_merge_handler_values))

        matches = []
        remapped = {}
        for record in records_in:
            data = record["data"]
            if record["type"] == from_file.get("sources_type", "file"):
                is_source = True
            else:
                is_source = False
            # Not 100% data.keys is guaranteed to come back the same twice in a
            # row
            keys = sorted(data.keys())
            atts = dict(zip(keys, [data[x]["value"] for x in keys]))
            min_ver = from_file.get("minimum_kosh_version", (0, 0, 0))
            if min_ver is not None and kosh.version(comparable=True) < min_ver:
                raise ValueError("Cannot import dataset it requires min kosh version of {}, we are at: {}".format(
                    min_ver, kosh.version(comparable=True)))

            if not is_source:
                # Ok now we need to see if dataset already exist?
                match_dict = {}
                for attribute in match_attributes:
                    if attribute in atts:
                        match_dict[attribute] = atts[attribute]
                    elif attribute == "id":
                        match_dict["id"] = record["id"]

                matching = list(self.find(**match_dict))
                if len(matching) > 1:
                    raise ValueError("dataset criteria: {} matches multiple ({}) "
                                     "datasets in store {}, try changing 'match_attributes' when calling"
                                     " this function".format(
                                         match_dict, len(matching), self.db_uri))
                elif len(matching) == 1:
                    # All right we do have a possible conflict here
                    match = matching[0]
                    merged_attributes = merge_handler(
                        match, atts, "data", **merge_handler_kargs)
                    # Ok at this point no conflict!
                    match.update(merged_attributes)
                    match_rec = match.get_record()
                    remapped[record["id"]] = match_rec.id
                else:  # Non existent dataset
                    cont = True
                    while cont:
                        try:
                            self.__record_handler__.get(record["id"])
                            # Ok this record already exists
                            # and we need a new unique one
                            record["id"] = uuid.uuid4().hex
                        except ValueError:
                            # Does not exists, let's keep the id
                            cont = False
                    match_rec = record
            else:  # ok it is a source
                # Let's find the source rec that match this uri
                current_sources = list(
                    self.find(
                        types=[self._sources_type, ],
                        uri=data["uri"]["value"],
                        ids_only=True))
                if len(current_sources) > 0:
                    match = self._load(current_sources[0])
                    if match.id != record["id"]:
                        # Darn this store already associated this uri
                        if match.mime_type != data["mime_type"]["value"]:
                            raise ValueError("trying to import an associated source {} with mime_type {} but "  # noqa
                                             "this store already associated"  # noqa
                                             " this source with mime_type {}".format(data["uri"]["value"],
                                                                                     data["mime_type"["value"],
                                                                                     match.mime_type]))
                    match_rec = match.get_record()
                else:
                    match_rec = record
            # update the record
            # But first make sure it is a record :)
            if isinstance(match_rec, dict):
                match_rec = sina.model.generate_record_from_json(match_rec)
            # User defined and files are preserved?
            for section in ["user_defined", "files", "library_data"]:
                if section in record:
                    if match_rec.raw[section] != record[section] and record[section] != {
                    }:
                        if merge_handler != merge_datasets_handler:
                            match_rec.raw[section].update(
                                merge_handler(match_rec, record[section],
                                              section, **merge_handler_kargs))
                        elif merge_handler_kargs["handling_method"] == "conservative":
                            raise RuntimeError(
                                "{} section do not match aborting under conservative merge option"
                                "\nImport: {}\nInstore: {}".format(section, record[section], match_rec.raw[section]))
                        elif merge_handler_kargs["handling_method"] == "overwrite":
                            match_rec.raw[section].update(record[section])
                        else:  # preserve
                            pass
            # Curves are preserved
            if "curve_sets" in record:
                for curve_set in record["curve_sets"]:
                    if curve_set not in match_rec.raw["curve_sets"]:
                        match_rec.raw["curve_sets"][curve_set] = record["curve_sets"][curve_set]
                    else:
                        if merge_handler != merge_datasets_handler:
                            raise RuntimeError(
                                "We do not know how to merge curves with custom merge handler")
                        if merge_handler_kargs["handling_method"] == "conservative":
                            if match_rec.raw["curve_sets"][curve_set] != record["curve_sets"][curve_set]:

                                raise RuntimeError(
                                    "curveset {} do not match, `conservative` method used, aborting".format(curve_set))
                        elif merge_handler_kargs["handling_method"] == "overwrite":
                            match_rec.raw["curve_sets"][curve_set]["independent"].update(
                                record["curve_sets"][curve_set]["independent"])
                            match_rec.raw["curve_sets"][curve_set]["dependent"].update(
                                record["curve_sets"][curve_set]["dependent"])
                        else:  # preserve
                            pass
            try:
                self.__record_handler__.update(match_rec)
            except ValueError:
                self.__record_handler__.insert(match_rec)
            matches.append(match_rec["id"])

        for relationship in relationships_in:
            try:
                rel = Relationship(subject_id=relationship.subject,
                                   predicate=relationship.predicate,
                                   object_id=relationship.object)
                self.get_sina_store().relationships.insert(rel)
            except Exception:  # sqlalchemy.exc.IntegrityError
                pass
        # We need to make sure any merged (remapped) dataset is still properly
        # associated
        for id_ in matches:
            rec = self.get_record(id_)
            if rec["type"] == self._sources_type == from_file.get(
                    "sources_type", "file"):
                associated = rec["data"]["associated"]["value"]
                altered = False
                for rem in remapped:
                    if rem in associated:
                        index = associated.index(rem)
                        associated[index] = remapped[rem]
                        d = self.open(remapped[rem])
                        d.associate(
                            rec["data"]["uri"]["value"],
                            rec["data"]["mime_type"]["value"])
                        altered = True
                if altered:
                    try:
                        self.__record_handler__.delete(rec["id"])
                    except ValueError:
                        pass
                    self.__record_handler__.insert(rec)

        return [self._load(x) for x in matches]

    def reassociate(self, target, source=None, absolute_path=True):
        """This function allows to re-associate data whose uri might have changed

        The source can be the original uri or sha and target is the new uri to use.
        :param target: New uri
        :type target: str
        :param source: uri or sha (long or short of reassociate) to reassociate
                       with target, if None then the short uri from target will be used
        :type source: str or None
        :param absolute_path: if file exists should we store its absolute_path
        :type absolute_path: bool
        :return: None
        :rtype: None
        """

        # First let's convert to abs path if necessary
        if absolute_path:
            if os.path.exists(target):
                target = os.path.abspath(target)
            if source is not None and os.path.exists(source):
                source = os.path.abspath(source)

        # Now, did we pass a source for uri to replace?
        if source is None:
            source = compute_fast_sha(target)

        # Ok now let's get all associated uri that match
        # Fist assuming it's a fast_sha search all "kosh files" that match this
        matches = list(
            self.find(
                types=[
                    self._sources_type,
                ],
                fast_sha=source,
                ids_only=True))
        # Now it could be simply a uri
        matches += list(
            self.find(
                types=[
                    self._sources_type,
                ],
                uri=source,
                ids_only=True))
        # And it's quite possible it's a long_sha too
        matches += list(self.find(types=[self._sources_type, ],
                                  long_sha=source, ids_only=True))

        # And now let's do the work
        for match_id in matches:
            try:
                match = self._load(match_id)
                match.uri = target
            except Exception:
                pass

    def cleanup_files(self, dry_run=False, interactive=False, clean_fastsha=False,
                      **dataset_search_keys):
        """Cleanup the store from references to dead files
        Also updates the fast_shas if necessary
        You can filter associated objects for each dataset by passing key=values
        e.g mime_type=hdf5 will only dissociate non-existing files associated with mime_type hdf5
        some_att=some_val will only dissociate non-existing files associated and having the attribute
        'some_att' with value of 'some_val'
        returns list of uris to be removed.
        :param dry_run: Only does a dry_run
        :type dry_run: bool
        :param interactive: interactive mode, ask before dissociating
        :type interactive: bool
        :param clean_fastsha: Do we want to update fast_sha if it changed?
        :type clean_fastsha: bool
        :returns: list of uris (to be) removed.
        :rtype: list
        """
        missings = []
        datasets = self.find()
        for dataset in datasets:
            missings += dataset.cleanup_files(dry_run=dry_run,
                                              interactive=interactive,
                                              clean_fastsha=clean_fastsha,
                                              **dataset_search_keys)
        return missings

    def check_integrity(self):
        """Runs a sanity check on the store:
        1- Are associated files reachable?
        2- Did fast_shas change since file was associated
        """
        return self.cleanup_files(dry_run=True, clean_fastsha=True)

    def associate(self, store, reciprocal=False):
        """Associate another store

        All associated stores will be used for queries purposes.

        WARNING: While associating stores will make them look like one big store,
                 ensembles' members MUST belong to the same store as the ensemble.

        :param store: The store to associate
        :type store: KoshStore

        :param reciprocal: By default, this is a one way relationship.
                           The associated store will NOT be aware of
                           this association, turning this on create
                           the association in both stores.
        :type reciprocal: bool
        """
        if not isinstance(store, KoshStore):
            raise TypeError("store must be a KoshStore or path to one")

        sina_recs = self.get_sina_records()
        store_info = list(sina_recs.find_with_type("__kosh_storeinfo__"))[0]
        if "associated_stores" not in store_info["data"]:
            store_info.add_data("associated_stores", [])
        stores = store_info["data"]["associated_stores"]["value"]
        if store.db_uri not in stores:
            stores.append(store.db_uri)
            store_info["data"]["associated_stores"]["value"] = stores
            sina_recs.delete(store_info["id"])
            sina_recs.insert(store_info)
            self._associated_stores_.append(store)
        if reciprocal:
            store.associate(self)

    def dissociate(self, store, reciprocal=False):
        """Dissociate another store

        :param store: The store to associate
        :type store: KoshStore or basestring

        :param reciprocal: By default, this is a one way relationship.
                           The disssociated store will NOT be aware of
                           this action, turning this on create
                           the dissociation in both stores.
        :type reciprocal: bool
        """
        if not isinstance(store, (basestring, KoshStore)):
            raise TypeError("store must be a KoshStore or path to one")

        sina_recs = self.get_sina_records()
        store_info = list(sina_recs.find_with_type("__kosh_storeinfo__"))[0]
        if "associated_stores" not in store_info["data"]:
            warnings.warn("No store is associated with this store: {}".format(self.db_uri))
            return
        # refresh value
        stores = store_info["data"]["associated_stores"]["value"]

        if isinstance(store, basestring):
            try:
                store_path = store
                store = self.get_associated_store(store_path)
            except Exception:
                raise ValueError("Could not open store at: {}".format(store_path))

        if store.db_uri in stores:
            stores.remove(store.db_uri)
            store_info["data"]["associated_stores"]["value"] = stores
            sina_recs.delete(store_info["id"])
            sina_recs.insert(store_info)
            self._associated_stores_.remove(store)
        else:
            warnings.warn("store {} does not seem to be associated with this store ({})".format(
                store.db_uri, self.db_uri))

        if reciprocal:
            store.dissociate(self)

    def get_associated_store(self, uri):
        """Returns the associated store based on its uri.

        :param uri: uri to the desired store
        :type uri: basestring
        :returns: Associated kosh store
        :rtype: KoshStore
        """

        if not isinstance(uri, basestring):
            raise TypeError("uri must be string")

        for store in self._associated_stores_:
            if store.db_uri == uri:
                return store
        raise ValueError(
            "{} store does not seem to be associated with this store: {}".format(uri, store.db_uri))

    def get_associated_stores(self, uris=True):
        """Return the list of associated stores
        :param uris: Return the list of uri pointing to the store if True,
                     or the actual stores otherwise.
        :type uris: bool
        :returns: generator to stores
        :rtype: generator
        """
        for store in self._associated_stores_:
            if uris:
                yield store.db_uri
            else:
                yield store

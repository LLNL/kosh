import uuid
from .loaders import KoshLoader, KoshFileLoader, PGMLoader, get_graph
from .utils import compute_fast_sha, merge_datasets_handler
import warnings
import os
import kosh
import time
import fcntl
import collections
from inspect import isfunction, ismethod
try:
    basestring
except NameError:
    basestring = str
try:
    import orjson
except ImportError:
    import json as orjson  # noqa
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
from .loaders import JSONLoader
from .loaders import NpyLoader

from .schema import KoshSchema
from .utils import compute_long_sha
from .utils import update_store_and_get_info_record
from .utils import create_kosh_users
from sina.datastore import connect as sina_connect
from sina.model import Record
from sina import get_version
import sina.utils
import pickle
import grp
import numpy
import hashlib


sina_version = float(".".join(get_version().split(".")[:2]))


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
                "You cannot specifiy `db` and `database_type` with different values")
        database_type = db
    sina_store = sina_connect(database=database,
                              keyspace=keyspace,
                              database_type=database_type,
                              allow_connection_pooling=allow_connection_pooling,
                              read_only=read_only)
    if not read_only:
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
    if delete_all_contents:
        store.delete_all_contents(force="SKIP PROMPT")
    return store


def cleanup_sina_record_from_kosh_sync(record):
    """Kosh adds data in the 'user_defined' section of records to keep track of syncing
    This removes these attributes
    :param record: The Sina record to cleanup
    :type record: sina.model.Record
    :return: json loaded representation of the record
    :rtype: dict"""
    # cleanup the record
    record["user_defined"].pop("last_update_from_db", None)
    for key in list(record["user_defined"].keys()):
        if key.endswith("last_modified"):
            record["user_defined"].pop(key)
    return orjson.loads(record.to_json())


class KoshSinaObject(object):
    """KoshSinaObject Base class for sina objects
    """

    def get_record(self):
        return self.__store__.get_record(self.id)

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
            "id", "__type__", "__protected__",
            "__record_handler__", "__store__", "id", "__schema__"] + protected
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
            self.__dict__["id"] = Id
        else:
            self.__dict__["id"] = Id
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
        if name == "__id__":
            warnings.warn(
                "the attribute '__id__' has been deprecated in favor of 'id'",
                DeprecationWarning)
            name = "id"
        if name in self.__dict__["__protected__"]:
            if name == "_associated_data_":
                record = self.get_record()
                # Any curve sets?
                if len(record["curve_sets"]) != 0:
                    out = [self.id, ]
                else:
                    out = []
                # we cannot use list comprehension
                # some pure sina rec have file but no kosh_id
                for file_rec in record["files"]:
                    file_entry = record["files"][file_rec]
                    if "kosh_id" in file_entry:
                        out.append(record["files"][file_rec]["kosh_id"])
                    else:
                        # Not an entry made by Kosh
                        # But maybe we can salvage this!
                        # did  the user added a mime_type?
                        if "mimetype" in file_entry:
                            out.append("{}__uri__{}".format(self.id, file_rec))
                return out
            else:
                return self.__dict__[name]
        record = self.get_record()
        if name == "__attributes__":
            return self.__getattributes__()
        elif name == "schema":
            if self.__dict__[
                    "__schema__"] is None and "schema" in record["data"]:
                schema = pickle.loads(
                    record["data"]["schema"]["value"].encode("latin1"))
                self.__dict__["__schema__"] = schema
            return self.__dict__["__schema__"]
        if name not in record["data"]:
            if name == "mime_type":
                return record["type"]
            else:
                raise AttributeError(
                    "Object {} does not have {} attribute".format(self.id,
                                                                  name))
        value = record["data"][name]["value"]
        if name == "creator":
            # old records have user id let's fix this
            if value in self.__store__.__record_handler__.find_with_type(
                    self.__store__._users_type, ids_only=True):
                value = self.__store__.get_record(
                    value)["data"]["username"]["value"]
        return value

    def get_sina_store(self):
        """Returns the sina store object"""
        return self.__store__.get_sina_store()

    def get_sina_records(self):
        """Returns sina store's records"""
        return self.__record_handler__

    def update(self, attributes):
        """update many attributes at once to limit db writes
        :param: attributes: dictionary with attributes to update
        :type attributes: dict
        """
        if 'id' in attributes:
            del(attributes['id'])
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
            return record
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
        if last_db > last and getattr(
                self, name) != record["data"][name]["value"]:  # Ooopsie someone touched it!
            raise AttributeError("Attribute {} of object id {} was modified since last sync\n"
                                 "Last modified in db at: {}, value: {}\n"
                                 "You last read it at: {}, with value: {}".format(
                                     name, self.id,
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
            self.__record_handler__.delete(self.id)
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
            self.__record_handler__.delete(self.id)
            self.__record_handler__.insert(record)
            self.__store__.unlock()

    def sync(self):
        """sync this object with database"""
        self.__store__.sync([self.id, ])

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
        attributes = list(record["data"].keys()) + ['id', ]
        for att in self.__protected__:
            if att in attributes and att != "id":
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
                if attributes[a] in self.__store__.__record_handler__.find_with_type(
                        self.__store__._users_type, ids_only=True):
                    attributes[a] = self.__store__.get_record(
                        attributes[a])["data"]["username"]["value"]
        return attributes

    def __str__(self):
        """String for printing"""
        st = "Id: {}".format(self.id)
        for att in sorted(self.listattributes()):
            if att != 'id':
                st += "\n\t{}: {}".format(att, getattr(self, att))
        return st


class KoshSinaFile(KoshSinaObject):
    """KoshSinaFile file representation in Kosh via Sina"""

    def open(self, *args, **kargs):
        """open opens the file
        :return: handle to file in open mode
        """
        return self.__store__.open(self.id, *args, **kargs)


class KoshDataset(KoshSinaObject):
    def __init__(self, id, store, schema=None, record=None):
        """KoshSinaDataset Sina representation of Kosh Dataset

        :param id: dataset's unique Id
        :type id: str
        :param store: store containing the dataset
        :type store: KoshSinaStore
        :param schema: Kosh schema validator
        :type schema: KoshSchema
        :param record: to avoid looking up in sina pass sina record
        :type record: Record
        """
        super(KoshDataset, self).__init__(id, koshType=store._dataset_record_type,
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

    def __str__(self):
        """string representation"""
        st = ""
        st += "KOSH DATASET\n"
        st += "\tid: {}\n".format(self.id)
        try:
            st += "\tname:{}\n".format(self.__name__)
        except Exception:
            st += "\tname:???\n"
        try:
            st += "\tcreator: {}\n".format(self.creator)
        except Exception:
            st += "\tcreator: ???\n"
        atts = self.__attributes__
        if len(atts) > 0:
            st += "\n--- Attributes ---\n"
            for a in sorted(atts):
                if a == "_associated_data_":
                    continue
                st += "\t{}: {}\n".format(a, atts[a])
        if self._associated_data_ is not None:
            st += "--- Associated Data ({})---\n".format(
                len(self._associated_data_))
            # Let's organize per mime_type
            associated = {}
            for a in self._associated_data_:
                if a == self.id:
                    st2 = "internal ( {} )".format(
                        ", ".join(self.get_record()["curve_sets"].keys()))
                    if "sina/curve" not in associated:
                        associated["sina/curve"] = [st2, ]
                    else:
                        associated["sina/curve"].append(st2)
                else:
                    if "__uri__" in a:
                        a_id, a_uri = a.split("__uri__")
                        a_mime_type = self.get_record(
                        )["files"][a_uri]["mimetype"]
                    else:
                        a_id, a_uri = a, None
                    a_obj = self.__store__._load(a_id)
                    if a_uri is None:
                        a_uri = a_obj.uri
                        a_mime_type = a_obj.mime_type
                    st2 = "{a_uri} ( {a} )".format(a_uri=a_uri, a=a_id)
                    if a_mime_type not in associated:
                        associated[a_mime_type] = [st2, ]
                    else:
                        associated[a_mime_type].append(st2)
            for mime in sorted(associated):
                st += "\tMime_type: {mime}".format(mime=mime)
                for uri in sorted(associated[mime]):
                    st += "\n\t\t{uri}".format(uri=uri)
                st += "\n"
        return st

    def _repr_pretty_(self, p, cycle):
        """Pretty display in Ipython"""
        p.text(self.__str__())

    def cleanup_files(self, dry_run=False, interactive=False, **search_keys):
        """Cleanup the dataset from references to dead files
        You can filter associated objects by passing key=values
        e.g mime_type=hdf5 will only dissociate non-existing files associated with mime_type hdf5
        some_att=some_val will only dissociate non-exisiting files associated and having the attribute
        'some_att' with value of 'some_val'
        returns list of uris to be removed.
        :param dry_run: Only does a dry_run
        :type dry_run: bool
        :param interactive: interactive mode, ask before dissociating
        :type interactive: bool
        :returns: list of uris (to be) removed.
        :rtype: list
        """
        print_some = False
        missings = []
        for associated in self.find(**search_keys):
            clean = 'n'
            if not os.path.exists(associated.uri):  # Ok this is gone
                missings.append(associated.uri)
                if not print_some and (interactive or dry_run):
                    print_some = True
                if dry_run:  # Dry run
                    clean = 'n'
                elif interactive:
                    clean = input("\tDo you want to dissociate {} (mime_type: {})? [Y/n]".format(
                        associated.uri, associated.mime_type)).strip()
                    if len(clean) > 0:
                        clean = clean[0]
                        clean = clean.lower()
                    else:
                        clean = 'y'
                else:
                    clean = 'y'
                if clean == 'y':
                    self.dissociate(associated.uri)
        return missings

    def open(self, Id=None, loader=None, *args, **kargs):
        """open an object associated with a dataset

        :param Id: id of object to open, defaults to None which means first one.
        :type Id: str, optional
        :param loader: loader to use for this object, defaults to None
        :type loader: KoshLoader, optional
        :raises RuntimeError: object id not associated with dataset
        :return: object ready to be used
        """
        if Id is None:
            if len(self._associated_data_) > 0:
                Id = self._associated_data_[0]
            else:
                for Id in self._associated_data_:
                    return self.__store__.open(Id, loader)
        elif Id not in self._associated_data_:
            raise RuntimeError(
                "object {Id} is not associated with this dataset".format(
                    Id=Id))
        return self.__store__.open(Id, loader, *args, **kargs)

    def list_features(self, Id=None, loader=None,
                      use_cache=True, *args, **kargs):
        """list_features list features available if multiple associated data lead to duplicate feature name
        then the associated_data uri gets appended to feature name

        :param Id: id of associated object to get list of features from, defaults to None which means all
        :type Id: str, optional
        :param loader: loader to use to search for feature, will return ONLY features that the loader knows about
        :type loader: kosh.loaders.KoshLoader
        :param use_cache: If features is found on cache use it (default: True)
        :type use_cache: bool
        :raises RuntimeError: object id not associated with dataset
        :return: list of features available
        :rtype: list
        """
        if use_cache and self.__dict__["__features__"].get(
                Id, {}).get(loader, None) is not None:
            return self.__dict__["__features__"][Id][loader]
        # Ok no need to sync any of this we will not touch the code
        saved_sync = self.__store__.is_synchronous()
        if saved_sync:
            # we will not update any rec in here, turnin off sync
            # it makes things much d=faster
            backup = self.__store__.__sync__dict__
            self.__store__.__sync__dict__ = {}
            self.__store__.synchronous()
        features = []
        loaders = []
        associated_data = self._associated_data_
        if Id is None:
            for associated in associated_data:
                if loader is None:
                    ld, _ = self.__store__._find_loader(associated)
                else:
                    if associated not in self.__store__._cached_loaders:
                        self.__store__._cached_loaders[associated] = loader(
                            self.__store__._load(associated))
                    ld, _ = self.__store__._cached_loaders[associated]
                loaders.append(ld)
                try:
                    features += ld._list_features(*
                                                  args, use_cache=use_cache, **kargs)
                except Exception:  # Ok the loader couldn't get the feature list
                    pass
            if len(features) != len(set(features)):
                # duplicate features we need to redo
                # Adding uri to feature name
                ided_features = []
                for index, associated in enumerate(associated_data):
                    obj = self.__store__._load(associated)
                    ld = loaders[index]
                    these_features = ld._list_features(
                        *args, use_cache=use_cache, **kargs)
                    for feature in these_features:
                        if features.count(feature) > 1:  # duplicate
                            ided_features.append(
                                "{feature}_@_{obj.uri}".format(feature=feature, obj=obj))
                        else:  # not duplicate name
                            ided_features.append(feature)
                features = ided_features
        elif Id not in self._associated_data_:
            raise RuntimeError(
                "object {Id} is not associated with this dataset".format(
                    Id=Id))
        else:
            ld, _ = self.__store__._find_loader(Id)
            features = ld._list_features(*args, use_cache=use_cache, **kargs)
        features_id = self.__dict__["__features__"].get(Id, {})
        features_id[loader] = features
        self.__dict__["__features__"][Id] = features_id
        if saved_sync:
            # we need to restore sync mode
            self.__store__.__sync__dict__ = backup
            self.__store__.synchronous()
        return features

    def get_execution_graph(self, feature=None, Id=None,
                            loader=None, transformers=[], *args, **kargs):
        """get data for a specific feature
        :param feature: feature (variable) to read, defaults to None
        :type feature: str, optional if loader does not require this
        :param Id: object to read in, defaults to None
        :type Id: str, optional
        :param loader: loader to use to get data,
                       defaults to None means pick for me
        :type loader: kosh.loaders.KoshLoader
        :param transformers: A list of transformers to use after the data is loaded
        :type transformers: kosh.transformer.KoshTranformer
        :returns: [description]
        :rtype: [type]
        """
        if feature is None:
            out = []
            for feat in self.list_features():
                out.append(self.get_execution_graph(Id=None,
                                                    feature=feat,
                                                    format=format,
                                                    loader=loader,
                                                    transformers=transformers,
                                                    *args, **kargs))
            return out
        # Need to make sure transformers are a list
        if not isinstance(transformers, (list, tuple)):
            transformers = [transformers, ]
        # we need to figure which associated data has the feature
        if not isinstance(feature, list):
            features = [feature, ]
        else:
            features = feature
        possibles = {}
        inter = None
        union = set()
        for feature_ in features:
            possible_ids = []
            if Id is None:
                for a in self._associated_data_:
                    a_original = a
                    if "__uri__" in a:
                        # Ok this is a pure sina file with mime_type
                        a, _ = a.split("__uri__")
                    a_obj = self.__store__._load(a)
                    if loader is None:
                        ld, _ = self.__store__._find_loader(a_original)
                        if ld is None:  # unknown mimetype probably
                            continue
                    else:
                        if a_obj.mime_type in loader.types:
                            ld = loader(a_obj)
                        else:
                            continue
                    # Dataset with urve have themsleves as uri
                    obj_uri = getattr(ld.obj, "uri", "self")
                    if ("_@_" not in feature_ and feature_ in ld._list_features()) or\
                            feature_ is None or\
                            (feature_[:-len(obj_uri) - 3] in ld._list_features() and
                             feature_[-len(obj_uri):] == obj_uri):
                        possible_ids.append(a_original)
                if possible_ids == []:  # All failed but could be something about the feature
                    raise ValueError(
                        "Cannot find feature {} in dataset".format(feature_))
            elif Id == self.id:
                # Ok asking for data not associated externally
                # Likely curve
                ld, _ = self.__store__._find_loader(Id)
                if feature_ in ld._list_features():
                    possible_ids = [Id, ]
                else:  # ok not a curve maybe a file?
                    rec = self.get_record()
                    for uri in rec["files"]:
                        if "mimetype" in rec["files"][uri]:
                            full_id = "{}__uri__{}".format(Id, uri)
                            ld, _ = self.__store__._find_loader(full_id)
                            if ld is not None and feature_ in ld.list_features():
                                possible_ids = [full_id, ]
            elif Id not in self._associated_data_:
                raise RuntimeError(
                    "object {Id} is not associated with this dataset".format(
                        Id=Id))
            else:
                possible_ids = [Id, ]
            if inter is None:
                inter = set(possible_ids)
            else:
                inter = inter.intersection(set(possible_ids))
            union = union.union(set(possible_ids))
            possibles[feature_] = possible_ids

        if len(inter) != 0:
            union = inter

        ids = {}
        # Now let's go through each possible uri
        # and group features in thems
        for id_ in union:
            matching_features = []
            for feature_ in features:
                if feature_ in possibles and id_ in possibles[feature_]:
                    matching_features.append(feature_)
                    del(possibles[feature_])
            if len(matching_features) > 0:
                ids[id_] = matching_features

        out = []
        for id_ in ids:
            features = ids[id_]
            for Id in possible_ids:
                tmp = None
                try:
                    if loader is None:
                        ld, _ = self.__store__._find_loader(Id)
                        mime_type = ld._mime_type
                    else:
                        if Id not in self.__store__._cached_loaders:
                            a_obj = self.__store__._load(Id)
                            self.__store__._cached_loaders[Id] = loader(a_obj)
                            mime_type = a_obj.mime_type
                        ld = self.__store__._cached_loaders[Id]
                    # Essentially make a copy
                    # Because we want to attach the feature to it
                    # But lets not lose the cached list_features
                    saved_listed_features = ld.__dict__[
                        "_KoshLoader__listed_features"]
                    ld_uri = getattr(ld, "uri", None)
                    ld = ld.__class__(
                        ld.obj, mime_type=ld._mime_type, uri=ld_uri)
                    ld.__dict__[
                        "_KoshLoader__listed_features"] = saved_listed_features
                    # Ensures there is a possible path to format
                    get_graph(mime_type, ld, transformers)
                    final_features = []
                    obj_uri = getattr(ld.obj, "uri", "")
                    for feature_ in features:
                        if (feature_[:-len(obj_uri) - 3] in ld._list_features()
                                and feature_[-len(obj_uri):] == obj_uri):
                            final_features.append(
                                feature_[:-len(obj_uri) - 3])
                        else:
                            final_features.append(feature_)
                    if len(final_features) == 1:
                        final_features = final_features[0]
                    tmp = ld.get_execution_graph(final_features,
                                                 transformers=transformers
                                                 )
                    ld.feature = final_features
                    ExecGraph = kosh.exec_graphs.KoshExecutionGraph(tmp)
                except Exception:
                    import traceback
                    traceback.print_exc()
                    ExecGraph = kosh.exec_graphs.KoshExecutionGraph(tmp)
                out.append(ExecGraph)

        if len(out) == 1:
            return out[0]
        else:
            return out

    def get(self, feature=None, format=None, Id=None, loader=None,
            group=False, transformers=[], *args, **kargs):
        """get data for a specific feature
        :param feature: feature (variable) to read, defaults to None
        :type feature: str, optional if loader does not require this
        :param format: desired format after extraction
        :type format: str
        :param Id: object to read in, defaults to None
        :type Id: str, optional
        :param loader: loader to use to get data,
                       defaults to None means pick for me
        :type loader: kosh.loaders.KoshLoader
        :param group: group multiple features in one get call, assumes loader can handle this
        :type group: bool
        :param transformers: A list of transformers to use after the data is loaded
        :type transformers: kosh.transformer.KoshTranformer
        :raises RuntimeException: could not get feature
        :raises RuntimeError: object id not associated with dataset
        :returns: [description]
        :rtype: [type]
        """
        G = self.get_execution_graph(
            feature=feature,
            Id=Id,
            loader=loader,
            transformers=transformers,
            *args,
            **kargs)
        if isinstance(G, list):
            return [g.traverse(format=format, *args, **kargs) for g in G]
        else:
            return G.traverse(format=format, *args, **kargs)

    def __getitem__(self, feature):
        """Shortcut to access a feature or list of
        :param feature: feature(s) to access in dataset
        :type feature: str or list of str
        :returns: (list of) access point to feature requested
        :rtype: (list of) kosh.execution_graph.KoshIoGraph
        """
        return self.get_execution_graph(feature)

    def __dir__(self):
        """__dir__ list functions and attributes associated with dataset
        :return: functions, methods, attribute associated with this dataset
        :rtype: list
        """
        current = set(super(KoshDataset, self).__dir__())
        try:
            atts = set(self.listattributes() + self.__protected__)
        except Exception:
            atts = set()
        return list(current.union(atts))

    def reassociate(self, target, source=None, absolute_path=True):
        """This function allows to re-associate data whose uri might have changed

        The source can be the original uri or sha and target is the new uri to use.
        :param target: New uri
        :type target: str
        :param source: uri or sha (long or short of reassociate)
                       to reassociate with target, if None then the short uri from target will be used
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
        # Fist assuming it's a fast_sha
        matches = list(self.find(fast_sha=source, ids_only=True))
        # Now it could be simply a uri
        matches += list(self.find(uri=source, ids_only=True))
        # And it's quite possible it's a long_sha too
        matches += list(self.find(long_sha=source, ids_only=True))

        # And now let's do the work
        for match_id in matches:
            match = self.__store__._load(match_id)
            match.uri = target

    def validate(self):
        """If dataset has a schema then make sure all attributes pass the schema"""
        if self.schema is not None:
            self.schema.validate(self)

    def searchable_source_attributes(self):
        """Returns all the attributes of associated sources
        :return: List of all attributes you can use to search sources in the dataset
        :rtype: set
        """
        searchable = set()
        for source in self.find():
            searchable = searchable.union(source.listattributes())
        return searchable

    def describe_feature(self, feature, Id=None, **kargs):
        """describe a feature

        :param feature: feature (variable) to read, defaults to None
        :type feature: str, optional if loader does not require this
        :param Id: id of associated object to get list of features from, defaults to None which means all
        :type Id: str, optional
        :param kargs: keywords to pass to list_features (optional)
        :type kargs: keyword=value
        :raises RuntimeError: object id not associated with dataset
        :return: dictionary describing the feature
        :rtype: dict
        """
        loader = None
        if Id is None:
            for a in self._associated_data_:
                ld, _ = self.__store__._find_loader(a)
                if feature in ld._list_features(**kargs) or \
                        (feature[:-len(ld.obj.uri) - 3] in ld._list_features()
                         and feature[-len(ld.obj.uri):] == ld.obj.uri):
                    loader = ld
                    break
        elif Id not in self._associated_data_:
            raise RuntimeError(
                "object {Id} is not associated with this dataset".format(
                    Id=Id))
        else:
            loader, _ = self.__store__._find_loader(Id)
        return loader.describe_feature(feature)

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
        rec["user_defined"]["{uri}___associated_last_modified".format(
            uri=uri)] = now
        if self.__store__.__sync__:
            self.__store__.lock()
            self.__record_handler__.delete(rec.id)
            self.__record_handler__.insert(rec)
            self.__store__.unlock()
        # Get all object that have been associated with this uri
        rec = self.__store__.get_record(kosh_id)
        if (not hasattr(rec, "associated")) or len(
                rec.associated) == 0:  # ok no other object is associated
            self.__store__.delete(kosh_id)
            if kosh_id in self.__store__._cached_loaders:
                del(self.__store__._cached_loaders[kosh_id])

        # Since we changed the associated, we need to cleanup
        # the features cache
        self.__dict__["__features__"][None] = {}
        self.__dict__["__features__"][kosh_id] = {}

    def associate(self, uri, mime_type, metadata={},
                  id_only=True, long_sha=False, absolute_path=True):
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
                rec["user_defined"]["{uri}___associated_last_modified".format(
                    uri=uri)] = now
                # We need to check if the uri was already associated somewhere
                tmp_uris = list(self.__store__.find(
                    types=[self.__store__._sources_type, ], uri=uri, ids_only=True))

                if len(tmp_uris) == 0:
                    Id = uuid.uuid4().hex
                    rec_obj = Record(id=Id, type=self.__store__._sources_type)
                else:
                    rec_obj = self.__store__.get_record(tmp_uris[0])
                    Id = rec_obj.id
                    existing_mime = rec_obj["data"]["mime_type"]["value"]
                    mime_type = mime_types[i]
                    if existing_mime != mime_types[i]:
                        rec["files"][uri]["mime_type"] = existing_mime
                        raise TypeError("source {} is already associated with another dataset with mimetype"
                                        " '{}' you specified mime_type '{}'".format(uri, existing_mime, mime_types[i]))
                rec.add_file(uri, mime_types[i])
                rec["files"][uri]["kosh_id"] = Id
                meta["uri"] = uri
                meta["mime_type"] = mime_types[i]
                meta["associated"] = [self.id, ]
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
            self.__store__.__record_handler__.delete(self.id)
            self.__store__.__record_handler__.insert(rec)
            self.__store__.unlock()
        else:
            self.__store__._added_unsync_handler.delete(self.id)
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
                                       koshType=self.__store__._sources_type,
                                       store=self.__store__,
                                       metadata=metadata,
                                       record_handler=self.__record_handler__)
            kosh_files.append(kosh_file)

        if single_element:
            return kosh_files[0]
        else:
            return kosh_files

    def search(self, *atts, **keys):
        """
        Deprecated use find
        """
        warnings.warn("The 'search' function is deprecated and now called `find`.\n"
                      "Please update your code to use `find` as `search` might disappear in the future",
                      DeprecationWarning)
        return self.find(*atts, **keys)

    def find(self, *atts, **keys):
        """find associated data matching some metadata
        arguments are the metadata name we are looking for e.g
        find("attr1", "attr2")
        you can further restrict by specifying exact value for a metadata
        via key=value
        you can return ids only by using: ids_only=True
        range can be specified via: sina.utils.DataRange(min, max)

        "file_uri" is a special key that will return the kosh object associated
        with this dataset for the given uri.  e.g store.find(file_uri=uri)

        :return: list of matching objects associated with dataset
        :rtype: list
        """

        if self._associated_data_ is None:
            return
        sina_kargs = {}
        ids_only = keys.pop("ids_only", False)
        # We are only interested in ids from Sina
        sina_kargs["ids_only"] = True

        inter_recs = self._associated_data_
        tag = "{}__uri__".format(self.id)
        tag_len = len(tag)
        virtuals = [x[tag_len:] for x in inter_recs if x.startswith(tag)]
        # Bug in sina 1.10.0 forces us to remove the virtual from the pool
        for v_id in virtuals:
            inter_recs.remove("{}{}".format(tag, v_id))
        if "file" in sina_kargs and "file_uri" in sina_kargs:
            raise ValueError(
                "The `file` keyword is being deprecated for `file_uri` you cannot use both")
        if "file" in sina_kargs:
            warnings.warn(
                "The `file` keyword has been renamed `file_uri` and may not be available in future versions",
                DeprecationWarning)
            file_uri = sina_kargs.pop("file")
            sina_kargs["file_uri"] = file_uri

        # The data dict for sina
        keys.pop("id_pool", None)
        sina_kargs["query_order"] = keys.pop(
            "query_order", ("data", "file_uri", "types"))
        sina_data = keys.pop("data", {})
        for att in atts:
            sina_data[att] = sina.utils.exists()
        sina_data.update(keys)
        sina_kargs["data"] = sina_data

        match = set(
            self.__record_handler__.find(
                id_pool=inter_recs,
                **sina_kargs))
        # instantly restrict to associated data
        if not self.__store__.__sync__:
            match_mem = set(self.__store__._added_unsync_handler.find(
                id_pool=inter_recs, **sina_kargs))
            match = match.union(match_mem)
        # ok now we need to search the data on the virtual datasets
        rec = self.get_record()
        for uri in virtuals:
            file_section = rec["files"][uri]
            tags = file_section.get("tags", [])
            # Now let's search the tags....
            match_it = True
            for key in sina_kargs["data"]:
                if key == "mime_type":
                    if file_section.get("mimetype", file_section.get(
                            "mime_type", None)) != sina_kargs["data"][key]:
                        match_it = False
                        break
                elif key not in tags or sina_kargs["data"][key] != sina.utils.exists():
                    match_it = False
                    break
            if match_it:
                # we can't have a set anymore
                match.add(self.id)

        for rec_id in match:
            # We need to cleanup for "virtual association".
            # e.g comes directly from a sina rec with 'file'/'mimetype' in it.
            rec_id = rec_id.split("__uri__")[0]
            yield rec_id if ids_only else self.__store__._load(rec_id)

    def export(self, file=None):
        """Exports this dataset
        :param file: export dataset to a file
        :type file: None or str
        :return: dataset and its associated data
        :rtype: dict"""
        rec = self.get_record()
        # cleanup the record
        rec_json = cleanup_sina_record_from_kosh_sync(rec)
        jsns = [rec_json, ]
        # ok now same for associated data
        for associated_id in self._associated_data_:
            rec = self.__store__._load(associated_id).get_record()
            rec_json = cleanup_sina_record_from_kosh_sync(rec)
            jsns.append(rec_json)

        # returns a dict that should be ingestable by sina
        output_dict = {
            "minimum_kosh_version": None,
            "kosh_version": kosh.version(comparable=True),
            "sources_type": self.__store__._sources_type,
            "records": jsns
        }

        if file is not None:
            if os.path.exists(file):
                with open(file) as f:
                    file_dict = orjson.loads(f.read())
                records_already_in_file = file_dict["records"]
                # You can't "set" records
                # This erased records
                file_dict.update(output_dict)
                # ids that are now in file_dict
                new_dataset_rec_ids = [x["id"] for x in output_dict["records"]]
                # Make sure we put the records that were in the file back in
                for record in records_already_in_file:
                    if record["id"] not in new_dataset_rec_ids:
                        file_dict["records"].append(record)
            else:
                file_dict = output_dict

            with open(file, "w") as f:
                f.write(orjson.dumps(file_dict).decode())
        return output_dict


class KoshSinaLoader(KoshLoader):
    """Sina base class for loaders"""
    types = {"dataset": ["numpy", ]}

    def __init__(self, obj, **kargs):
        """KoshSinaLoader generic sina-based loader
        """
        super(KoshSinaLoader, self).__init__(obj, **kargs)

    def open(self, *args, **kargs):
        """open the object
        """
        record = self.obj.__store__.get_record(self.obj.id)
        if record["type"] not in self.obj.__store__._kosh_reserved_record_types:
            return KoshDataset(
                self.obj.id, store=self.obj.__store__, record=record)
        if record["type"] == "file":
            return KoshSinaFile(
                self.obj.id, store=self.obj.__store__, record=record)
        else:
            return KoshSinaObject(self.obj.id, record["type"], protected=[
            ], record_handler=self.obj.__store__.__record_handler__, record=record)

    def list_features(self):
        record = self.obj.__store__.get_record(self.obj.id)
        # Using set in case a variable is both in independent and dependent
        # Dependent would win when getting the data
        curves = set()
        for curve in record["curve_sets"]:
            curves.add(curve)
            for curve_type in ["independent", "dependent"]:
                for name in record["curve_sets"][curve][curve_type]:
                    curves.add("{}/{}".format(curve, name))
        return sorted(curves)

    def extract(self, *args, **kargs):
        features = self.feature
        if not isinstance(features, list):
            features = [self.feature, ]
        record = self.obj.__store__.get_record(self.obj.id)
        out = []
        for feature in features:
            sp = feature.split("/")
            # Here we are assuming the curve root name cannot have "/" in it
            curve_root = record["curve_sets"][sp[0]]
            if len(sp) > 1:
                curve_name = "/".join(sp[1:])
                if curve_name in curve_root["dependent"]:
                    curve = curve_root["dependent"][curve_name]["value"]
                else:
                    curve = curve_root["independent"][curve_name]["value"]
                out.append(numpy.array(curve))
            else:
                # we want all curves
                all = []
                # Matching order (indep/dep) that we used in list_features
                for curve_type in ["independent", "dependent"]:
                    # Same order as list_features()
                    for curve_name in sorted(curve_root[curve_type].keys()):
                        curve = curve_root[curve_type][curve_name]["value"]
                        all.append(numpy.array(curve))
                out.append(all)

        if not isinstance(self.feature, list):
            return out[0]
        else:
            return out


class KoshStore(object):
    """Kosh store, relies on Sina"""

    def __init__(self, db_uri=None, username=os.environ["USER"], db=None,
                 keyspace=None, sync=True, dataset_record_type="dataset",
                 verbose=True, use_lock_file=False, kosh_reserved_record_types=[],
                 read_only=False, allow_connection_pooling=False):
        """__init__ initialize a new Sina-based store

        :param db: type of database, defaults to 'sql', can be 'cass'
        :type db: str, optional
        :param username: user name defautl to user id
        :type username: str
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
        :param kosh_reserved_record_types: list of record types that are reserved for Kosh internal
                                           use, will be ignored when searching store
        :type kosh_reserved_record_types: list of strings
        :param read_only: Can we modify the database source?
        :type read_only: bool
        :param allow_connection_pooling: Allow "pooling" behavior that recycles connections,
                                        which may prevent them from closing fully on .close().
                                        Only used for the sql backend.
        :type allow_connection_pooling: bool
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

        rec = update_store_and_get_info_record(self.__sina_store.records)

        self._sources_type = rec["data"]["sources_type"]["value"]
        self._users_type = rec["data"]["users_type"]["value"]
        self._groups_type = rec["data"]["groups_type"]["value"]
        self._loaders_type = rec["data"]["loaders_type"]["value"]
        self._kosh_reserved_record_types = kosh_reserved_record_types + \
            rec["data"]["reserved_types"]["value"]

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
        mem = sina_connect(None)
        self._added_unsync_handler = mem.records
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
            self._added_unsync_handler.delete(Id)
            if Id in self.__sync__dict__:
                del(self.__sync__dict__[Id])
                self.__sync__deleted__[Id] = rec
                rec["user_defined"]["deleted_time"] = time.time()
        else:
            self.__record_handler__.delete(Id)

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
        :param sina_type: If you want to query the store for a specific sina record type, not just a datset
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
            self._added_unsync_handler.insert(rec)
        try:
            ds = KoshDataset(Id, store=self, schema=schema, record=rec)
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
        if record["type"] not in self._kosh_reserved_record_types and uri is None:
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
                    feats = ld.list_features()
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
                    feats = ld.list_features()
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
        if record["type"] == "file":
            return KoshSinaFile(Id, koshType=record["type"],
                                record_handler=self.__record_handler__,
                                store=self, record=record)
        else:
            return KoshSinaObject(Id, koshType=record["type"],
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

    def find(self, *atts, **keys):
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
            # it makes things much d=faster
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
            record_types = self.__record_handler__.get_types(
            ) + self._added_unsync_handler.get_types()
            for rec_type in self._kosh_reserved_record_types:
                if rec_type in record_types:
                    record_types.remove(rec_type)
            if record_types == []:
                # Ok this stores is essentially empty
                # Before we go back turn sync mode back to what it was
                if mode:
                    # we need to restore sync mode
                    self.__sync__dict__ = backup
                    self.synchronous()
                return

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

        # is it a blak search, e.g get me everything?
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
                    self._added_unsync_handler.get_all(
                        ids_only=True))
            else:
                match_mem = set(self._added_unsync_handler.find(**sina_kargs))
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
        :param file: optional file to dump datset to
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
        """import datasets that were exported from another store, or load them from a json file
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
                                    will be passed in addtion to this store dataset and the imported dataset.
        :type merge_handler_kargs: dict
        :return: list of datasets
        :rtype: list of KoshSinaDataset
        """
        if isinstance(datasets, str):
            with open(datasets) as f:
                from_file = orjson.loads(f.read())
                records_in = from_file.get("records", [])
        elif isinstance(datasets, dict):
            from_file = datasets
            records_in = from_file["records"]
        elif isinstance(datasets, KoshDataset):
            from_file = datasets.export()
            records_in = from_file["records"]

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
                    raise ValueError("dataset criterias: {} matches multiple ({}) "
                                     "datasets in store {}, try changing 'match_attributes' when calling"
                                     " this function".format(
                                         match_dict, len(matching), self.db_uri))
                elif len(matching) == 1:
                    # All right we do have a possible conflict here
                    match = matching[0]
                    merged_attributes = merge_handler(
                        match, atts, **merge_handler_kargs)
                    # Ok at this point no conflict!
                    match.update(merged_attributes)
                    match_rec = match.get_record()
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
                    if match_rec.raw[section] != record[section]:
                        if merge_handler != merge_datasets_handler:
                            raise RuntimeError("We do not know how to merge curves with custom merge handler")
                        if merge_handler_kargs["handling_method"] == "conservative":
                            raise RuntimeError("{} section do not match aborting under conservative merge option")
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
                            raise RuntimeError("We do not know how to merge curves with custom merge handler")
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
                self.__record_handler__.delete(match_rec["id"])
            except ValueError:
                pass
            self.__record_handler__.insert(match_rec)
            matches.append(match_rec["id"])
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

    def cleanup_files(self, dry_run=False, interactive=False,
                      **dataset_search_keys):
        """Cleanup the store from references to dead files
        You can filter associated objects for each dataset by passing key=values
        e.g mime_type=hdf5 will only dissociate non-existing files associated with mime_type hdf5
        some_att=some_val will only dissociate non-exisiting files associated and having the attribute
        'some_att' with value of 'some_val'
        returns list of uris to be removed.
        :param dry_run: Only does a dry_run
        :type dry_run: bool
        :param interactive: interactive mode, ask before dissociating
        :type interactive: bool
        :returns: list of uris (to be) removed.
        :rtype: list
        """
        missings = []
        datasets = self.find()
        for dataset in datasets:
            missings += dataset.cleanup_files(dry_run=dry_run,
                                              interactive=interactive, **dataset_search_keys)
        return missings

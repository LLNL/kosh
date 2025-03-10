import uuid
import warnings
import time
import reprlib
from .schema import KoshSchema
from .utils import KoshPickler
from . import lock_strategies


kosh_pickler = KoshPickler()


def __getattr__(name):
    if name == "sina_version":
        from sina import get_version
        return float(".".join(get_version().split(".")[:2]))


class KoshSinaObject(object):
    """KoshSinaObject Base class for sina objects
    """
    @lock_strategies.lock_method
    def get_record(self):
        return self.__store__.get_record(self.id)

    def __init__(self, Id, store, kosh_type,
                 record_handler, protected=[], metadata={}, schema=None,
                 record=None):
        from sina.model import Record
        """__init__ sina object base class

        :param Id: id to use for unique identification, if None is passed set for you via uui4()
        :type Id: str
        :param store: Kosh store associated
        :type store: KoshSinaStore
        :param kosh_type: type of Kosh object (dataset, file, project, ...)
        :type kosh_type: str
        :param record_handler: sina record handler object
        :type record_handler: RecordDAO
        :param protected: list of protected parameters, e.g internal params not to be stored
        :type protected: list, optional
        :param metadata: dictionary of attributes/value to initialize object with, defaults to {}
        :type metadata: dict, optional
        :param record: sina record to prevent looking it up again and again in sina
        :type record: Record
        """
        with store.lock_strategy:
            self.__dict__["__store__"] = store
            self.__dict__["__schema__"] = schema
            self.__dict__["__record_handler__"] = record_handler
            self.__dict__["__protected__"] = [
                "id", "__type__", "__protected__",
                "__record_handler__", "__store__", "id", "__schema__"] + protected
            self.__dict__["__type__"] = kosh_type
            self.__dict__["lock_strategy"] = store.lock_strategy
            if Id is None:
                Id = uuid.uuid4().hex
                record = Record(id=Id, type=kosh_type, user_defined={'kosh_information': {}})
                if store.__sync__:
                    store.lock()
                    store.__record_handler__.insert(record)
                    store.unlock()
                else:
                    record["user_defined"]['kosh_information']["last_update_from_db"] = time.time()
                    self.__store__.__sync__dict__[Id] = record
                self.__dict__["id"] = Id
            else:
                self.__dict__["id"] = Id
                if record is None:
                    try:
                        record = self.get_record()
                    except BaseException:  # record exists nowhere
                        record = Record(id=Id, type=kosh_type, user_defined={'kosh_information': {}})
                        if store.__sync__:
                            store.lock()
                            store.__record_handler__.insert(record)
                            store.unlock()
                        else:
                            self.__store__.__sync__dict__[Id] = record
                            record["user_defined"]['kosh_information']["last_update_from_db"] = time.time()
                else:
                    deleted_items = False
                    for att in self.__dict__["__protected__"]:
                        if att in record["data"]:
                            del record["data"][att]
                            deleted_items = True
                    if deleted_items:
                        if store.__sync__:
                            self._update_record(record)
                        else:
                            self.__store__.__sync__dict__[Id] = record

            metadata_copy = metadata.copy()
            for key in metadata:
                if key in self.__dict__["__protected__"]:
                    del metadata_copy[key]
            self.update(metadata_copy)

    @lock_strategies.lock_method
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
            if name == "_associated_datasets_" and self.__type__ == self.__store__._ensembles_type:
                rels = self.get_sina_store().relationships.find(
                    None, "is a member of ensemble", self.id)
                return [str(x.subject_id) for x in rels]
            if name == "__features__":
                record = self.get_record()
                try:
                    return KoshPickler().loads(record["user_defined"]["__features__"])
                except Exception:
                    return {None: {}}
            if name == "_associated_data_":
                from kosh.dataset import KoshDataset
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
                # Now we need to add the parent ensembles associated data
                if isinstance(self, KoshDataset):
                    for ensemble in self.get_ensembles():
                        out += ensemble._associated_data_
                return out
            else:
                return self.__dict__[name]
        record = self.get_record()
        if name == "__attributes__":
            return self.__getattributes__()
        elif name == "schema":
            if self.__dict__[
                    "__schema__"] is None and "schema" in record["data"]:
                schema = kosh_pickler.loads(record["data"]["schema"]["value"])
                self.__dict__["__schema__"] = schema
            return self.__dict__["__schema__"]
        elif name in ['alias_feature', 'loader_kwargs']:
            if name in record["data"]:
                return kosh_pickler.loads(record["data"][name]["value"])
            else:
                return {}
        if name not in record["data"]:
            if name == "mime_type":
                return record["type"]
            elif name == "uri":
                return ""
            else:
                raise AttributeError(
                    "Object {} does not have {} attribute".format(self.id,
                                                                  name))
        value = record["data"][name]["value"]
        if name == "creator":
            # old records have user id let's fix this
            if value in self.__store__.__record_handler__.find_with_type(
                    self.__store__._users_type, ids_only=True):
                value = self.__store__.get_record(value)["data"]["username"]["value"]
        return value

    @lock_strategies.lock_method
    def get_sina_store(self):
        """Returns the sina store object"""
        return self.__store__.get_sina_store()

    @lock_strategies.lock_method
    def get_sina_records(self):
        """Returns sina store's records"""
        return self.__record_handler__

    @lock_strategies.lock_method
    def update(self, attributes):
        """update many attributes at once to limit db writes
        :param: attributes: dictionary with attributes to update
        :type attributes: dict
        """
        if 'id' in attributes:
            del attributes['id']
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

    @lock_strategies.lock_method
    def __setattr__(self, name, value):
        """set an attribute
        We are calling the ___setattr___
        because of special case that needs extra args and return values
        """
        self.___setattr___(name, value)

    @lock_strategies.lock_method
    def ___setattr___(self, name, value, record=None,
                      update_db=True, force=False):
        """__setattr__ set an attribute on an object

        :param name: name of attribute
        :type name: str
        :param value: value to set attribute to
        :type value: object
        :param record: sina record if already extracted before, save db access
        :type record: sina.model.Record
        :param force: force dataset attribute setting (when sent from ensemble)
        :type force: bool
        :return: sina record updated
        :rtype: sina.model.Record
        """
        if name in self.__protected__:  # Cannot set protected attributes
            return record
        if record is None:
            record = self.get_record()
        if name == "schema":
            assert isinstance(value, KoshSchema)
            value.validate(self)
        elif name == "__features__":
            value = kosh_pickler.dumps(value)
            record["user_defined"]["__features__"] = value
        elif name in ['alias_feature', 'loader_kwargs']:
            if isinstance(value, dict):
                value = kosh_pickler.dumps(value)
            else:  # Pre-pickled at dataset level
                value = value
        elif self.schema is not None:
            self.schema.validate_attribute(name, value)

        # For datasets we need to check if the att comes from ensemble
        from kosh.dataset import KoshDataset
        if isinstance(self, KoshDataset) and not force:
            sina_store = self.get_sina_store()
            # Let's get the relationships it's in
            relationships = sina_store.relationships.find(
                self.id, self.__store__._ensemble_predicate, None)
            for relationship in relationships:
                ensemble = self.__store__.open(relationship.object_id)
                ens_tags = self.list_ensemble_tags(ensemble.id, dictionary=True, obscure=False)
                inherit_attributes = ens_tags.get(f"{ensemble.id}_ENSEMBLE_TAG_INHERIT_ATTRIBUTES", True)
                if inherit_attributes:
                    if name in ensemble.list_attributes() and name not in ensemble.__dict__["__ok_duplicates__"]:
                        if value != getattr(ensemble, name):
                            raise KeyError(
                                "The attribute {} is controlled by ensemble: {} and cannot be set here".format(
                                    name, relationship.object_id))
                        else:
                            warnings.warn(
                                "The attribute {} is controlled by ensemble: {}"
                                ". You should NOT set this attribute at the dataset level"
                                ". Values match so we will accept it here".format(
                                    name, relationship.object_id), UserWarning)

        # For Ensembles we need to set it on all members
        from kosh.ensemble import KoshEnsemble
        if isinstance(self, KoshEnsemble):
            # First we make a pass to collect all other ensembles datasets are
            # part of
            for dataset in self.get_members():
                for ensemble in dataset.get_ensembles():
                    ens_tags = dataset.list_ensemble_tags(ensemble.id, dictionary=True, obscure=False)
                    inherit_attributes = ens_tags.get(f"{ensemble.id}_ENSEMBLE_TAG_INHERIT_ATTRIBUTES", True)
                    if inherit_attributes:
                        if ensemble.id != self.id:
                            for att in ensemble.list_attributes():
                                if att in self.__dict__["__ok_duplicates__"]:
                                    continue
                                if att == name:
                                    raise NameError("A member of this ensemble belongs to ensemble {} "
                                                    "which already controls attribute {}".format(ensemble.id, att))

                        dataset.___setattr___(
                            name=name,
                            value=value,
                            record=None,
                            update_db=update_db,
                            force=True)

        # Did it change on db since we last read it?
        last_modif_att = "{name}_last_modified".format(name=name)
        try:
            # Time we last read its value
            last = self.__dict__[last_modif_att]
        except KeyError:
            last = time.time()
        try:
            # Time we last read its value
            last_db = record["user_defined"]['kosh_information'][last_modif_att]
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
        record["user_defined"]['kosh_information'][last_modif_att] = now
        if name == "schema":
            self.__dict__["__schema__"] = value
            value = kosh_pickler.dumps(value)
        record["data"][name] = {"value": value}
        if update_db and self.__store__.__sync__:
            self._update_record(record)
        return record

    @lock_strategies.lock_method
    def _update_record(self, record, store=None):
        """Updates a record in the sina store
        :param record: The record to update
        :type record: sina.model.Record
        :param store: sina store to update
        :type store: sina.datastore.DataStore"""
        self.__store__.lock()
        if store is None:
            store = self.__store__.get_sina_store()
        id_ = record.id
        rels = store.relationships.find(id_, None, None)
        rels += store.relationships.find(None, None, id_)
        if not self.__store__.__sync__:
            store.records.delete(id_)
            store.records.insert(record)
            store.relationships.insert(rels)
        store.records.update(record)
        self.__store__.unlock()

    @lock_strategies.lock_method
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
        record["user_defined"]['kosh_information'][last_modif_att] = now
        # We need to remember we touched it otherwise
        # if we create it again the db will look
        # out of sync.
        self.__dict__[last_modif_att] = now
        del record["data"][name]
        if self.__store__.__sync__:
            self._update_record(record)

    @lock_strategies.lock_method
    def sync(self):
        """sync this object with database"""
        self.__store__.sync([self.id, ])

    @lock_strategies.lock_method
    def list_attributes(self, dictionary=False, ensemble_id=None):
        __doc__ = self.listattributes.__doc__.replace("listattributes", "list_attributes")  # noqa
        return self.listattributes(dictionary=dictionary, ensemble_id=ensemble_id)

    @lock_strategies.lock_method
    def listattributes(self, dictionary=False, ensemble_id=None):
        """listattributes list all non protected attributes

        :param dictionary: return a dictionary of value/pair rather than just attributes names
        :type dictionary: bool
        :param ensemble_id: Provide ensemble ID(s) to return ensemble tags
        :type ensemble_id: str, lst

        :return: list of attributes set on object
        :rtype: list
        """
        record = self.get_record()
        attributes = list(record["data"].keys()) + ['id', ]

        no_tags = [a for a in attributes if "_ENSEMBLE_TAG_" not in a]  # Remove ensemble tags
        ens_tags = []
        if ensemble_id is not None:
            if isinstance(ensemble_id, str):
                ensemble_id = [ensemble_id]
            for ens_id in ensemble_id:
                ens_tags += sorted([a for a in attributes if f"{ens_id}_ENSEMBLE_TAG_" in a])
        attributes = sorted(no_tags)

        for att in self.__protected__:
            if att in attributes and att != "id":
                attributes.remove(att)
        attributes += ens_tags
        if dictionary:
            out = {}
            for att in attributes:
                out[att] = getattr(self, att)
            return out
        else:
            return attributes

    @lock_strategies.lock_method
    def __getattributes__(self):
        """__getattributes__ return dictionary with pairs of attribute/value

        :return: dictionary with pairs of attribute/value
        :rtype: dict
        """
        record = self.get_record()
        attributes = {}
        for a in record["data"]:
            if a in ['alias_feature', 'loader_kwargs']:
                continue
            attributes[a] = record["data"][a]["value"]
            if a == "creator":
                # old records have user id let's fix this
                if attributes[a] in self.__store__.__record_handler__.find_with_type(
                        self.__store__._users_type, ids_only=True):
                    attributes[a] = self.__store__.get_record(attributes[a])["data"]["username"]["value"]
        return attributes

    @lock_strategies.lock_method
    def __str__(self):
        """String for printing"""
        if self.__dict__["__store__"].verbose_attributes:
            def reprtool(item):
                return item
        else:
            def reprtool(item):
                if isinstance(item, str):
                    return reprlib.repr(item)[1:-1]
                else:
                    return reprlib.repr(item)
        st = "Id: {}".format(self.id)
        for att in sorted(self.listattributes()):
            if att != 'id':
                st += "\n\t{}: {}".format(att, reprtool(getattr(self, att)))
        return st


class KoshSinaFile(KoshSinaObject):
    """KoshSinaFile file representation in Kosh via Sina"""

    @lock_strategies.lock_method
    def open(self, *args, **kargs):
        """open opens the file
        :return: handle to file in open mode
        """
        return self.__store__.open(self.id, *args, **kargs)

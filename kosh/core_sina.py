import uuid
import warnings
import time
from .schema import KoshSchema
from sina.model import Record
from sina import get_version
import pickle


sina_version = float(".".join(get_version().split(".")[:2]))


class KoshSinaObject(object):
    """KoshSinaObject Base class for sina objects
    """

    def get_record(self):
        return self.__store__.get_record(self.id)

    def __init__(self, Id, store, kosh_type,
                 record_handler, protected=[], metadata={}, schema=None,
                 record=None):
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
        self.__dict__["__store__"] = store
        self.__dict__["__schema__"] = schema
        self.__dict__["__record_handler__"] = record_handler
        self.__dict__["__protected__"] = [
            "id", "__type__", "__protected__",
            "__record_handler__", "__store__", "id", "__schema__"] + protected
        self.__dict__["__type__"] = kosh_type
        if Id is None:
            Id = uuid.uuid4().hex
            record = Record(id=Id, type=kosh_type)
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
                    record = Record(id=Id, type=kosh_type)
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
            if name == "_associated_datasets_" and self.__type__ == self.__store__._ensembles_type:
                rels = self.get_sina_store().relationships.find(
                    None, "is a member of ensemble", self.id)
                return [str(x.subject_id) for x in rels]
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
            assert(isinstance(value, KoshSchema))
            value.validate(self)
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
                if name in ensemble.list_attributes() and name not in ensemble.__dict__["__ok_duplicates__"]:
                    raise KeyError(
                        "The attribute {} is controlled by ensemble: {} and cannot be set here".format(
                            name, relationship.object_id))

        # For Ensembles we need to set it on all members
        from kosh.ensemble import KoshEnsemble
        if isinstance(self, KoshEnsemble):
            # First we make a pass to collect all other ensembles datasets are
            # part of
            other_ensembles = set()
            for dataset in self.get_members():
                for e in dataset.get_ensembles():
                    other_ensembles.add(e)
            for ensemble in other_ensembles:
                if ensemble.id == self.id:
                    continue
                for att in ensemble.list_attributes():
                    if att in self.__dict__["__ok_duplicates__"]:
                        continue
                    if att == name:
                        raise NameError("A member of this ensemble belongs to ensemble {} "
                                        "which already controls attribute {}".format(ensemble.id, att))
            for dataset in self.get_members():
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
            self._update_record(record)
        return record

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
        store.records.delete(id_)
        store.records.insert(record)
        store.relationships.insert(rels)
        self.__store__.unlock()

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
            self._update_record(record)

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

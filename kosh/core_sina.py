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

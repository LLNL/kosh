import warnings
import sina
from .dataset import KoshDataset
from .utils import cleanup_sina_record_from_kosh_sync
from .utils import update_json_file_with_records_and_relationships
try:
    import orjson
except ImportError:
    import json as orjson  # noqa
import kosh


class KoshEnsemble(KoshDataset):
    def __init__(self, id, store, schema=None, record=None):
        """Kosh Ensemble
Ensemble allows to link together many datasets.
These datasets will inherit attributes and associated sources from the ensemble.

        :param id: dataset's unique Id
        :type id: str
        :param store: store containing the dataset
        :type store: KoshSinaStore
        :param schema: Kosh schema validator
        :type schema: KoshSchema
        :param record: to avoid looking up in sina pass sina record
        :type record: Record
        """
        super(KoshEnsemble, self).__init__(id, store,
                                           schema=schema, record=record,
                                           kosh_type=store._ensembles_type)
        self.__dict__["__protected__"] = ["__name__", "__creator__", "__store__",
                                          "_associated_data_", "__features__",
                                          "_associated_datasets_", "__ok_duplicates__"]
        # Attributes that the members can have on their own
        self.__dict__["__ok_duplicates__"] = ["creator", "id", "name"]

    def __str__(self):
        """string representation"""
        st = super(KoshEnsemble, self).__str__()
        st = st.replace("KOSH DATASET", "KOSH ENSEMBLE")
        st = st[:st.find("--- Ensembles") - 1]
        if self._associated_datasets_ is not None:
            st += "\n--- Member Datasets ({})---\n".format(
                len(self._associated_datasets_))
            st += "\t{}".format(self._associated_datasets_)
        return st

    def cleanup_files(self, dry_run=False, interactive=False, **search_keys):
        """Cleanup the ensemble's members from references to dead files.
        You can filter associated objects by passing key=values
        e.g mime_type=hdf5 will only dissociate non-existing files associated with mime_type hdf5
        some_att=some_val will only dissociate non-existing files associated and having the attribute
        'some_att' with value of 'some_val'
        returns list of uris to be removed.
        :param dry_run: Only does a dry_run
        :type dry_run: bool
        :param interactive: interactive mode, ask before dissociating
        :type interactive: bool
        :returns: list of uris (to be) removed.
        :rtype: list
        """
        missings = super(
            KoshEnsemble,
            self).cleanup_files(
            dry_run=dry_run,
            interactive=interactive,
            **search_keys)
        for dataset in self.get_members():
            missings += dataset.cleanup_files(dry_run=dry_run,
                                              interactive=interactive, **search_keys)
        return missings

    def export(self, file=None):
        """Exports this ensemble datasets
        :param file: export datasets to a file
        :type file: None or str
        :return: dataset and its associated data
        :rtype: dict"""
        records = [cleanup_sina_record_from_kosh_sync(self.get_record()), ]
        for dataset_id in self.get_members(ids_only=True):
            records.append(
                cleanup_sina_record_from_kosh_sync(
                    self.__store__.get_record(dataset_id)))
        # We also need to export the relationships
        rels = self.get_sina_store().relationships.find(
            None, self.__store__._ensemble_predicate, self.id)
        relationships = []
        for rel in rels:
            relationships.append(rel.to_json())
        output_dict = {
            "minimum_kosh_version": None,
            "kosh_version": kosh.version(comparable=True),
            "sources_type": self.__store__._sources_type,
            "records": records,
            "relationships": relationships
        }

        update_json_file_with_records_and_relationships(file, output_dict)
        return output_dict

    def create(self, name="Unnamed Dataset", id=None,
               metadata={}, schema=None, sina_type=None, **kargs):
        """create a new (possibly named) dataset as a member of this ensemble.

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
        if sina_type == self.__store__._ensembles_type:
            raise ValueError("You cannot create an ensemble from an ensemble")

        attributes = self.list_attributes()
        for key in metadata:
            if key in attributes:
                raise ValueError(
                    "'{}' is an attribute of this ensemble and "
                    "therefore cannot be an attribute of its descendants".format(key))
        ds = self.__store__.create(
            name=name,
            id=id,
            metadata=metadata,
            schema=schema,
            sina_type=sina_type,
            **kargs)
        self.add(ds)
        return ds

    def add(self, dataset):
        """Adds a dataset to this ensemble
        :param dataset: The dataset to add to this ensemble""
        :type dataset: KoshDataset or str
        """
        # Step1 make sure the dataset does not belong to another ensemble
        if isinstance(dataset, KoshDataset):
            dataset_id = dataset.id
        else:
            dataset_id = dataset
            dataset = self.__store__._load(dataset_id)
        relationships = self.get_sina_store().relationships.find(
            dataset_id, self.__store__._ensemble_predicate, None)
        for rel in relationships:
            if rel.object_id != self.id:
                other_ensemble = self.__store__.open(rel.object_id)
                # Ok... Already a member of another ensemble.
                # let's make sure there are no conflict here
                for att in self.list_attributes():
                    if att in self.__dict__["__ok_duplicates__"]:
                        continue
                    if att in other_ensemble.list_attributes():
                        raise ValueError(
                            "Dataset {} is already part of ensemble {} "
                            "which already provides support for attribute: {}. Bailing".format(
                                dataset_id, rel.object_id, att))
            else:
                # ok it's already done, no need to do anything else
                return

        # Ok we're good, let's now makes sure attributes are ok
        attributes = self.list_attributes(dictionary=True)
        dataset_attributes = dataset.list_attributes(dictionary=True)
        for att in dataset.list_attributes():
            if att in self.__dict__["__ok_duplicates__"]:
                continue
            if att in attributes and dataset_attributes[att] != attributes[att]:
                raise ValueError(
                    "Dataset {} has attribute `{}` with value {}, this ensemble ({}) has value `{}`".format(
                        dataset_id, att, dataset_attributes[att], self.id, attributes[att]))
        # At this point we need to add the ensemble attributes to the dataset
        for att in self.list_attributes():
            if att in self.__dict__["__ok_duplicates__"]:
                continue
            dataset.___setattr___(att, getattr(self, att), force=True)
        # Ok We are clear let's create the relationship
        rel = sina.model.Relationship(
            self.id, dataset_id, self.__store__._ensemble_predicate)
        self.get_sina_store().relationships.insert(rel)

    def remove(self, dataset):
        """Removes a dataset from this ensemble. Does not delete the dataset.
        :param dataset: The dataset to remove
        :type dataset: KoshDataset or str
        """
        # Step1 make sure the dataset does not belong to another ensemble
        if isinstance(dataset, KoshDataset):
            dataset_id = dataset.id
        else:
            dataset_id = dataset
            dataset = self.__store__._load(dataset_id)
        relationships = self.get_sina_store().relationships.find(
            dataset_id, self.__store__._ensemble_predicate, self.id)
        if len(relationships) == 0:
            warnings.warn(
                "Dataset {} is not a member of ensemble {}".format(
                    dataset_id, self.id))
            return

        rel = relationships[0]
        self.get_sina_store().relationships.delete(rel.subject_id, rel.predicate, rel.object_id)

    delete = remove

    def get_members(self, ids_only=False):
        """Generator for member datasets
        :param ids_only: generator will return ids if True Kosh datasets otherwise
        :type ids_only: bool
        :returns: generator of dataset (or ids)
        :rtype: str or KoshDataset
        """
        for id in self._associated_datasets_:
            if ids_only:
                yield id
            else:
                yield self.__store__.open(id)

    def find_datasets(self, *atts, **keys):
        """Find datasets members of this ensemble that are matching some metadata.
        Arguments are the metadata names we are looking for e.g
        find("attr1", "attr2")
        you can further restrict by specifying exact value for a metadata
        via key=value
        you can return ids only by using: ids_only=True
        range can be specified via: sina.utils.DataRange(min, max)

        :return: generator of matching datasets in this ensemble
        :rtype: generator
        """

        members_ids = list(self.get_members(ids_only=True))
        return self.__store__.find(id_pool=members_ids, *atts, **keys)

    def clone(self, *atts, **keys):
        """We cannot clone an ensemble"""
        raise NotImplementedError("Ensembles objects cannot clone themselves")

import warnings
from .dataset import KoshDataset
from .utils import cleanup_sina_record_from_kosh_sync
from .utils import update_json_file_with_records_and_relationships
from .utils import __check_valid_connection_type__
from . import lock_strategies
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
        with store.lock_strategy:
            super(KoshEnsemble, self).__init__(id, store,
                                               schema=schema, record=record,
                                               kosh_type=store._ensembles_type)
            self.__dict__["__protected__"] = ["__name__", "__creator__", "__store__",
                                              "_associated_data_", "__features__",
                                              "_associated_datasets_", "__ok_duplicates__"]
            # Attributes that the members can have on their own
            self.__dict__["__ok_duplicates__"] = ["creator", "id", "name"]

    @lock_strategies.lock_method
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

    @lock_strategies.lock_method
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
        __check_valid_connection_type__(self.__store__.__connection_type__, ['write'])
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

    @lock_strategies.lock_method
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
        relationships = self.get_sina_store().relationships.find(
            None, self.__store__._ensemble_predicate, self.id)
        output_dict = {
            "minimum_kosh_version": None,
            "kosh_version": kosh.version(comparable=True),
            "sources_type": self.__store__._sources_type,
            "records": records,
            "relationships": relationships
        }

        update_json_file_with_records_and_relationships(file, output_dict)
        return output_dict

    @lock_strategies.lock_method
    def create(self, name="Unnamed Dataset", id=None,
               metadata={}, schema=None, sina_type=None, ensemble_tags=None,
               inherit_attributes=True, **kargs):
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
        :param inherit_attributes: Whether datasets inherit attributes from ensembles.
                                   If False, datasets can have the same attributes as ensembles
                                   and the different ensembles that the dataset belongs to can also
                                   have the same attributes. They can also have different values.
                                   Defaults to True.
        :type inherit_attributes: bool
        :param ensemble_tags: Organize datasets within ensemble by using tags. These "attributes"
                              are only available at the ensemble level for each dataset. These tags
                              can also be used in `ensemble.find_datasets(ensemble_tags=dict)`.
                              e.g., ensemble_tags={"even_or_odd": "even", "data_type": "test data"}
        :type ensemble_tags: dict
        :param kargs: extra keyword arguments (ignored)
        :type kargs: dict
        :raises RuntimeError: Dataset already exists
        :return: KoshDataset
        :rtype: KoshDataset
        """
        __check_valid_connection_type__(self.__store__.__connection_type__, ['write', 'append'])
        if sina_type == self.__store__._ensembles_type:
            raise ValueError("You cannot create an ensemble from an ensemble")

        attributes = self.list_attributes()
        for key in metadata:
            if key in attributes:
                if metadata[key] != getattr(self, key):
                    raise ValueError(
                        "'{}' is an attribute of this ensemble and "
                        "therefore cannot be an attribute of its descendants".format(key))
                else:
                    warnings.warn(
                        "'{}' is an attribute of this ensemble and "
                        "therefore cannot be an attribute of its descendants"
                        ". Values match so we will accept it here.".format(key), UserWarning)

        ds = self.__store__.create(
            name=name,
            id=id,
            metadata=metadata,
            schema=schema,
            sina_type=sina_type,
            **kargs)
        self.add(ds, inherit_attributes=inherit_attributes, ensemble_tags=ensemble_tags)
        return ds

    @lock_strategies.lock_method
    def add(self, dataset, inherit_attributes=True, ensemble_tags=None):
        """Adds a dataset to this ensemble
        :param dataset: The dataset to add to this ensemble
        :type dataset: KoshDataset or str
        :param inherit_attributes: Whether datasets inherit attributes from ensembles.
                                   If False, datasets can have the same attributes as ensembles
                                   and the different ensembles that the dataset belongs to can also
                                   have the same attributes. They can also have different values.
                                   Defaults to True.
        :type inherit_attributes: bool
        :param ensemble_tags: Organize datasets within ensemble by using tags. These "attributes"
                              are only available at the ensemble level for each dataset. These tags
                              can also be used in `ensemble.find_datasets(ensemble_tags=dict)`.
                              e.g., ensemble_tags={"even_or_odd": "even", "data_type": "test data"}
        :type ensemble_tags: dict
        """
        from sina.model import Relationship
        __check_valid_connection_type__(self.__store__.__connection_type__, ['write', 'append'])
        # Step1 make sure the dataset does not belong to another ensemble
        if isinstance(dataset, KoshDataset):
            dataset_id = dataset.id
        else:
            dataset_id = dataset
            dataset = self.__store__._load(dataset_id)
        relationships = self.get_sina_store().relationships.find(
            dataset_id, self.__store__._ensemble_predicate, None)

        if inherit_attributes:
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
                        f"Dataset {dataset_id} has attribute `{att}` with value {dataset_attributes[att]}, "
                        f"this ensemble ({self.id}) has value `{attributes[att]}`\n"
                        "This error can by bypassed by setting `inherit_attributes=False` which makes it so "
                        "datasets can have the same attributes as ensembles and the different ensembles that "
                        "the dataset belongs to can also have the same attributes.")
            # At this point we need to add the ensemble attributes to the dataset
            for att in self.list_attributes():
                if att in self.__dict__["__ok_duplicates__"]:
                    continue
                dataset.___setattr___(att, getattr(self, att), force=True)
        else:
            # Use for core_sina.py ___setattr___()
            if isinstance(ensemble_tags, dict):
                ensemble_tags['INHERIT_ATTRIBUTES'] = False
            else:
                ensemble_tags = {'INHERIT_ATTRIBUTES': False}
        # Ok We are clear let's create the relationship
        rel = Relationship(
            self.id, dataset_id, self.__store__._ensemble_predicate)
        # Add ensemble tags
        if ensemble_tags is not None:
            if self.schema is not None:
                self.schema.validate(ensemble_tags, f"Ensemble {self.id}")
            if dataset.schema is not None:
                dataset.schema.validate(ensemble_tags, f"Dataset {dataset.id}")
            dataset.add_ensemble_tags(self.id, ensemble_tags)
        self.get_sina_store().relationships.insert(rel)

    @lock_strategies.lock_method
    def remove(self, dataset):
        """Removes a dataset from this ensemble. Does not delete the dataset.
        :param dataset: The dataset to remove
        :type dataset: KoshDataset or str
        """
        __check_valid_connection_type__(self.__store__.__connection_type__, ['write'])
        # Step1 make sure the dataset does not belong to another ensemble
        if isinstance(dataset, KoshDataset):
            dataset_id = dataset.id
        else:
            dataset_id = dataset
            dataset = self.__store__.open(dataset_id)
        relationships = self.get_sina_store().relationships.find(
            dataset_id, self.__store__._ensemble_predicate, self.id)
        # Delete ensemble tags from dataset
        ensemble_tags = dataset.list_ensemble_tags(ensemble_id=self.id)
        ensemble_tags = [et.replace(f"{self.id}_ENSEMBLE_TAG_", "") for et in ensemble_tags]
        dataset.delete_ensemble_tags(ensemble_id=self.id, ensemble_tags=ensemble_tags)
        if len(relationships) == 0:
            warnings.warn(
                "Dataset {} is not a member of ensemble {}".format(
                    dataset_id, self.id))
            return

        rel = relationships[0]
        self.get_sina_store().relationships.delete(rel.subject_id, rel.predicate, rel.object_id)

    delete = remove

    @lock_strategies.lock_method
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

    @lock_strategies.lock_method
    def find_datasets(self, ensemble_tags=None, *atts, **keys):
        """Find datasets members of this ensemble that are matching some metadata.
        Arguments are the metadata names we are looking for e.g
        find("attr1", "attr2")
        you can further restrict by specifying exact value for a metadata
        via key=value
        you can return ids only by using: ids_only=True
        range can be specified via: sina.utils.DataRange(min, max)

        :param ensemble_tags: Further filter out datasets with ensemble tags
        :type ensemble_tags: dict
        :return: generator of matching datasets in this ensemble
        :rtype: generator
        """
        if ensemble_tags is not None:
            for key, val in ensemble_tags.items():
                keys[f"{self.id}_ENSEMBLE_TAG_{key}"] = val
        members_ids = list(self.get_members(ids_only=True))
        return self.__store__.find(id_pool=members_ids, *atts, **keys)

    @lock_strategies.lock_method
    def clone(self, *atts, **keys):
        """We cannot clone an ensemble"""
        __check_valid_connection_type__(self.__store__.__connection_type__, ['write', 'append'])
        raise NotImplementedError("Ensembles objects cannot clone themselves")

    @lock_strategies.lock_method
    def list_attributes(self, dictionary=False, no_duplicate=False):
        """list_attributes list all non protected attributes

        :param dictionary: return a dictionary of value/pair rather than just attributes names
        :type dictionary: bool

        :param no_duplicate: return only attributes that cannot be duplicated in members
        :type no_duplicate: bool

        :return: list of attributes set on object
        :rtype: list
        """

        attributes = super(KoshEnsemble, self).list_attributes(dictionary)
        if no_duplicate:
            return [x for x in attributes if x not in self.__dict__["__ok_duplicates__"]]
        else:
            return attributes

    @lock_strategies.lock_method
    def to_dataframe(self, data_columns=[], include_ensemble_attributes=True, include_ensemble_tags=True,
                     *atts, **keys):
        """Return the find_datasets object as a Pandas DataFrame.

        Pass in the same arguments and keyword arguments as the find method.

        Arguments are the metadata name we are looking for e.g
        find("attr1", "attr2")
        you can further restrict by specifying exact value for a metadata
        via key=value
        you can return ids only by using: ids_only=True
        range can be specified via: sina.utils.DataRange(min, max)

        "file_uri" is a reserved key that will return all records being associated
                   with the given "uri", e.g store.find(file_uri=uri)
        "types" let you search over specific sina record types only.
        "id_pool" will search based on id of Sina record or Kosh dataset. Can be a list.

        :param data_columns: Columns to extract. By default this will include ['id', 'name', 'creator'].
                             If nothing is passed, will return all data.
        :type data_columns: Union(str, list), optional
        :param include_ensemble_attributes: Include ensemble attributes in DataFrame.
        :type include_ensemble_attributes: bool, optional
        :param include_ensemble_tags: Include ensemble tags in DataFrame.
        :type include_ensemble_tags: bool, optional
        :return: Pandas DataFrame
        :rtype: Pandas DataFrame
        """
        import pandas as pd
        if isinstance(data_columns, str):
            data_columns = [data_columns]

        keys['load_type'] = 'dictionary'
        keys['ids_only'] = False
        datasets = list(self.find_datasets(*atts, **keys))

        attr_dict = {}
        total_datasets = len(datasets)

        # Always have these by default
        defaults = ['id', 'name', 'creator']

        unique_keys = set()
        for dataset in datasets:
            unique_keys.update(list(dataset['data'].keys()))
        data_columns_all = sorted(unique_keys)
        # Remove all ensemble tags and only keep dataset attributes
        data_columns_no_ensemble_tags = [dc for dc in data_columns_all if "_ENSEMBLE_TAG_" not in dc]

        # Acquire all data if `data_columns` was not passed
        if not data_columns:
            data_columns = data_columns_no_ensemble_tags

        data_columns = defaults + data_columns  # Want defaults in front

        # Acquire ensemble attributes
        if include_ensemble_attributes:
            data_columns_ens_default_attributes = [f"{self.id}_ENSEMBLE_ATTRIBUTE_id",
                                                   f"{self.id}_ENSEMBLE_ATTRIBUTE_name",
                                                   f"{self.id}_ENSEMBLE_ATTRIBUTE_creator"]

            ens_attrs = self.list_attributes(dictionary=True)
            data_columns_ens_other_attributes = [f"{self.id}_ENSEMBLE_ATTRIBUTE_{attr}" for attr in ens_attrs.keys()
                                                 if attr not in ['id', 'name', 'creator']]

            data_columns += data_columns_ens_default_attributes + data_columns_ens_other_attributes

        # Remove all other ensemble tags and only keep this ensemble's tags
        if include_ensemble_tags:
            data_columns += [dc for dc in data_columns_all if f"{self.id}_ENSEMBLE_TAG_" in dc and
                             "_ENSEMBLE_TAG_INHERIT_ATTRIBUTES" not in dc]

        attr_dict = {d: [pd.NA] * total_datasets for d in data_columns}

        for i, dataset in enumerate(datasets):
            attr_dict['id'][i] = dataset['id']
            for column, values in dataset['data'].items():
                if column in data_columns:
                    attr_dict[column][i] = values.get('value', pd.NA)

        # Ensemble attributes
        if include_ensemble_attributes:
            for key, val in ens_attrs.items():
                attr_dict[f"{self.id}_ENSEMBLE_ATTRIBUTE_{key}"] = [val] * total_datasets

        df = pd.DataFrame(attr_dict)
        return df

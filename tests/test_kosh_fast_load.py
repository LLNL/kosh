from __future__ import print_function
import koshbase
import os
import random
from datetime import datetime


def create_dataset(datastore, num):

    for i in range(num):
        ds = datastore.create(i, metadata={"param1": random.random() * 2.,
                                           "param2": random.random() * 1.5,
                                           "param3": random.random() * 5,
                                           "param4": random.random() * 3,
                                           "param5": random.random() * 2.5,
                                           "param6": chr(random.randint(65, 91)),
                                           })
        print(ds)
    return datastore


class TestKoshFastLoad(koshbase.KoshTest):

    def test_load_types(self):
        store, kosh_db = self.connect()

        start = datetime.now()
        store = create_dataset(store, 64)
        create_time = datetime.now()-start

        start = datetime.now()
        for dataset in store.find():
            dataset.param1
        dataset_time = datetime.now()-start

        start = datetime.now()
        for dataset in store.find(load_type='record'):
            dataset.param1
        record_time = datetime.now()-start

        start = datetime.now()
        for dataset in store.find(load_type='dictionary'):
            dataset['data']['param1']
        dictionary_time = datetime.now()-start

        print('\nCreate: ', create_time,)
        print('Dataset Attribute: ', dataset_time)
        print('Record Attribute: ', record_time)
        print('Dictionary Attribute: ', dictionary_time)

        self.assertGreater(create_time, dataset_time)
        self.assertGreater(dataset_time, record_time)
        self.assertGreater(record_time, dictionary_time)

        store.close()
        os.remove(kosh_db)

    def test_to_pandas(self):
        store, kosh_db = self.connect()

        store = create_dataset(store, 23)

        # Everything
        df = store.to_dataframe()
        print(df.columns)
        assert df.columns.values.tolist() == ['id', 'name', 'creator', 'creation_date', 'last_modified_date',
                                              'param1', 'param2', 'param3', 'param4', 'param5', 'param6']

        # Only certain columns
        df = store.to_dataframe(data_columns='param1')
        assert df.columns.values.tolist() == ['id', 'name', 'creator', 'creation_date', 'last_modified_date',
                                              'param1']

        df = store.to_dataframe(data_columns=['param1'])
        assert df.columns.values.tolist() == ['id', 'name', 'creator', 'creation_date', 'last_modified_date',
                                              'param1']

        df = store.to_dataframe(data_columns=['param1', 'param6'])
        assert df.columns.values.tolist() == ['id', 'name', 'creator', 'creation_date', 'last_modified_date',
                                              'param1', 'param6']

        # Everything with unique data
        store.create('new_dataset', metadata={'mynewattribute': 5})
        store.create('new_dataset2', metadata={'myotherattribute': 10})

        df = store.to_dataframe()
        assert df.columns.values.tolist() == ['id', 'name', 'creator', 'creation_date', 'last_modified_date',
                                              'mynewattribute', 'myotherattribute',
                                              'param1', 'param2', 'param3', 'param4', 'param5', 'param6']

        # Find data
        target_data = {'mynewattribute': 5}
        df = store.to_dataframe(data=target_data)
        for val in df["mynewattribute"].values:
            self.assertEqual(val, 5)
        assert df.shape == (1, 6)

        # Find data with missing columns
        target_data = {'mynewattribute': 5}
        df = store.to_dataframe(data=target_data, data_columns=['param1', 'param6'])
        assert df.shape == (1, 7)
        assert df.columns.values.tolist() == ['id', 'name', 'creator', 'creation_date', 'last_modified_date',
                                              'param1', 'param6']

    def test_dataset_to_pandas(self):

        store, kosh_db = self.connect()

        dataset = store.create()

        # hdf5
        dataset.associate("baselines/node_extracts2/node_extracts2.hdf5",
                          mime_type="hdf5",
                          metadata={"param10": "my value",
                                    "my other param": "Example Text"},
                          absolute_path=False)

        # csv
        dataset.associate("baselines/csv/my_csv_file.csv",
                          mime_type="pandas/csv",
                          metadata={"param10": "my value",
                                    "param20": "my other value",
                                    "my param": 10},
                          loader_kwargs={'index_col': 0},
                          absolute_path=False)

        # ultra
        dataset.associate("../examples/my_ult_file.ult",
                          metadata={"param30": 45,
                                    "my param": 560},
                          mime_type="ultra")

        # Everything
        df = dataset.to_dataframe()
        assert df.columns.values.tolist() == ['id', 'mime_type', 'uri', 'associated',
                                              'loader_kwargs', 'my other param', 'my param',
                                              'param10', 'param20', 'param30']

        # Only certain columns
        df = dataset.to_dataframe(data_columns='loader_kwargs')
        assert df.columns.values.tolist() == ['id', 'mime_type', 'uri', 'associated',
                                              'loader_kwargs']

        df = dataset.to_dataframe(data_columns=['loader_kwargs'])
        assert df.columns.values.tolist() == ['id', 'mime_type', 'uri', 'associated',
                                              'loader_kwargs']

        df = dataset.to_dataframe(data_columns=['loader_kwargs', 'my other param'])
        assert df.columns.values.tolist() == ['id', 'mime_type', 'uri', 'associated',
                                              'loader_kwargs', 'my other param']

        # Find data
        target_data = {'param10': "my value"}
        df = dataset.to_dataframe(data=target_data)
        for val in df["param10"].values:
            self.assertEqual(val, "my value")
        assert df.shape == (2, 9)

        # Find data with missing columns
        target_data = {'param10': "my value"}
        df = dataset.to_dataframe(data=target_data, data_columns=['param1', 'param6'])
        assert df.shape == (2, 6)
        assert df.columns.values.tolist() == ['id', 'mime_type', 'uri', 'associated',
                                              'param1', 'param6']

    def test_ensemble_to_pandas(self):
        store, kosh_db = self.connect()

        n_ensembles = 10
        n_datasets = 10

        datasets = []

        for i in range(n_datasets):
            metadata = {f"{ia}": f"dataset_{i}_attributes_{ia}" for ia in range(n_datasets)}
            if i % 2 == 0:
                metadata['my_dataset_attribute'] = 0
            else:
                metadata['my_dataset_attribute'] = 9
            ds = store.create(id=f"dataset_{i}", metadata=metadata)
            datasets.append(ds)

        ensembles = []
        for i in range(n_ensembles):
            metadata = {f"{ia}": f"ensemble_{i}_attributes_{ia}" for ia in range(n_ensembles)}
            ens = store.create_ensemble(id=f"ensemble_{i}", metadata=metadata)
            ensembles.append(ens)
            for j, ds in enumerate(datasets):
                ensemble_tags = {}
                if j % 2 == 0:
                    ensemble_tags[f"eoo{i}"] = "even"
                else:
                    ensemble_tags[f"eoo{i}"] = "odd"

                if j % 5 == 0:
                    ensemble_tags[f"data_type{i}"] = "test data"
                else:
                    ensemble_tags[f"data_type{i}"] = "train data"

                ens.add(ds, inherit_attributes=False, ensemble_tags=ensemble_tags)

        # Everything
        df = ensembles[0].to_dataframe()
        assert df.columns.values.tolist() == ['id', 'name', 'creator', 'creation_date', 'last_modified_date',
                                              # dataset attributes
                                              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'my_dataset_attribute',
                                              # ensemble attributes
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_id',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_name',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_creator',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_creation_date',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_last_modified_date',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_0', 'ensemble_0_ENSEMBLE_ATTRIBUTE_1',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_2', 'ensemble_0_ENSEMBLE_ATTRIBUTE_3',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_4', 'ensemble_0_ENSEMBLE_ATTRIBUTE_5',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_6', 'ensemble_0_ENSEMBLE_ATTRIBUTE_7',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_8', 'ensemble_0_ENSEMBLE_ATTRIBUTE_9',
                                              # ensemble tags
                                              'ensemble_0_ENSEMBLE_TAG_data_type0', 'ensemble_0_ENSEMBLE_TAG_eoo0']

        df = ensembles[1].to_dataframe()
        assert df.columns.values.tolist() == ['id', 'name', 'creator', 'creation_date', 'last_modified_date',
                                              # dataset attributes
                                              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'my_dataset_attribute',
                                              # ensemble attributes
                                              'ensemble_1_ENSEMBLE_ATTRIBUTE_id',
                                              'ensemble_1_ENSEMBLE_ATTRIBUTE_name',
                                              'ensemble_1_ENSEMBLE_ATTRIBUTE_creator',
                                              'ensemble_1_ENSEMBLE_ATTRIBUTE_creation_date',
                                              'ensemble_1_ENSEMBLE_ATTRIBUTE_last_modified_date',
                                              'ensemble_1_ENSEMBLE_ATTRIBUTE_0', 'ensemble_1_ENSEMBLE_ATTRIBUTE_1',
                                              'ensemble_1_ENSEMBLE_ATTRIBUTE_2', 'ensemble_1_ENSEMBLE_ATTRIBUTE_3',
                                              'ensemble_1_ENSEMBLE_ATTRIBUTE_4', 'ensemble_1_ENSEMBLE_ATTRIBUTE_5',
                                              'ensemble_1_ENSEMBLE_ATTRIBUTE_6', 'ensemble_1_ENSEMBLE_ATTRIBUTE_7',
                                              'ensemble_1_ENSEMBLE_ATTRIBUTE_8', 'ensemble_1_ENSEMBLE_ATTRIBUTE_9',
                                              # ensemble tags
                                              'ensemble_1_ENSEMBLE_TAG_data_type1', 'ensemble_1_ENSEMBLE_TAG_eoo1']

        # Only certain columns
        df = ensembles[0].to_dataframe(data_columns='4')
        assert df.columns.values.tolist() == ['id', 'name', 'creator', 'creation_date', 'last_modified_date',
                                              # dataset attributes
                                              '4',
                                              # ensemble attributes
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_id',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_name',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_creator',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_creation_date',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_last_modified_date',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_0', 'ensemble_0_ENSEMBLE_ATTRIBUTE_1',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_2', 'ensemble_0_ENSEMBLE_ATTRIBUTE_3',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_4', 'ensemble_0_ENSEMBLE_ATTRIBUTE_5',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_6', 'ensemble_0_ENSEMBLE_ATTRIBUTE_7',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_8', 'ensemble_0_ENSEMBLE_ATTRIBUTE_9',
                                              # ensemble tags
                                              'ensemble_0_ENSEMBLE_TAG_data_type0', 'ensemble_0_ENSEMBLE_TAG_eoo0']

        df = ensembles[0].to_dataframe(data_columns=['4'])
        assert df.columns.values.tolist() == ['id', 'name', 'creator', 'creation_date', 'last_modified_date',
                                              # dataset attributes
                                              '4',
                                              # ensemble attributes
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_id',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_name',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_creator',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_creation_date',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_last_modified_date',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_0', 'ensemble_0_ENSEMBLE_ATTRIBUTE_1',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_2', 'ensemble_0_ENSEMBLE_ATTRIBUTE_3',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_4', 'ensemble_0_ENSEMBLE_ATTRIBUTE_5',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_6', 'ensemble_0_ENSEMBLE_ATTRIBUTE_7',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_8', 'ensemble_0_ENSEMBLE_ATTRIBUTE_9',
                                              # ensemble tags
                                              'ensemble_0_ENSEMBLE_TAG_data_type0', 'ensemble_0_ENSEMBLE_TAG_eoo0']

        df = ensembles[0].to_dataframe(data_columns=['4', '9'])
        assert df.columns.values.tolist() == ['id', 'name', 'creator', 'creation_date', 'last_modified_date',
                                              # dataset attributes
                                              '4', '9',
                                              # ensemble attributes
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_id',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_name',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_creator',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_creation_date',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_last_modified_date',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_0', 'ensemble_0_ENSEMBLE_ATTRIBUTE_1',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_2', 'ensemble_0_ENSEMBLE_ATTRIBUTE_3',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_4', 'ensemble_0_ENSEMBLE_ATTRIBUTE_5',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_6', 'ensemble_0_ENSEMBLE_ATTRIBUTE_7',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_8', 'ensemble_0_ENSEMBLE_ATTRIBUTE_9',
                                              # ensemble tags
                                              'ensemble_0_ENSEMBLE_TAG_data_type0', 'ensemble_0_ENSEMBLE_TAG_eoo0']

        # Find data
        target_data = {'my_dataset_attribute': 0}
        df = ensembles[0].to_dataframe(data=target_data)
        for val in df['my_dataset_attribute'].values:
            self.assertEqual(val, 0)
        for i in range(n_ensembles):
            for val in df[f'ensemble_0_ENSEMBLE_ATTRIBUTE_{i}'].values:
                self.assertEqual(val, f"ensemble_0_attributes_{i}")
        assert df.shape == (5, 33)

        # Find data with missing columns
        target_data = {'my_dataset_attribute': 0}
        df = ensembles[0].to_dataframe(data=target_data, data_columns=['param1', 'param6'])
        assert df.shape == (5, 24)
        assert df.columns.values.tolist() == ['id', 'name', 'creator', 'creation_date', 'last_modified_date',
                                              # dataset attributes
                                              'param1', 'param6',
                                              # ensemble attributes
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_id',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_name',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_creator',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_creation_date',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_last_modified_date',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_0', 'ensemble_0_ENSEMBLE_ATTRIBUTE_1',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_2', 'ensemble_0_ENSEMBLE_ATTRIBUTE_3',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_4', 'ensemble_0_ENSEMBLE_ATTRIBUTE_5',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_6', 'ensemble_0_ENSEMBLE_ATTRIBUTE_7',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_8', 'ensemble_0_ENSEMBLE_ATTRIBUTE_9',
                                              # ensemble tags
                                              'ensemble_0_ENSEMBLE_TAG_data_type0', 'ensemble_0_ENSEMBLE_TAG_eoo0']

        # Find data with ensemble tags
        target_data = {'my_dataset_attribute': 0}
        ensemble_tags = {"eoo0": "even", "data_type0": "test data"}
        df = ensembles[0].to_dataframe(data=target_data,
                                       ensemble_tags=ensemble_tags)
        assert df.shape == (1, 33)

        # Find data with ensemble tags and with missing columns
        target_data = {'my_dataset_attribute': 0}
        ensemble_tags = {"eoo0": "even", "data_type0": "test data"}
        df = ensembles[0].to_dataframe(data=target_data,
                                       ensemble_tags=ensemble_tags,
                                       data_columns=['param1', 'param6'])
        assert df.shape == (1, 24)
        assert df.columns.values.tolist() == ['id', 'name', 'creator', 'creation_date', 'last_modified_date',
                                              # dataset attributes
                                              'param1', 'param6',
                                              # ensemble attributes
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_id',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_name',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_creator',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_creation_date',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_last_modified_date',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_0', 'ensemble_0_ENSEMBLE_ATTRIBUTE_1',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_2', 'ensemble_0_ENSEMBLE_ATTRIBUTE_3',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_4', 'ensemble_0_ENSEMBLE_ATTRIBUTE_5',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_6', 'ensemble_0_ENSEMBLE_ATTRIBUTE_7',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_8', 'ensemble_0_ENSEMBLE_ATTRIBUTE_9',
                                              # ensemble tags
                                              'ensemble_0_ENSEMBLE_TAG_data_type0', 'ensemble_0_ENSEMBLE_TAG_eoo0']

        # Don't include ensemble attributes
        target_data = {'my_dataset_attribute': 0}
        ensemble_tags = {"eoo0": "even", "data_type0": "test data"}
        df = ensembles[0].to_dataframe(data=target_data,
                                       ensemble_tags=ensemble_tags,
                                       include_ensemble_attributes=False)
        assert df.shape == (1, 18)
        assert df.columns.values.tolist() == ['id', 'name', 'creator', 'creation_date', 'last_modified_date',
                                              # dataset attributes
                                              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'my_dataset_attribute',
                                              # ensemble tags
                                              'ensemble_0_ENSEMBLE_TAG_data_type0', 'ensemble_0_ENSEMBLE_TAG_eoo0']

        # Don't include ensemble tags
        target_data = {'my_dataset_attribute': 0}
        ensemble_tags = {"eoo0": "even", "data_type0": "test data"}
        df = ensembles[0].to_dataframe(data=target_data,
                                       ensemble_tags=ensemble_tags,
                                       include_ensemble_tags=False)
        assert df.shape == (1, 31)
        assert df.columns.values.tolist() == ['id', 'name', 'creator', 'creation_date', 'last_modified_date',
                                              # dataset attributes
                                              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'my_dataset_attribute',
                                              # ensemble attributes
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_id',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_name',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_creator',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_creation_date',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_last_modified_date',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_0', 'ensemble_0_ENSEMBLE_ATTRIBUTE_1',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_2', 'ensemble_0_ENSEMBLE_ATTRIBUTE_3',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_4', 'ensemble_0_ENSEMBLE_ATTRIBUTE_5',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_6', 'ensemble_0_ENSEMBLE_ATTRIBUTE_7',
                                              'ensemble_0_ENSEMBLE_ATTRIBUTE_8', 'ensemble_0_ENSEMBLE_ATTRIBUTE_9']

        # Don't include ensemble attributes or ensemble tags
        target_data = {'my_dataset_attribute': 0}
        ensemble_tags = {"eoo0": "even", "data_type0": "test data"}
        df = ensembles[0].to_dataframe(data=target_data,
                                       ensemble_tags=ensemble_tags,
                                       include_ensemble_attributes=False,
                                       include_ensemble_tags=False)
        assert df.shape == (1, 16)
        assert df.columns.values.tolist() == ['id', 'name', 'creator', 'creation_date', 'last_modified_date',
                                              # dataset attributes
                                              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'my_dataset_attribute']

    def test_sort_by(self):

        store, kosh_db = self.connect()

        n_ensembles = 10
        n_datasets = 10

        datasets = []

        for i in range(n_datasets):
            metadata = {"ds_attribute": f"dataset_{i}"}
            ds = store.create(id=f"dataset_{i}", metadata=metadata)
            datasets.append(ds)

        for i in range(n_ensembles):
            metadata = {"ens_attribute": f"ensemble_{i}"}
            ens = store.create_ensemble(id=f"ensemble_{i}", metadata=metadata)
            ens.add(ds, inherit_attributes=False)

        # Default Ascending
        for i, ds in enumerate(list(store.find(sort_by="ds_attribute"))):
            self.assertEqual(f"dataset_{i}", ds.ds_attribute)

        for i_ens, ens in enumerate(list(store.find_ensembles(sort_by="ens_attribute"))):
            self.assertEqual(f"ensemble_{i_ens}", ens.ens_attribute)
            for i, ds in enumerate(list(ens.find(sort_by="ds_attribute"))):
                self.assertEqual(f"dataset_{i}", ds.ds_attribute)

        # Descending sort_by_descending=True
        for i, ds in enumerate(list(store.find(sort_by="ds_attribute", sort_by_descending=True))):
            self.assertEqual(f"dataset_{n_datasets - 1 - i}", ds.ds_attribute)

        for i_ens, ens in enumerate(list(store.find_ensembles(sort_by="ens_attribute", sort_by_descending=True))):
            self.assertEqual(f"ensemble_{n_ensembles - 1 - i_ens}", ens.ens_attribute)
            for i, ds in enumerate(list(ens.find(sort_by="ds_attribute", sort_by_descending=True))):
                self.assertEqual(f"dataset_{n_datasets - 1 - i}", ds.ds_attribute)

    def test_session_sort_by(self):

        store, kosh_db = self.connect()

        n_ensembles = 1
        n_datasets = 10

        datasets = []

        for i in range(n_datasets):
            metadata = {"ds_attribute": f"dataset_{i}"}
            ds = store.create(id=f"dataset_{i}", metadata=metadata)
            datasets.append(ds)

        for i in range(n_ensembles):
            metadata = {"ens_attribute": f"ensemble_{i}"}
            ens = store.create_ensemble(id=f"ensemble_{i}", metadata=metadata)
            ens.add(ds, inherit_attributes=False)

        # Default Ascending
        store, kosh_db = self.connect(session_sort_by="ds_attribute")
        for i, ds in enumerate(list(store.find())):
            self.assertEqual(f"dataset_{i}", ds.ds_attribute)

        for i_ens, ens in enumerate(list(store.find_ensembles())):
            self.assertEqual(f"ensemble_{i_ens}", ens.ens_attribute)
            for i, ds in enumerate(list(ens.find())):
                self.assertEqual(f"dataset_{i}", ds.ds_attribute)

        # Descending sort_by_descending=True
        store, kosh_db = self.connect(session_sort_by="ds_attribute", session_sort_by_descending=True)
        for i, ds in enumerate(list(store.find())):
            self.assertEqual(f"dataset_{n_datasets - 1 - i}", ds.ds_attribute)

        for i_ens, ens in enumerate(list(store.find_ensembles())):
            self.assertEqual(f"ensemble_{n_ensembles - 1 - i_ens}", ens.ens_attribute)
            for i, ds in enumerate(list(ens.find())):
                self.assertEqual(f"dataset_{n_datasets - 1 - i}", ds.ds_attribute)

        # Override session sort preferences to be in ascending order
        for i, ds in enumerate(list(store.find(sort_by_descending=False))):
            self.assertEqual(f"dataset_{i}", ds.ds_attribute)

        for i_ens, ens in enumerate(list(store.find_ensembles(sort_by_descending=False))):
            self.assertEqual(f"ensemble_{i_ens}", ens.ens_attribute)
            for i, ds in enumerate(list(ens.find(sort_by_descending=False))):
                self.assertEqual(f"dataset_{i}", ds.ds_attribute)

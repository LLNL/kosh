from __future__ import print_function
import koshbase
import os
import random
from datetime import datetime


def create_dataset(datastore, num):

    for i in range(num):
        datastore.create(i)
        dataset = list(datastore.search(name=i))[0]
        metadata = {"param1": random.random() * 2.,
                    "param2": random.random() * 1.5,
                    "param3": random.random() * 5,
                    "param4": random.random() * 3,
                    "param5": random.random() * 2.5,
                    "param6": chr(random.randint(65, 91)),
                    }
        dataset.update(metadata)

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
        assert df.columns.values.tolist() == ['id', 'name', 'creator',
                                              'param1', 'param2', 'param3', 'param4', 'param5', 'param6']

        # Only certain columns
        df = store.to_dataframe(data_columns='param1')
        assert df.columns.values.tolist() == ['id', 'name', 'creator',
                                              'param1']

        df = store.to_dataframe(data_columns=['param1'])
        assert df.columns.values.tolist() == ['id', 'name', 'creator',
                                              'param1']

        df = store.to_dataframe(data_columns=['param1', 'param6'])
        assert df.columns.values.tolist() == ['id', 'name', 'creator',
                                              'param1', 'param6']

        # Everything with unique data
        store.create('new_dataset', metadata={'mynewattribute': 5})
        store.create('new_dataset2', metadata={'myotherattribute': 10})

        df = store.to_dataframe()
        assert df.columns.values.tolist() == ['id', 'name', 'creator',
                                              'mynewattribute', 'myotherattribute',
                                              'param1', 'param2', 'param3', 'param4', 'param5', 'param6']

        # Find data
        target_data = {'mynewattribute': 5}
        df = store.to_dataframe(data=target_data)
        assert df.shape == (1, 4)

        # Find data with missing columns
        target_data = {'mynewattribute': 5}
        df = store.to_dataframe(data=target_data, data_columns=['param1', 'param6'])
        assert df.shape == (1, 5)

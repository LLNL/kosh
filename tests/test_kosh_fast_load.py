from __future__ import print_function
import koshbase
import os
import random
from datetime import datetime


def create_dataset(datastore, num):

    metadata = {"param1": random.random() * 2.,
                "param2": random.random() * 1.5,
                "param3": random.random() * 5,
                "param4": random.random() * 3,
                "param5": random.random() * 2.5,
                "param6": chr(random.randint(65, 91)),
                }

    for i in range(num):
        datastore.create(i)
        dataset = list(datastore.search(name=i))[0]
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

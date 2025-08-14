from __future__ import print_function
import koshbase
import os


class TestProtectedAttributes(koshbase.KoshTest):

    def test_protected_attributes(self):

        store, kosh_db = self.connect()

        # Dataset
        dataset = store.create(id='test1')

        creator = dataset.creator
        creation_date = dataset.creation_date
        last_modified_date = dataset.last_modified_date
        # print(dataset) has different last_modified at top and in list attributes

        dataset.creator = 'test'
        dataset.creation_date = 'test'

        self.assertEqual(dataset.creator, creator)
        self.assertEqual(dataset.creation_date, creation_date)
        self.assertEqual(dataset.last_modified_date, last_modified_date)

        dataset = list(store.find(id_pool='test1'))[0]

        dataset.creator = 'test'
        dataset.creation_date = 'test'

        self.assertEqual(dataset.creator, creator)
        self.assertEqual(dataset.creation_date, creation_date)

        # Ensemble
        ensemble = store.create_ensemble(id='test2')

        creator = ensemble.creator
        creation_date = ensemble.creation_date

        ensemble.creator = 'test'
        ensemble.creation_date = 'test'

        self.assertEqual(ensemble.creator, creator)
        self.assertEqual(ensemble.creation_date, creation_date)

        ensemble = list(store.find_ensembles(id_pool='test2'))[0]

        ensemble.creator = 'test'
        ensemble.creation_date = 'test'

        self.assertEqual(ensemble.creator, creator)
        self.assertEqual(ensemble.creation_date, creation_date)

        store.close()
        os.remove(kosh_db)

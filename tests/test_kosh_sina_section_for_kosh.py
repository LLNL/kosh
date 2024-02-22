from __future__ import print_function
import koshbase
import os


class TestKoshSection(koshbase.KoshTest):

    def test_kosh_section(self):

        store, kosh_db = self.connect()

        store.create(id='test1', metadata={'param1': 1, 'param2': 2, 'param3': 3, 'param4': 4.5})
        store.create(id='test2', metadata={'PARAM1': 1, 'PARAM2': 2, 'PARAM3': 3.5, 'PARAM4': 4})
        store.import_dataset("tests/baselines/sina/sina_curve_rec_mimes_and_curves.json",
                             match_attributes=["name", "id"])
        store.import_dataset("tests/baselines/sina/sina_curve_rec_2.json",
                             match_attributes=["name", "id"])
        store.import_dataset("tests/baselines/sina/sina_curve_rec_mimes_and_curves_3.json",
                             match_attributes=["name", "id"])

        for dataset in store.find():
            record = store.__record_handler__.get(dataset.id)
            self.assertIn('kosh_information', record['user_defined'])

        store.close()
        os.remove(kosh_db)

from __future__ import print_function
from koshbase import KoshTest
import kosh


class KoshTestDataset(KoshTest):
    def test_load_pure_sina(self):
        store = kosh.KoshStore(
            db_uri="tests/baselines/sina/data.sqlite",
            dataset_record_type="obs")
        all_recs = store.search(ids_only=True)
        self.assertEqual(len(all_recs), 10)
        A = store.open("2014-04-05-06-21-43")
        self.assertEqual(A.PARAM1, 243.3184)
        self.assertEqual(A.PARAM2, 149.5)
        self.assertEqual(A.PARAM3, 21242.52)
        self.assertEqual(A.PARAM4, 997057.0)
        self.assertEqual(A.date, "4/5/2014")
        self.assertEqual(A.latitude, 53.760735)
        self.assertEqual(A.longitude, 10.95989)
        self.assertEqual(A.time, "6:21:43 AM")
        # print(A)

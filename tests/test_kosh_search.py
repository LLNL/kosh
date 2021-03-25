from __future__ import print_function
import koshbase
import time
import os
import numpy


class TestKoshSearch(koshbase.KoshTest):
    def testSearchManyTypes(self):
        store, kosh_db = self.connect(sync=False)
        standard = store.create(datasetId="my_id")
        store._dataset_record_type = 'some_new_type'
        store.create(datasetId="some_id")
        store.create(datasetId="some_other_id", sina_type="some_other_new_type")
        datasets = sorted(list(store.search()), key=lambda x: x.__id__)
        self.assertEqual(len(datasets), 3)
        types = ["blah", "some_new_type", "some_other_new_type"]
        for i, ds in enumerate(datasets):
            rec = store.get_record(ds.__id__)
            self.assertEqual(rec["type"], types[i])

if __name__ == "__main__":
    A = TestKoshSearch()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

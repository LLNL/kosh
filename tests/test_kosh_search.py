from __future__ import print_function
import koshbase
import os


class TestKoshSearch(koshbase.KoshTest):
    def testSearchManyTypes(self):
        store, kosh_db = self.connect(sync=False)
        store.create(id="my_id")
        store._dataset_record_type = 'some_new_type'
        store.create(id="some_id")
        store.create(id="some_other_id", sina_type="some_other_new_type")
        datasets = sorted(list(store.search()), key=lambda x: x.id)
        self.assertEqual(len(datasets), 3)
        types = ["blah", "some_new_type", "some_other_new_type"]
        for i, ds in enumerate(datasets):
            rec = store.get_record(ds.id)
            self.assertEqual(rec["type"], types[i])
        os.remove(kosh_db)


if __name__ == "__main__":
    A = TestKoshSearch()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

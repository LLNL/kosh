import os

from koshbase import KoshTest


class KoshTestList(KoshTest):
    def test_cache_list_features(self):
        store, uri = self.connect()
        baseline_dir = os.path.join(os.path.dirname(__file__), "baselines", "sina", "firstdir", "seconddir", "thirddir")

        baseline_jsons = [
            os.path.join(baseline_dir, "sina_rec_1_sina.json"),
            os.path.join(baseline_dir, "fourthdir", "sina_rec_2_sina.json"),
        ]

        for filename in baseline_jsons:
            print("file:", filename)
            store.import_dataset(filename)
            store.delete_all_contents(force="SKIP PROMPT")
        store.close()
        os.remove(uri)


if __name__ == "__main__":
    A = KoshTestList()
    for nm in dir(A):
        if nm.startswith("test_"):
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

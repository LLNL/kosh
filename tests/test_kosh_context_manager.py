from __future__ import print_function
from koshbase import KoshTest
import kosh
import time
import os


def kosh_task(store_name):
    with kosh.KoshStore(store_name, read_only=True) as store:
        dataset = list(store.find(name='Dataset1'))[0]
    return dataset.attr1 + ' ' + dataset.attr2

# Define a dummy worker function for some other task


def dummy_task():
    time.sleep(1)
    return False


class KoshTestDataset(KoshTest):
    def test_context_manager(self):
        _, uri = self.connect()
        with kosh.connect(uri, delete_all_contents=True) as store:
            store.create()
        os.remove(store.db_uri)

    def test_ThreadPool(self):
        try:
            from concurrent.futures import ThreadPoolExecutor
            hasThreadPool = True
        except ImportError:
            hasThreadPool = False
        if hasThreadPool:
            store, store_uri = self.connect()
            store.create(
                name="Dataset1", metadata={
                    "attr1": "1", "attr2": "2"})
            with ThreadPoolExecutor() as pool:
                dummy_thread = pool.submit(dummy_task)
                kosh_thread = pool.submit(kosh_task, store_uri)
                print("Dummy", dummy_thread.result())
                print("kosh", kosh_thread.result())
            store.close()
            os.remove(store_uri)

    def test_edit_dataset_after_close(self):
        _, uri = self.connect()
        with kosh.connect(uri, delete_all_contents=True) as store:
            ds = store.create()

        ds.after_close = "closed"
        os.remove(store.db_uri)

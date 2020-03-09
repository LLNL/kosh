import koshbase
import time

class TestKoshSearchSpeed(koshbase.KoshTest):
    def testSpeedSearch(self):
        store, kosh_db = self.connect(sync=False)
        meta = {}
        for i in range(65, 122):
            meta[chr(i)] = str(i)
            meta[f"A_{chr(i)}"] = str(i)
            meta[f"B_{chr(i)}"] = str(i)
            meta[f"C_{chr(i)}"] = str(i)
            meta[f"D_{chr(i)}"] = str(i)

        search_times = []
        for i in range(200):
            start = time.time()
            search = store.search(**meta)
            search_times.append(time.time() - start)
            print(f"Iter: {i} took {search_times[-1]}")
            ds = store.create(metadata=meta)
            ds.associate("/some_path", mime_type="some type")

import koshbase
import time
import numpy

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
        for i in range(50):
            start = time.time()
            search = store.search(ids_only=True,**meta)
            search_times.append(time.time() - start)
            ds = store.create(metadata=meta)
            ds.associate("/some_path", mime_type="some type")
        store.sync()
        # Skip first 5s to ensure disk/startup issues are removed
        a, b = numpy.polyfit(numpy.arange(len(search_times)-5), numpy.array(search_times[5:]), 1)
        print("A, B:", a, b)
        self.assertLessEqual(b, .11)


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
        for i in range(90):
            start = time.time()
            search = store.search(ds_only=True,**meta)
            search_times.append(time.time() - start)
            ds = store.create(metadata=meta)
            ds.associate("/some_path", mime_type="some type")
        store.sync()
        a, b = numpy.polyfit(numpy.arange(len(search_times)), numpy.array(search_times), 1)
        print("A, B:", a, b)


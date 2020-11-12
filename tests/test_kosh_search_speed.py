from __future__ import print_function
import koshbase
import time
import numpy


class TestKoshSearchSpeed(koshbase.KoshTest):
    def testSpeedSearch(self):
        store, kosh_db = self.connect(sync=False)
        meta = {}
        for i in range(65, 122):
            meta[chr(i)] = str(i)
            meta["A_{}".format(chr(i))] = str(i)
            meta["B_{}".format(chr(i))] = str(i)
            meta["C_{}".format(chr(i))] = str(i)
            meta["D_{}".format(chr(i))] = str(i)

        search_times = []
        for i in range(50):
            start = time.time()
            store.search(ids_only=True, **meta)
            search_times.append(time.time() - start)
            ds = store.create(metadata=meta)
            ds.associate("/some_path", mime_type="some type")
        store.sync()
        # Skip first 5s to ensure disk/startup issues are removed
        a, b = numpy.polyfit(numpy.arange(
            len(search_times) - 5), numpy.array(search_times[5:]), 1)
        print("A, B:", a, b)
        self.assertLessEqual(b, .18)

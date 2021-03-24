from __future__ import print_function
import koshbase
import time
import os
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
        for i in range(70):
            start = time.time()
            ds = store.create(metadata=meta)
            ds.associate("/some_path", mime_type="some type")
            ids = store.search(ids_only=True, **meta)
            search_times.append(time.time() - start)
            print(i, len(ids), search_times[-1])
            ds.dissociate("/some_path")
            store.delete(ds)
        store.sync()
        # Skip first 10s to ensure disk/startup issues are removed
        a, b = numpy.polyfit(numpy.arange(
            len(search_times) - 10), numpy.array(search_times[10:]), 1)
        print("A, B:", a, b)
        # Make sure it's pretty much constant
        self.assertLessEqual(a, .001)
        os.remove(kosh_db)


if __name__ == "__main__":
    A = TestKoshSearchSpeed()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

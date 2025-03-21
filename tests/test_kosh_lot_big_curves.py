import kosh
import time
from koshbase import KoshTest
import os
import random


class KoshTestLotCurves(KoshTest):
    def create_file(self, name, var, len_curve):
        if not os.path.exists(name):
            print("Creating dummy ultra file", name)
            with open(name, "w") as f:
                print("# Dummy ultra file", file=f)
                print(f"# {var}", file=f)
                for i in range(len_curve):
                    print(f"{i/100.} {random.randint(0,10)}", file=f)

    def test_read_lots_of_curves(self):
        store, uri = self.connect()
        ds = store.create()
        num_files = 1000
        len_curves = 10000
        t0 = time.time()
        names = []
        print("Creating")
        for i in range(num_files):
            name = f'ultra_{i}.ultra'
            self.create_file(name, f"variable_{i}", len_curves)
            names.append(name)
        t = time.time()
        dt = t - t0
        print("Creation time:", dt)
        t0 = time.time()
        print("Associating")
        ds.associate(names, "ultra")
        t = time.time()
        dt = t - t0
        print("Association time:", dt)
        t0 = time.time()
        ds.list_features()
        t = time.time()
        dt0 = t - t0
        print(f"First list time: {dt0:.2f}s")
        store.close()
        store = kosh.connect(uri)
        ds = next(store.find())
        t0 = time.time()
        ds.list_features()
        t = time.time()
        dt = t - t0
        print(f"Second list time: {dt:.2f}s speedup: {dt0/dt*100:.2f}%")
        self.assertGreater(dt0/dt, 5)
        store.close()
        os.remove(uri)
        for name in names:
            os.remove(name)

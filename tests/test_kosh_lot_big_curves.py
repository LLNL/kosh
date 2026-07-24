import kosh
import time
from koshbase import KoshTest
import os
import random


class KoshTestLotCurves(KoshTest):
    def create_file(self, name, var, len_curve):
        if os.path.exists(name):
            return
        print("Creating dummy ultra file", name)
        rng = random.Random(0)
        with open(name, "w") as f:
            f.write("# Dummy ultra file\n")
            f.write(f"# {var}\n")
            for i in range(len_curve):
                f.write(f"{i/100.} {rng.randint(0, 10)}\n")

    def test_read_lots_of_curves(self):
        store, uri = self.connect()
        ds = store.create()
        # This is effectively a performance test. Keep the default small enough
        # for CI, but allow stressing via env var.
        stress = os.environ.get("KOSH_STRESS_TESTS") == "1"
        num_files = 1000 if stress else 50
        len_curves = 10000 if stress else 1000
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
        # This is a caching/performance regression check; allow a smaller ratio
        # for the non-stress (fast) configuration.
        speedup = dt0 / max(dt, 1e-9)
        self.assertGreater(speedup, 5 if stress else 1.2)
        store.close()
        os.remove(uri)
        for name in names:
            os.remove(name)

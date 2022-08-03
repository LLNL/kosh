from __future__ import print_function
import os
from koshbase import KoshTest
from kosh.utils import compute_fast_sha


class KoshTestFixFastSha(KoshTest):
    def test_clean_fastsha(self):
        store, db = self.connect()
        d = store.create()
        asso = d.associate("setup.py", "py")
        asso = store._load(asso)
        # change the fast_sha
        asso.fast_sha = "blah"

        # Let's also associate an non existing uri
        asso_not_here = store._load(d.associate("I_don_not_exit.txt", "txt"))

        # Now let's check store integrity
        bad = store.check_integrity()
        self.assertTrue(asso.uri in bad)
        self.assertTrue(asso_not_here.uri in bad)

        # Let's fix it
        d.cleanup_files(clean_fastsha=True)

        # Ensure it's fixed
        self.assertTrue(len(store.check_integrity()) == 0)
        self.assertTrue(len(d.check_integrity()) == 0)

        # is sha really corrected?
        self.assertEqual(asso.fast_sha, compute_fast_sha("setup.py"))
        store.close()
        os.remove(db)


if __name__ == "__main__":
    A = KoshTestFixFastSha()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

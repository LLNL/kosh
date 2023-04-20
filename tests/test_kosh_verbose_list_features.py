import os
from koshbase import KoshTest
import kosh
import tempfile
try:
    from contextlib import redirect_stdout
    from io import StringIO
except ImportError:
    import contextlib
    import sys
    from io import BytesIO as StringIO

    @contextlib.contextmanager
    def redirect_stdout(target):
        original = sys.stdout
        try:
            sys.stdout = target
            yield
        finally:
            sys.stdout = original


class LoaderMissingPackage(kosh.KoshLoader):
    types = {"missing_module": ["list", ]}

    def list_features(self, *args, **kargs):
        import missing_module  # noqa
        return ["feature", ]

    def extract(self, feature, format):
        import missing_module  # noqa
        return [1, 2, 3]


class FakeTextLoader(kosh.KoshLoader):
    types = {"fake_txt": ["list"]}

    def list_features(self, *args, **kargs):
        with open(self.obj.uri):
            pass
        return ["fake", ]

    def extract(self, feature, format):
        with open(self.obj.uri):
            pass
        return [1, 2, 3]


class KoshTestListFeatureVerbose(KoshTest):
    def testMissingModule(self):
        store, uri = self.connect()
        store.add_loader(LoaderMissingPackage)
        ds = store.create()
        ds.associate("setup.py", "missing_module")
        f = StringIO()
        with redirect_stdout(f):
            ds.list_features(verbose=True)
        out = f.getvalue()
        print("OUT:", out)
        self.assertTrue("No module named " in out and "missing_module" in out)
        store.close()
        os.remove(uri)

    def testBadPermissions(self):
        store, uri = self.connect()
        store.add_loader(FakeTextLoader)
        ds = store.create()
        perm_file = tempfile.NamedTemporaryFile()
        ds.associate(perm_file.name, "fake_txt")
        os.chmod(perm_file.name, 0o000)
        f = StringIO()
        with redirect_stdout(f):
            ds.list_features(verbose=True)
        out = f.getvalue()
        self.assertTrue("Permission denied" in out)
        store.close()
        os.remove(uri)

    def testFileNotFound(self):
        store, uri = self.connect()
        store.add_loader(FakeTextLoader)
        ds = store.create()
        ds.associate("not_here.txt", "fake_txt")
        f = StringIO()
        with redirect_stdout(f):
            ds.list_features(verbose=True)
        out = f.getvalue()
        self.assertTrue("No such file or directory" in out)
        store.close()
        os.remove(uri)

    def testNoMime(self):
        store, uri = self.connect()
        store.add_loader(FakeTextLoader)
        ds = store.create()
        ds.associate("setup.py", "nomime")
        print(ds.list_features())
        store.close()
        os.remove(uri)


if __name__ == "__main__":
    A = KoshTestListFeatureVerbose()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

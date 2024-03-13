import koshbase
import kosh
import numpy
import os


class FakeLoader(kosh.loaders.KoshLoader):
    types = {"fake": ["numpy", ], }

    def list_features(self):
        return ["data", ]

    def extract(self):
        out = numpy.array([[1, 2, 3, 2, 1], [2, 3, 4, 3, 2]]).transpose()
        return out

    def describe_feature(self, feature):
        return {"size": (5, 2)}


def MathVStackOperator(*args, **kargs):
    """
    Perform a hstack across feature inputs
    """
    return numpy.vstack([arg[:] for arg in args])


class VStackOperator(kosh.KoshOperator):
    types = {"numpy": ["numpy", ]}

    def __init__(self, *args, **kargs):
        super(VStackOperator, self).__init__(*args, **kargs)
        self.kargs = kargs

    def operate(self, *inputs, **kargs):
        """
        Perform a vstack across feature inputs
        """
        local_kargs = {**self.kargs, **kargs}
        return MathVStackOperator(
            *[_input[:] for _input in inputs], **local_kargs)


class TestKoshOperators(koshbase.KoshTest):
    def testOpeChainedPrepro(self):
        store, db_uri = self.connect()

        store.add_loader(FakeLoader)
        ds = store.create(name="testit")
        ds.associate("fake.fake", "fake")
        ds = store.create(name="testit2")
        ds.associate("fake.fake", "fake")

        payload = VStackOperator(*[ds["data"] for ds in store.find()])

        #
        # Set up kosh cluster operator
        #
        clust_op = kosh.operators.KoshCluster(
            payload,
            method="DBSCAN",
            eps=1,
            output='indices',
            batch=False,
            verbose=False,
        )

        print(clust_op[:])
        store.close()
        os.remove(db_uri)


if __name__ == "__main__":
    A = TestKoshOperators()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

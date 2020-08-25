# inspired from: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html#sklearn.datasets.make_blobs
import kosh
from sklearn import cluster, datasets, mixture


class SKDatasetLoader(kosh.loaders.KoshLoader):
    types = {"sk_dataset" : ["numpy",]}

    def extract(self):
        args, kargs = self._user_passed_parameters
        if self.feature == "moon":
            return datasets.make_moons(random_state=8, **kargs)[0]
        elif self.feature == "circle":
            return datasets.make_circles(random_state=8, **kargs)[0]
        elif self.feature == "blob":
            return datasets.make_blobs(random_state=8, **kargs)[0]
        else:
            raise RuntimeError("not a feature")
    def list_features(self):
        return ["moon", "circle", "blob"]

store = kosh.utils.create_new_db("crp.sql")

store.add_loader(SKDatasetLoader)

ds = store.create()
ds.associate("/blah", mime_type="sk_dataset")
print(ds._associated_data_)
# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 1500
features = [("circle", {'factor':.5, 'noise':.05}),
        ("moon",{'noise':0.05}),
        ("blob", {})]

SC = kosh.transformers.StandardScaler()

estimators = [(kosh.transformers.KMeans, {"n_clusters":3}), (kosh.transformers.DBSCAN, {'eps':.3})]
colors = ["orange", "blue", "red" ,"green", "purple", "salmon", "pink", "grey", "brown", "beige"]
import matplotlib.pyplot as plt
f, axarr = plt.subplots(len(estimators), len(features))
for i, (feature, args) in enumerate(features):
    raw = ds.get(feature, n_samples=n_samples, transformers=[SC,], **args)
    #print(raw.shape)
    for j, (est, kargs) in enumerate(estimators):
        axarr[j,i].scatter(raw[:,0], raw[:, 1], color="black", s=5)
        E = est(**kargs)
        estimator = ds.get(feature, n_samples=n_samples,transformers=[SC, E,], format="estimator", **args)
        #print(estimator)
        E = est(n_samples=100., sampling_method="percent", **kargs)
        labels, data = ds.get(feature, n_samples=n_samples, transformers=[SC, E,], format="numpy", **args)
        #print(len(labels))
        for k, dat in enumerate(data):
            print(dat.shape, k)
            axarr[j,i].scatter(dat[:,0], dat[:,1], color=colors[k], s=5)

plt.show()

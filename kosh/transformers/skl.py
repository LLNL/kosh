# Kosh transformer for scikit learn models
from .core import KoshTransformer
import warnings
import numpy
try:
    import sklearn.cluster
    import sklearn.preprocessing
    import sklearn.model_selection
except ImportError:
    warnings.warn(
        "Could not import sklearn, Scikit Learn-based transformers will not be available")


class Splitter(KoshTransformer):
    types = {"numpy": ["numpy", ]}

    def __init__(self, train_size=None,
                 test_size=None,
                 validation_size=None,
                 splitter=sklearn.model_selection.ShuffleSplit,
                 random_state=None,
                 n_splits=1,
                 *args, **kargs):
        if train_size is None and test_size is None and validation_size is None:
            train_size = .9
            test_size = .1
            validation_size = 0.
        elif train_size is None and validation_size is None:
            train_size = 1. - test_size
            validation_size = 0.
        elif train_size is None and test_size is None:
            test_size = validation_size
            train_size = 1. - test_size - validation_size
        elif test_size is None and validation_size is None:
            validation_size = 0.
            test_size = 1. - train_size
        elif validation_size is None:
            validation_size = 1. - train_size - test_size
        elif test_size is None:
            test_size = 1. - train_size - validation_size
        elif train_size is None:
            train_size = 1. - test_size - validation_size

        self.groups = kargs.pop("groups", None)
        if not (0. <= train_size <= 1.):
            raise ValueError("train size must be between 0 and 1")
        if not (0. <= test_size <= 1.):
            raise ValueError("test size must be between 0 and 1")
        if not (0. <= validation_size <= 1.):
            raise ValueError("validation size must be between 0 and 1")
        if train_size + test_size + validation_size > 1.:
            raise ValueError("You ask for a {}/{}/{} split which is more than 100%".format(
                train_size, test_size, validation_size))
        self.splitter = splitter(train_size=train_size,
                                 test_size=test_size+validation_size,
                                 random_state=random_state,
                                 n_splits=n_splits,
                                 *args, **kargs)
        self.validation_size = validation_size
        super(Splitter, self).__init__(train_size=train_size,
                                       test_size=test_size,
                                       validation_size=validation_size,
                                       splitter=splitter,
                                       random_state=random_state,
                                       n_splits=1,
                                       groups=self.groups, *args, **kargs)
        kargs.pop("test_size", None)
        kargs.pop("train_size", None)
        kargs["n_splits"] = 1
        kargs["random_state"] = random_state
        if validation_size != 0:
            self.validation_splitter = splitter(
                test_size=validation_size/(validation_size+test_size), *args, **kargs)

    def transform(self, input, format):
        out = list(self.splitter.split(input, groups=self.groups))
        if self.validation_size > 0:
            for i, [_, tmp_test] in enumerate(out):
                out[i] = list(out[i])  # Need to convert to list...
                tmp_test, tmp_validation = list(self.validation_splitter.split(tmp_test))[0]
                out[i][1] = tmp_test
            out[i].append(tmp_validation)
        return out


class StandardScaler(KoshTransformer):
    types = {"numpy": ["numpy", ]}

    def __init__(self, *args, **kargs):
        self.scaler = sklearn.preprocessing.StandardScaler(*args, **kargs)
        super(StandardScaler, self).__init__(*args, **kargs)

    def transform(self, input, format):
        return self.scaler.fit_transform(input)


class SKL(KoshTransformer):
    types = {"numpy": ["estimator", "numpy"]}

    def __init__(self, *args, **kargs):
        kw = {}
        for arg in ["n_samples", "sampling_method"]:
            setattr(self, arg, kargs.pop(arg, None))
            kw[arg] = getattr(self, arg)
        skl_class = kargs.pop("skl_class")
        self.skl_class = skl_class(*args, **kargs)
        kw.update(kargs)
        super(SKL, self).__init__(*args, **kargs)

    def transform(self, input, format):
        estimator = self.skl_class.fit(input)
        if format is not None and "estimator" in format.lower():
            return estimator
        labels = estimator.labels_
        out = []
        for each in sorted(set(labels)):
            class_member_mask = (labels == each)
            if self.n_samples is not None:
                if self.sampling_method in [None, "unit"]:
                    out.append(input[class_member_mask][:self.n_samples])
                elif self.sampling_method == "percent":
                    out.append(input[class_member_mask][:int(
                        self.n_samples / 100. * input.shape[0])])
                elif self.sampling_method == "random":
                    indices = numpy.random.rand_integers(
                        0, input.shape - 1, self.n_samples)
                    out.append(input[class_member_mask][indices])
                elif self.sampling_method == "percent_random":
                    indices = numpy.random.rand_integers(
                        0, input.shape - 1, int(self.n_samples / 100. * input.shape[0]))
                    out.append(input[class_member_mask][indices])
            else:
                out.append(input[class_member_mask])
        return sorted(set(labels)), out


class DBSCAN(SKL):
    """A DBSCAN Estimator from sklearn
    This transformer either return an estimator or a set of labels/numpy arrays for each class found by the estimator
    If you chose to eturn the arrays (format=numpy)
    When initiating the transformer you can pass any argument necessary for the DBSCAN initialzation

    """
    types = {"numpy": ["estimator", "numpy"]}

    def __init__(self, *args, **kargs):
        kargs["skl_class"] = sklearn.cluster.DBSCAN
        super(DBSCAN, self).__init__(**kargs)


class KMeans(SKL):
    types = {"numpy": ["estimator", "numpy"]}

    def __init__(self, *args, **kargs):
        kargs["skl_class"] = sklearn.cluster.KMeans
        super(KMeans, self).__init__(**kargs)

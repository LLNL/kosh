# Kosh transformer for scikit learn models
from .core import KoshTransformer
import numpy
try:
    import sklearn.cluster
    import sklearn.preprocessing
    import sklearn.model_selection
    has_skl = True
except ImportError:
    has_skl = False


class Splitter(KoshTransformer):
    """SKL-based class to split dataset into test, train and validation"""
    types = {"numpy": ["numpy", ]}

    def __init__(self, train_size=None,
                 test_size=None,
                 validation_size=None,
                 splitter=sklearn.model_selection.ShuffleSplit,
                 random_state=None,
                 n_splits=1,
                 *args, **kargs):
        """Initialize Splitt Transformer

        At least 2 of train/test/validation size are required, must add to 100%

        :param train_size: size of the dataset to reserve for training (.9 = 90% default)
        :type train_size: float
        :param test_size: size of the dataset to reserve for testing (.1 = 10% default)
        :type test_size: float
        :param validation_size: size of the dataset to reserve for validating (.0 = 0% default)
        :type validation_size: float
        :param splitter: SKL splitter to use to split
                         Same one will be used to first select training set
                         and then split again the rest between test and validation
                         default: sklearn.model_selection.ShuffleSplit
        :type splitter: sklearn.model_selection Splitter class
        :param random_state: random state for reproducibility
                             Controls the randomness of the training and
                             testing indices produced.
                             Pass an int for reproducible output across
                             multiple function calls.
        :type random_state: int
        :param n_splits: total number of split iteration to generate (default 1)
        :type n_splits: int
        :param groups: split data according to a third-party provided group.
                       This group information can be used to encode
                       arbitrary domain specific stratifications of the samples
                       as integers.
                       For instance the groups could be the year of collection
                       of the samples and thus allow for cross-validation against
                       time-based splits.
        :type groups: list or None
        :return: initialzed Splitter transformer
        :rtype: Splitter
        """

        if not has_skl:
            raise RuntimeError(
                "Could not import sklearn, Scikit Learn-based transformers are not available")

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
        """Split input data between n_splits sets of training/test/validation
        :param input: array from previous loader or transformer
        :type input: ndarray
        :param format: output format
        :type format: str
        :return: n_splits sets of train/test/validation
        :rtype: n_split list of train, test, validation ndarrays
        """
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
        """SKL-based scaler transformer"""
        if not has_skl:
            raise RuntimeError(
                "Could not import sklearn, Scikit Learn-based transformers are not available")

        self.scaler = sklearn.preprocessing.StandardScaler(*args, **kargs)
        super(StandardScaler, self).__init__(*args, **kargs)

    def transform(self, input, format):
        """calls the `fit_transform` function of the scaler on the input data
        :param input: array from previous loader or transformer
        :type input: ndarray
        :param format: output format
        :type format: str
        :return: scaled input data
        :rtype: ndarray
        """

        return self.scaler.fit_transform(input)


class SKL(KoshTransformer):
    """base class for SKL-based Kosh classifier
    This transformer returns either:
    * an estimator or
    * a set of labels/numpy arrays for each class found by the estimator
      If you chose to return the arrays (format=numpy) you can control
      how much data is return for each class/label
    When initiating the transformer you can pass any argument necessary for the SKL classifier initialization
    """
    types = {"numpy": ["estimator", "numpy"]}

    def __init__(self, *args, **kargs):
        """initialize Kosh classifier
        :param n_samples: number/percent of samples to
                          send back for each class
        :type n_samples: float
        :param sampling_method: units of percent random, or random_percent
                                units to returnthe number of samples
                                unit: return the first "n_samples"
                                percent: return the first n_samples % of the class
                                random_unit: return 'n_samples' random point from each class
                                random_percent: return n_samples % of the class randomly
        :type sampling_method: float
        :param random_state: random state for reproducibility
        :type random_state: int
        :param skl_class: SKL classifier
        :type skl_class: sklearn classifier
        :return: estimator from classifier.fit(input) function or
                 labels, list of ndarray with samples in each class
                        possibly sub-sampled via n_sample/sampling_method
        """
        if not has_skl:
            raise RuntimeError(
                "Could not import sklearn, Scikit Learn-based transformers are not available")

        kw = {}
        for arg in ["n_samples", "sampling_method", "random_state"]:
            setattr(self, arg, kargs.pop(arg, None))
            kw[arg] = getattr(self, arg)
        skl_class = kargs.pop("skl_class")
        self.skl_class = skl_class(*args, **kargs)
        kw.update(kargs)
        super(SKL, self).__init__(*args, **kargs)

    def transform(self, input, format):
        """If format is `numpy` scales the input data
        If format is `estimator` returns the estimator
        Possibly pads the ends with a value
        :param input: array from previous loader or transformer
        :type input: ndarray
        :param format: output format
        :type format: str
        :return: input taken over transformer's axis and indices
        """
        estimator = self.skl_class.fit(input)
        if format is not None and "estimator" in format.lower():
            return estimator
        labels = estimator.labels_
        out = []
        sorted_labels = sorted(set(labels))
        numpy.random.seed = self.random_state
        for each in sorted_labels:
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
        return sorted_labels, out


class DBSCAN(SKL):
    """SKL-based DBSCAN Kosh classifier
    This transformer returns either:
    * an estimator or
    * a set of labels/numpy arrays for each class found by the estimator
      If you chose to return the arrays (format=numpy) you can control
      how much data is return for each class/label
    When initiating the transformer you can pass any argument necessary for the SKL classifier initialization
    """
    types = {"numpy": ["estimator", "numpy"]}

    def __init__(self, *args, **kargs):
        if not has_skl:
            raise RuntimeError(
                "Could not import sklearn, Scikit Learn-based transformers are not available")

        kargs["skl_class"] = sklearn.cluster.DBSCAN
        super(DBSCAN, self).__init__(**kargs)


class KMeans(SKL):
    """SKL-based KMeans Kosh classifier
    This transformer returns either:
    * an estimator or
    * a set of labels/numpy arrays for each class found by the estimator
      If you chose to return the arrays (format=numpy) you can control
      how much data is return for each class/label
    When initiating the transformer you can pass any argument necessary for the SKL classifier initialization
    """
    types = {"numpy": ["estimator", "numpy"]}

    def __init__(self, *args, **kargs):
        if not has_skl:
            raise RuntimeError(
                "Could not import sklearn, Scikit Learn-based transformers are not available")

        kargs["skl_class"] = sklearn.cluster.KMeans
        super(KMeans, self).__init__(**kargs)

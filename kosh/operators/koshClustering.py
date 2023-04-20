import numpy as np
import pandas as pd
from ..sampling_methods.cluster_sampling import Cluster
from ..sampling_methods.cluster_sampling import makeBatchClusterParallel
from .core import KoshOperator


class KoshCluster(KoshOperator):
    types = {"numpy": ["numpy", "pandas"]}

    def __init__(self, *args, **options):
        """Clusters together similar samples from a dataset, and then returns cluster representatives
        to form a non-redundant subsample of the original dataset. The datasets need to be of shape
        (n_samples, n_features). All datasets must have the same number of features. If the datasets
        are more than two dimensions there is an option to flatten them.
        """
        super(KoshCluster, self).__init__(*args, **options)
        self.options = options

        # In case they don't have mpi4py
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        except ImportError:
            class Comm():
                def Get_size(self):
                    return 1

                def Get_rank(self):
                    return 0
            comm = Comm()

        self.comm = comm

        # Decide on parallel or serial batch clustering
        self.rank = self.comm.Get_rank()
        self.nprocs = self.comm.Get_size()
        self.do_parallel = (self.nprocs > 1)
        if self.rank == 0:
            print("Number of ranks: %s" % self.nprocs)
        # Check batching options
        self.batching = self.options.get("batch", False)

    def operate(self, *inputs, **kargs):

        if self.rank == 0:
            print("Reading in %s datasets." % len(inputs))

        # Get the sizes of each kosh dataset
        input_sizes = []
        desc = list(self.describe_entries())
        for i in range(len(inputs)):
            input_sizes.append(desc[i]["size"][0])

        if self.batching and self.do_parallel:
            r_data = _koshParallelClustering_(
                inputs, self.options, self.comm, input_sizes)
        else:
            r_data = _koshSerialClustering_(inputs, self.options)

        return r_data


def _koshParallelClustering_(inputs, options, comm, input_sizes):
    """
    :param inputs: One or more arrays of size (n_samples, n_features). datasets must have same number of n_features.
    :type inputs: kosh datasets
    :param flatten: Flattens data to two dimensions. (n_samples, n_features_1*n_features_2* ... *n_features_m)
    :type flatten: bool
    :param distance_function: distance metric 'euclidean', 'seuclidean', 'sqeuclidean',
                              'beuclidean', or user defined function. Defaults to 'euclidean'
    :type distance_function: string or user defined function
    :param scaling_function: Scaling function to use on data before it is clustered.
    :type scaling_function: string or user defined function
    :param batch_size: Size of the batches
    :type batch_size: int
    :param convergence_num: Converged if the data size is the same for 'num' iterations. The default is 2.
    :type convergence_num: int
    :param core_sample: Whether to retain a sample from the center of the cluster (core sample),
                        or a randomly chosen sample.
    :type core_sample: bool
    :param eps: The distance around a sample that defines its neighbors.
    :type eps: float
    :param min_samples: The minimum number of samples to form a cluster.
    :type min_samples: int
    :param output: The retained data or the indices to get the retained data from the original dataset.
    :type output: string
    :param format: Returns the indices as numpy array ('numpy') or defaults to pandas dataframe.
    :type format: string
    :returns: clustered subsample of original dataset, or indices of subsample
    :rtype: numpy array or pandas dataframe
    """

    # Read in the data in parallel; each processor has its own data
    data, global_ind = _koshParallelReader_(inputs, comm, input_sizes)

    # Parse the input arguments
    flatten = options.get("flatten", False)
    batch_size = options.get("batch_size", 10000)
    convergence_num = options.get("convergence_num", 2)
    distance_function = options.get("distance_function", "euclidean")
    scaling_function = options.get("scaling_function", "")
    core_sample = options.get("core_sample", True)
    output = options.get("output", "samples")
    eps = options.get('eps', .5)
    min_samples = options.get("min_samples", 2)
    # format = options.get("format", "numpy")

    local_data = makeBatchClusterParallel(data, global_ind, comm, flatten=flatten,
                                          batch_size=batch_size, convergence_num=convergence_num,
                                          distance_function=distance_function,
                                          scaling_function=scaling_function, core_sample=core_sample,
                                          output=output, eps=eps, min_samples=min_samples)

    return local_data


def _koshParallelReader_(inputs, comm, input_sizes):

    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    total_data_size = sum(input_sizes)

    if rank == 0:
        print("Total data size: %s" % total_data_size)

    # Divide all data as evenly as possible between ranks
    size_div = total_data_size // nprocs         # Some will have this much data
    procs_to_add_one = total_data_size % nprocs  # Others will need to +1

    # Create a list of all the data sizes needed
    data_to_procs = np.repeat(size_div, nprocs - procs_to_add_one)
    size_div_p_1 = np.repeat(size_div + 1, procs_to_add_one)
    data_to_procs = np.append(data_to_procs, size_div_p_1)

    # Get global indices
    start = sum(data_to_procs[0:rank])
    end = start + data_to_procs[rank]
    global_ind = np.arange(start, end)
    global_ind = global_ind.reshape(-1, 1)

    # Process for ranks to claim data assigned to them
    counter = 0
    data = []

    for i in range(len(input_sizes)):

        readData = True

        for d in range(input_sizes[i]):
            global_index = counter
            iStart = sum(data_to_procs[0:rank])
            iEnd = iStart + data_to_procs[rank]
            counter += 1
            if (global_index >= iStart) and (
                    global_index <= iEnd) and readData:

                start_local = d
                try:
                    ndata = len(np.concatenate(data))
                except BaseException:
                    ndata = 0

                end_local = min(input_sizes[i], iEnd - iStart + d - ndata)
                # input_ind = np.repeat(i, (end_local-start_local)).reshape(-1,1)
                # local_ind = np.arange(start_local, end_local).reshape(-1,1)
                # this_data = np.concatenate((this_data, global_ind), axis=1)
                data.append(inputs[i][start_local: end_local])

                readData = False

    data = np.concatenate(data)

    return data, global_ind


def _koshSerialClustering_(inputs, options):
    """
    :param inputs: One or more arrays of size (n_samples, n_features). datasets must have same number of n_features.
    :type inputs: kosh datasets
    :param method: DBSCAN or HAC (Hierarchical Agglomerative Clustering)
    :type method: str
    :param scaling_function: function for scaling the features
    :type scaling_function: String or callable
    :param flatten: Flattens data to two dimensions. (n_samples, n_features_1*n_features_2* ... *n_features_m)
    :type flatten: bool
    :param distance_function: distance metric 'euclidean', 'seuclidean', 'sqeuclidean',
                              'beuclidean', or user defined function. Defaults to 'euclidean'
    :type distance_function: string or user defined function
    :param batch: Whether to cluster data in batches
    :type batch: bool
    :param batch_size: Size of the batches
    :type batch_size: int
    :param convergence_num: Converged if the data size is the same for 'num' iterations. The default is 2.
    :type convergence_num: int
    :param eps: The distance around a sample that defines its neighbors. (Only for DBSCAN)
    :type eps: float
    :param min_samples: The minimum number of samples to form a cluster. (Only for DBSCAN)
    :type min_samples: int
    :param min_cluster_size: The smallest size grouping to be considered a cluster
    :type min_cluster_size: int
    :param HAC_distance_scaling: Scales the default distance (self.default_distance), Must be greater than zero
    :type HAC_distance_scaling: float
    :param HAC_distance_value: User defines cut-off distance for clustering
    :type HAC_distance_value: float
    :param Nclusters: Number of clusters to find (Instead of using a distance for clustering)
    :type Nclusters: int
    :param core_sample: Whether to retain a sample from the center of the cluster (core sample),
                        or a randomly chosen sample.
    :type core_sample: bool
    :param n_jobs: The number of parallel jobs to run. -1 means using all processors.
    :type n_jobs: int
    :param return_labels: Returns list of retained samples/indices and labels.
                          If using HDBSCAN, labels will be cluster labels and probabilities.
    :type return_labels: bool
    :param format: Returns the indices as numpy array ('numpy') or defaults to pandas dataframe.
    :type format: string
    :param output: The retained data or the indices to get the retained data from the original dataset.
    :type output: string
    :returns: clustered subsample of original dataset, or indices of subsample
    :rtype: numpy array or pandas dataframe
    """

    data = inputs[0][:]
    for input_ in inputs[1:]:
        data = np.append(data, input_[:], axis=0)

    method = options.get("method", "DBSCAN")
    scaling_function = options.get("scaling_function", "")
    flatten = options.get("flatten", False)

    distance_function = options.get("distance_function", "euclidean")
    core_sample = options.get("core_sample", True)
    Nclusters = options.get("Nclusters", -1)
    n_jobs = options.get("n_jobs", 1)
    return_labels = options.get("return_labels", False)
    output = options.get("output", "samples")

    format = options.get("format", "numpy")
    batch = options.get("batch", False)
    batch_size = options.get("batch_size", 10000)
    convergence_num = options.get("convergence_num", 2)

    my_cluster = Cluster(
        data,
        method=method,
        scaling_function=scaling_function,
        flatten=flatten)

    if method == 'DBSCAN':

        eps = options.get('eps', .5)
        min_samples = options.get("min_samples", 2)

        if batch:
            out = my_cluster.makeBatchCluster(batch_size=batch_size,
                                              convergence_num=convergence_num,
                                              output=output, core_sample=core_sample,
                                              Nclusters=Nclusters, n_jobs=n_jobs,
                                              eps=eps, min_samples=min_samples,
                                              distance_function=distance_function)
        else:
            my_cluster.makeCluster(eps=eps, min_samples=min_samples,
                                   distance_function=distance_function,
                                   Nclusters=Nclusters, n_jobs=n_jobs)

            labels = my_cluster.pd_data['clust_labels']

            out = my_cluster.subsample(output=output,
                                       core_sample=core_sample,
                                       distance_function=distance_function,
                                       n_jobs=n_jobs)

    elif method == 'HDBSCAN':

        min_cluster_size = options.get("min_cluster_size", 2)
        min_samples = options.get("min_samples", 2)

        my_cluster.makeCluster(min_cluster_size=min_cluster_size,
                               distance_function=distance_function,
                               min_samples=min_samples)

        labels = my_cluster.pd_data[["clust_labels", "probabilities"]]

        out = my_cluster.subsample(output=output,
                                   core_sample=core_sample,
                                   distance_function=distance_function)

    elif method == 'HAC':

        HAC_distance_scaling = options.get('HAC_distance_scaling', 1.0)
        HAC_distance_value = options.get("HAC_distance_value", -1)

        if batch:
            out = my_cluster.makeBatchCluster(batch_size,
                                              convergence_num=convergence_num,
                                              output=output, core_sample=core_sample,
                                              distance_function=distance_function,
                                              HAC_distance_scaling=HAC_distance_scaling,
                                              Nclusters=Nclusters,
                                              HAC_distance_value=HAC_distance_value,
                                              n_jobs=n_jobs)
        else:
            my_cluster.makeCluster(distance_function=distance_function,
                                   HAC_distance_scaling=HAC_distance_scaling,
                                   HAC_distance_value=HAC_distance_value,
                                   Nclusters=Nclusters)

            labels = my_cluster.pd_data['clust_labels']

            out = my_cluster.subsample(output=output,
                                       core_sample=core_sample,
                                       distance_function=distance_function,
                                       n_jobs=n_jobs)

    else:
        print("Error: no valid clustering method given")
        exit()

    if return_labels:
        if format == 'numpy':
            return [np.array(out), np.array(labels)]
        elif format == 'pandas':
            return [pd.DataFrame(out), labels]
    else:
        if format == 'numpy':
            return np.array(out)
        elif format == 'pandas':
            return pd.DataFrame(out)


class KoshHopkins(KoshOperator):
    """Calculates the Hopkins statistic or cluster tendency of the data

    """
    types = {"numpy": ["numpy", ]}

    def __init__(self, *args, **options):
        super(KoshHopkins, self).__init__(*args, **options)
        self.options = options

    def operate(self, *inputs, **kargs):
        """
        from a sample of the dataset. A value close to 0 means uniformly
        distributed, .5 means randomly distributed, and a value close to 1
        means highly clustered.

        :param inputs: One or more arrays of size (n_samples, n_features). Datasets must have same number of n_features.
        :type inputs: kosh datasets
        :param sample_ratio: Proportion of data for sample
        :type sample_ratio: float, between zero and one
        :param scaling_function: Scaling function to use on data before it is clustered.
        :type scaling_function: string or user defined function
        :param flatten: Flattens data to two dimensions. (n_samples, n_features_1*n_features_2* ... *n_features_m)
        :type flatten: bool
        :return: Hopkins statistic
        :rtype: float
        """

        data = inputs[0][:]
        for input_ in inputs[1:]:
            data = np.append(data, input_[:], axis=0)

        sample_ratio = self.options.get("sample_ratio", .1)
        scaling_function = self.options.get("scaling_function", '')
        flatten = self.options.get("flatten", False)

        cluster_object = Cluster(data, scaling_function=scaling_function, flatten=flatten)
        hopkins_stat = cluster_object.hopkins(sample_ratio=sample_ratio)
        return hopkins_stat


class KoshClusterLossPlot(KoshOperator):
    types = {"numpy": ["mpl", "mpl/png", "numpy"]}
    """Calculates sample size and estimated information loss for a range of distance values.
"""

    def __init__(self, *args, **options):
        super(KoshClusterLossPlot, self).__init__(*args, **options)
        self.options = options

    def operate(self, *inputs, **kargs):
        """
        :param inputs: One or more arrays of size (n_samples, n_features). Datasets must have same number of n_features.
        :type inputs: kosh datasets
        :param method: DBSCAN, HDBSCAN, or HAC (Hierarchical Agglomerative Clustering)
        :type method: string
        :param flatten: Flattens data to two dimensions. (n_samples, n_features_1*n_features_2* ... *n_features_m)
        :type flatten: bool
        :param val_range: Range of distance values to use for clustering/subsampling
        :type val_range: array
        :param val_type: Choose the type of value range for clustering: raw distance ('raw'),
                         scaled distance ('scaled'), or number of clusters ('Nclusters').
        :type val_type: string
        :param scaling_function: Scaling function to use on data before it is clustered.
        :type scaling_function: string or user defined function
        :param distance_function: A valid pairwise distance option from scipy.spatial.distance,
                                  or a user defined distance function.
        :type distance_function: string, or callable
        :param draw_plot: Whether to plot the plt object. otherwise it returns a list of three arrays:
                          the distance value range, loss estimate, and sample size.
                          You can pass a matplotlib Axes instance if you want.
        :type draw_plot: bool or matplotlib.pyplot.Axes object
        :param outputFormat: Returns the information as matplotlib pyplot object ('mpl'), png file ('mpl/png'),
                             or numpy array ('numpy')
        :type outputFormat: string
        :param min_samples: The minimum number of samples to form a cluster. (Only for DBSCAN)
        :type min_samples: int
        :param n_jobs: The number of parallel jobs to run. -1 means using all processors.
        :type n_jobs: int
        :return: plt object showing loss/sample size information, location of the saved file,
                 or an array with val_range, loss estimate, and sample size
        :rtype: object, string, array
        """

        data = inputs[0][:]
        for input_ in inputs[1:]:
            data = np.append(data, input_[:], axis=0)

        # self.outputDir        = self.options.get("outputDir", '.')
        self.fileNameTemplate = self.options.get(
            "fileNameTemplate", "./clusterLossPlot")
        method = self.options.get("method", "DBSCAN")
        flatten = self.options.get("flatten", False)
        val_range = self.options.get("val_range", np.linspace(1e-4, 1.5, 30))
        val_type = self.options.get("val_type", "raw")
        distance_function = self.options.get("distance_function", "euclidean")
        # options are: 'mpl',, 'mpl/png', 'numpy'
        outputFormat = self.options.get("outputFormat", 'mpl')
        min_samples = self.options.get("min_samples", 2)
        n_jobs = self.options.get("n_jobs", 1)
        scaling_function = self.options.get("scaling_function", '')

        cluster_object = Cluster(
            data,
            method=method,
            scaling_function=scaling_function,
            flatten=flatten)

        draw_plot = self.options.get("draw_plot", (outputFormat == 'mpl') or (outputFormat == 'mpl/png'))

        output = cluster_object.lossPlot(
            val_range=val_range,
            val_type=val_type,
            distance_function=distance_function,
            draw_plot=draw_plot,
            min_samples=min_samples,
            n_jobs=n_jobs)

        if outputFormat == 'mpl/png':
            fileName = "{}_{}_{:.2g}_{:.2g}.png".format(
                self.fileNameTemplate,
                distance_function, val_range[0], val_range[-1])
            output.savefig(fileName)
            return fileName
        else:  # return output for obj or array options
            return output

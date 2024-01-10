import numpy as np
import pandas as pd
from ..sampling_methods.cluster_sampling import Cluster
from ..sampling_methods.cluster_sampling import SubsampleWithLoss
from ..sampling_methods.cluster_sampling import ParallelClustering
from ..sampling_methods.cluster_sampling import SerialClustering
from .core import KoshOperator


class KoshCluster(KoshOperator):
    """Clusters together similar samples from a dataset, and then
    returns cluster representatives to form a non-redundant
    subsample of the original dataset. The datasets need to be of
    shape (n_samples, n_features). All datasets must have the same
    number of features. If the datasets are more than two dimensions
    there is an option to flatten them.
    """
    types = {"numpy": ["numpy", "pandas"]}

    def __init__(self, *args, **options):
        """
        :param inputs: One or more arrays of size (n_samples, n_features).
        datasets must have same number of n_features.
        :type inputs: kosh datasets
        :param flatten: Flattens data to two dimensions.
        (n_samples, n_features_1*n_features_2* ... *n_features_m)
        :type flatten: bool
        :param distance_function: distance metric 'euclidean', 'seuclidean',
        'sqeuclidean', 'beuclidean', or user defined function. Defaults to
        'euclidean'
        :type distance_function: string or user defined function
        :param scaling_function: Scaling function to use on data before it
        is clustered.
        :type scaling_function: string or user defined function
        :param batch: Whether to cluster data in batches
        :type batch: bool
        :param batch_size: Size of the batches
        :type batch_size: int
        :param gather_to: Which process to gather data to if samples are
        smaller than number of processes or batch size.
        type gather_to: int
        :param convergence_num: If int, converged after the data size is the same for
        'num' iterations. The default is 2. If float, converged after the change in data
        size is less than convergence_num*100 percent of the original data size.
        :type convergence_num: int or float between 0 and 1
        :param core_sample: Whether to retain a sample from the center of
        the cluster (core sample), or a randomly chosen sample.
        :type core_sample: bool
        :param eps: The distance around a sample that defines its neighbors.
        :type eps: float
        :param auto_eps: Use the algorithm to find the epsilon distance for
        clustering based on the desired information loss.
        :type auto_eps: bool
        :param eps_0: The initial epsilon guess for the auto eps algorithm.
        :type eps_0: float
        :param min_samples: The minimum number of samples to form a cluster.
        :type min_samples: int
        :param target_loss: The proportion of information loss allowed from removing
        samples from the original dataset. The default is .01 or 1% loss.
        :type target_loss: float
        :param verbose: Verbose message
        :type verbose: bool
        :param output: The retained data or the indices to get the retained
        data from the original dataset.
        :type output: string
        :param format: Returns the indices as numpy array ('numpy') or
        defaults to pandas dataframe.
        :type format: string
        :returns: A list containing: 1. The reduced dataset or indices to reduce the original
        dataset. 2. The estimated information loss or if using the auto eps algorithm (eps=-1)
        the second item in the list will be the epsilon value found with auto eps.
        :rtype: list with elements in the list being either numpy array or pandas dataframe
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

        # Verbose options
        self.verbose = self.options.get("verbose", False)
        options['verbose'] = self.verbose
        # Define primary rank
        self.primary = self.options.get("gather_to", 0)
        options['gather_to'] = self.primary
        self.pverbose = (self.rank == self.primary) and self.verbose

        if self.pverbose:
            print("Number of ranks: %s" % self.nprocs)

        # Check batching options
        self.batching = self.options.get("batch", False)
        self.batch_size = options.get("batch_size", 3000)
        # Guarantees value exists in options
        options['batch_size'] = self.batch_size

        # Check for automatic loss-based subsampling
        self.target_loss = self.options.get('target_loss', .01)
        self.autoEPS = self.options.get('auto_eps', False)

    def operate(self, *inputs, **kargs):
        """
        Checks for serial or parallel clustering and calls
        those functions
        """

        if self.pverbose:
            print("Reading in %s datasets." % len(inputs))

        # Get the sizes of each kosh dataset
        input_sizes = []
        for input_ in inputs:
            input_sizes.append(input_.shape[0])

        total_sample_size = sum(input_sizes)

        # Logical checks for batch and parallel clustering
        if (total_sample_size <= self.batch_size):
            self.batching = False
            if self.pverbose:
                print("Total sample size is less than batch size.")

        # Case where user has multiple processes but no batching
        if (not self.batching and self.do_parallel):
            self.batching = True
            if self.pverbose:
                print("Parallel requires batch=True;")
                print("Switching to batch clustering.")

        # Case where data size is smaller than number of processes
        if total_sample_size <= self.nprocs:
            self.do_parallel = False
            if self.pverbose:
                print("Total sample size is less than number of processors.")
                print("Switching to serial clustering.")
                print("Idling all non-primary processors.")
            if self.rank != self.primary:
                return [None, ]

        if not self.autoEPS:
            # Standard calls to operator just calls either option
            if self.batching and self.do_parallel:
                r_data = _koshParallelClustering_(inputs,
                                                  self.options,
                                                  self.comm,
                                                  input_sizes)
            else:
                r_data = _koshSerialClustering_(inputs, self.options)

        else:
            # AutoEPS will compute needed EPS for the desired loss
            #   and return a list with the data and found EPS value
            [data, epsActual] = _koshAutoEPS_(inputs,
                                              self.options,
                                              self.target_loss,
                                              input_sizes,
                                              self.comm,
                                              self.do_parallel)
            r_data = [data, epsActual]
            # When data is None return None instead of list
            if data is None:
                return [None, ]

        return r_data


def _koshAutoEPS_(inputs, options, target_loss, input_sizes, comm, parallel):
    """
    Finds the appropriate epsilon value for clustering based on the target_loss.
    """

    gather_to = options.get("gather_to", 0)
    verbose = options.get("verbose", False)

    if parallel:
        data, global_ind = _koshParallelReader_(inputs,
                                                comm,
                                                input_sizes,
                                                gather_to,
                                                verbose)
    else:
        global_ind = None
        data = inputs[0][:]
        for input_ in inputs[1:]:
            data = np.append(data, input_[:], axis=0)

    [data, epsActual] = SubsampleWithLoss(data,
                                          target_loss,
                                          options,
                                          parallel=parallel,
                                          comm=comm,
                                          indices=global_ind)

    return [data, epsActual]


def _koshParallelClustering_(inputs, options, comm, input_sizes):
    """
    Data are randomly distributed to processors and reduced with batch clustering.
    The surviving data are randomly mixed and reduced, and the process continues
    until convergence.
    """
    gather_to = options.get("gather_to")
    verbose = options.get("verbose")

    # Read in the data in parallel; each processor has its own data
    data, global_ind = _koshParallelReader_(inputs,
                                            comm,
                                            input_sizes,
                                            gather_to,
                                            verbose)

    [local_data, loss] = ParallelClustering(data, comm, global_ind, options)

    return [local_data, loss]


def _koshParallelReader_(inputs, comm, input_sizes, gather_to, verbose):
    """
    Based on input sizes of the datasets, processors read in the data they
    have been assigned. The data will be evenly distributed.
    """

    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    total_data_size = sum(input_sizes)

    pverbose = (rank == gather_to) and verbose

    if pverbose:
        print("Total data size: %s" % total_data_size)

    # Divide all data as evenly as possible between ranks
    # Some will have this much data
    size_div = total_data_size // nprocs
    # Others will need to +1
    procs_to_add_one = total_data_size % nprocs

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
                data.append(inputs[i][start_local: end_local])
                readData = False

    data = np.concatenate(data)

    return data, global_ind


def _koshSerialClustering_(inputs, options):
    """
    Reads in all the datasets and reduces data with cluster sampling.
    """

    data = inputs[0][:]
    for input_ in inputs[1:]:
        data = np.append(data, input_[:], axis=0)

    return_labels = options.get("return_labels", False)

    format = options.get("format", "numpy")

    [out, labels, loss] = SerialClustering(data, options)

    result = []

    # Return data as either numpy array or Pandas DataFrame
    if format == 'numpy':
        result.append(np.array(out))
    elif format == 'pandas':
        result.append(pd.DataFrame(out))
    else:
        print("Error: no valid output format given; numpy|pandas")

    # Optionally return labels instead of loss
    if return_labels:
        result.append(np.array(labels))
    else:
        result.append(loss)

    return result


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

        :param inputs: One or more arrays of size (n_samples, n_features).
        Datasets must have same number of n_features.
        :type inputs: kosh datasets
        :param sample_ratio: Proportion of data for sample
        :type sample_ratio: float, between zero and one
        :param scaling_function: Scaling function to use on data before
        it is clustered.
        :type scaling_function: string or user defined function
        :param flatten: Flattens data to two dimensions.
        (n_samples, n_features_1*n_features_2* ... *n_features_m)
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

        cluster_object = Cluster(data,
                                 scaling_function=scaling_function,
                                 flatten=flatten)
        hopkins_stat = cluster_object.hopkins(sample_ratio=sample_ratio)
        return hopkins_stat


class KoshClusterLossPlot(KoshOperator):
    types = {"numpy": ["mpl", "mpl/png", "numpy"]}
    """Calculates sample size and estimated information loss
    for a range of distance values.
"""

    def __init__(self, *args, **options):
        super(KoshClusterLossPlot, self).__init__(*args, **options)
        self.options = options

    def operate(self, *inputs, **kargs):
        """
        :param inputs: One or more arrays of size (n_samples, n_features).
        Datasets must have same number of n_features.
        :type inputs: kosh datasets
        :param method: DBSCAN, HDBSCAN, or HAC
        (Hierarchical Agglomerative Clustering)
        :type method: string
        :param flatten: Flattens data to two dimensions.
        (n_samples, n_features_1*n_features_2* ... *n_features_m)
        :type flatten: bool
        :param val_range: Range of distance values to use for
        clustering/subsampling
        :type val_range: array
        :param val_type: Choose the type of value range for clustering:
        raw distance ('raw'), scaled distance ('scaled'), or number of
        clusters ('Nclusters').
        :type val_type: string
        :param scaling_function: Scaling function to use on data before
        it is clustered.
        :type scaling_function: string or user defined function
        :param distance_function: A valid pairwise distance option from
        scipy.spatial.distance, or a user defined distance function.
        :type distance_function: string, or callable
        :param draw_plot: Whether to plot the plt object. otherwise it
        returns a list of three arrays: the distance value range,
        loss estimate, and sample size. You can pass a matplotlib Axes
        instance if desired.
        :type draw_plot: bool or matplotlib.pyplot.Axes object
        :param outputFormat: Returns the information as matplotlib pyplot
        object ('mpl'), png file ('mpl/png'),
                             or numpy array ('numpy')
        :type outputFormat: string
        :param min_samples: The minimum number of samples to form a cluster.
        (Only for DBSCAN)
        :type min_samples: int
        :param n_jobs: The number of parallel jobs to run. -1 means
        using all processors.
        :type n_jobs: int
        :return: plt object showing loss/sample size information, location
        of the saved file, or an array with val_range, loss estimate, and
        sample size
        :rtype: object, string, array
        """

        data = inputs[0][:]
        for input_ in inputs[1:]:
            data = np.append(data, input_[:], axis=0)

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

        draw_plot = self.options.get("draw_plot",
                                     (outputFormat == 'mpl') or
                                     (outputFormat == 'mpl/png'))

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

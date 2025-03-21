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
        :param non_dim_return: The option to return non-dimensional information loss.
        :type non_dim_return: bool
        :param data_source: The rank the kosh operator will obtain the dataset from. -1 is
        the default and all ranks will provide the datasets, a positive int will indicate
        data is read in from a specific rank.
        :type data_source: int
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

        # Option to return non-dimensional info loss
        self.non_dim_return = self.options.get('non_dim_return', False)
        options['non_dim_return'] = self.non_dim_return

        # Where is data coming from
        self.data_source = self.options.get('data_source', -1)
        options['data_source'] = self.data_source

    def operate(self, *inputs, **kargs):
        """
        Checks for serial or parallel clustering and calls
        those functions
        """

        if self.pverbose:
            print("Reading in %s datasets." % len(inputs))

        # Get the sizes of each kosh dataset
        input_sizes = []

        if self.data_source > -1:
            if self.rank == self.data_source:
                for input_ in inputs:
                    input_sizes.append(input_.shape[0])
                total_sample_size = sum(input_sizes)
            else:
                total_sample_size = None

            total_sample_size = self.comm.bcast(total_sample_size, root=self.data_source)
            input_sizes = self.comm.bcast(input_sizes, root=self.data_source)

        else:
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
        #     or data size is smaller than batch size
        if (total_sample_size <= self.nprocs or total_sample_size <= self.batch_size):
            self.do_parallel = False
            if self.pverbose:
                if total_sample_size <= self.nprocs:
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
                r_data = _koshSerialClustering_(inputs,
                                                self.options)

        else:
            # AutoEPS will compute needed EPS for the desired loss
            #   and return a list with the data and found EPS value
            [data, loss, epsActual] = _koshAutoEPS_(inputs,
                                                    self.options,
                                                    self.target_loss,
                                                    input_sizes,
                                                    self.comm,
                                                    self.do_parallel)
            r_data = [data, loss, epsActual]
            # When data is None return None instead of list
            if data is None:
                return [None, ]

        return r_data


def _koshAutoEPS_(inputs, options, target_loss, input_sizes, comm, parallel):
    """
    Finds the appropriate epsilon value for clustering based on the target_loss.
    """
    import numpy as np

    from ..sampling_methods.cluster_sampling import SubsampleWithLoss
    gather_to = options.get("gather_to", 0)
    data_source = options.get("data_source", -1)
    verbose = options.get("verbose", False)

    if parallel:
        data, global_ind = _koshParallelReader_(inputs,
                                                comm,
                                                input_sizes,
                                                gather_to,
                                                data_source,
                                                verbose)
    else:
        global_ind = None
        data = inputs[0][:]
        for input_ in inputs[1:]:
            data = np.append(data, input_[:], axis=0)

    [data, loss, epsActual] = SubsampleWithLoss(data,
                                                target_loss,
                                                options,
                                                parallel=parallel,
                                                comm=comm,
                                                indices=global_ind)

    return [data, loss, epsActual]


def _koshParallelClustering_(inputs, options, comm, input_sizes):
    """
    Data are randomly distributed to processors and reduced with batch clustering.
    The surviving data are randomly mixed and reduced, and the process continues
    until convergence.
    """
    from ..sampling_methods.cluster_sampling import ParallelClustering
    from ..sampling_methods.cluster_sampling import GetMaxLoss
    gather_to = options.get("gather_to")
    verbose = options.get("verbose")
    non_dim_return = options.get("non_dim_return")
    data_source = options.get("data_source")

    # Read in the data in parallel; each processor has its own data
    data, global_ind = _koshParallelReader_(inputs,
                                            comm,
                                            input_sizes,
                                            gather_to,
                                            data_source,
                                            verbose)

    [reduced_data, loss] = ParallelClustering(data, comm, global_ind, options)

    if non_dim_return:
        # Estimate the max loss to calculate the non-dimensional information loss
        [epsMax, maxLoss] = GetMaxLoss(data,
                                       options,
                                       parallel=True,
                                       comm=comm,
                                       indices=global_ind)
        loss = loss/maxLoss

    return [reduced_data, loss]


def _koshParallelReader_(inputs, comm, input_sizes, gather_to, data_source, verbose):
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

    def get_indices(rank, nprocs):
        import numpy as np
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

    if data_source > -1:
        if rank == data_source:
            for p in range(1, nprocs):
                data, global_ind = get_indices(p, nprocs)
                comm.send(data, dest=p, tag=p)
                comm.send(global_ind, dest=p, tag=p+500)
            data, global_ind = get_indices(data_source, nprocs)
        else:
            data = comm.recv(source=data_source, tag=rank)
            global_ind = comm.recv(source=data_source, tag=rank+500)

    else:
        data, global_ind = get_indices(rank, nprocs)

    return data, global_ind


def _koshSerialClustering_(inputs, options):
    """
    Reads in all the datasets and reduces data with cluster sampling.
    """
    from ..sampling_methods.cluster_sampling import SerialClustering
    from ..sampling_methods.cluster_sampling import GetMaxLoss
    import numpy as np
    from pandas import DataFrame
    data = inputs[0][:]
    for input_ in inputs[1:]:
        data = np.append(data, input_[:], axis=0)

    return_labels = options.get("return_labels", False)
    non_dim_return = options.get("non_dim_return")
    format = options.get("format", "numpy")

    [out, loss, labels] = SerialClustering(data, options)

    # Return data as either numpy array or Pandas DataFrame
    if format == 'numpy':
        reduced_data = np.array(out)
    elif format == 'pandas':
        reduced_data = DataFrame(out)
    else:
        print("Error: no valid output format given; numpy|pandas")

    if non_dim_return:
        # Estimate the max loss to calculate the non-dimensional information loss
        [epsMax, maxLoss] = GetMaxLoss(data, options)
        if maxLoss > 0.0:
            loss = loss/maxLoss
        else:
            loss = 0.0

    result = [reduced_data, loss]

    # Optionally return labels as a third element in the return list
    if return_labels:
        result.append(np.array(labels))

    return result


class KoshHopkins(KoshOperator):
    """Calculates the Hopkins statistic or cluster tendency of the data

    """
    types = {"numpy": ["numpy", ]}

    def __init__(self, *args, **options):
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
        super(KoshHopkins, self).__init__(*args, **options)
        self.options = options

    def operate(self, *inputs, **kargs):
        import numpy as np
        from ..sampling_methods.cluster_sampling import Cluster

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
    """Calculates sample size and estimated information loss
    for a range of distance values.
    """
    types = {"numpy": ["mpl", "png", "numpy"]}

    def __init__(self, *args, **options):
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
        :param non_dim_return: The option to return non-dimensional information loss.
        :type non_dim_return: bool
        :param data_source: The ranks the kosh operator will obtain the dataset from. -1 is
        the default and all ranks will provide the datasets, a positive int will indicate
        data is read in from a specific rank.
        :type data_source: int
        :param verbose: Verbose message
        :type verbose: bool
        :param draw_plot: Whether to plot the plt object. otherwise it
        returns a list of three arrays: the distance value range,
        loss estimate, and sample size. You can pass a matplotlib Axes
        instance if desired.
        :type draw_plot: bool or matplotlib.pyplot.Axes object
        :param fileNameTemplate: The name to save the plot object
        :type fileNameTemplate: string
        :param outputFormat: Returns the information as matplotlib pyplot
        object ('mpl'), png file ('png'),
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
        super(KoshClusterLossPlot, self).__init__(*args, **options)
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

        # Option to return non-dimensional info loss
        self.non_dim_return = self.options.get('non_dim_return', False)
        options['non_dim_return'] = self.non_dim_return

        # Where is data coming from
        self.data_source = self.options.get('data_source', -1)
        options['data_source'] = self.data_source

    def operate(self, *inputs, **kargs):
        """Calculates sample size and estimated information loss
        for a range of distance values.
        """
        from ..sampling_methods.cluster_sampling import SerialClustering
        from ..sampling_methods.cluster_sampling import ParallelClustering
        from ..sampling_methods.cluster_sampling import GetMaxLoss
        import numpy as np
        from tqdm import tqdm
        import matplotlib.pyplot as plt
        try:
            from mpi4py import MPI
        except ImportError:
            print("mpi4py not found. Using serial clustering for plot.")

        if self.pverbose:
            print("Reading in %s datasets." % len(inputs))

        input_sizes = []
        if self.data_source > -1:
            if self.rank == self.data_source:
                for input_ in inputs:
                    input_sizes.append(input_.shape[0])
                total_sample_size = sum(input_sizes)
            else:
                total_sample_size = None
            total_sample_size = self.comm.bcast(total_sample_size, root=self.data_source)
            input_sizes = self.comm.bcast(input_sizes, root=self.data_source)
        else:
            for input_ in inputs:
                input_sizes.append(input_.shape[0])
            total_sample_size = sum(input_sizes)
        self.comm.barrier()

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
        #     or data size is smaller than batch size
        if (total_sample_size <= self.nprocs or total_sample_size <= self.batch_size):
            self.do_parallel = False
            if self.pverbose:
                if total_sample_size <= self.nprocs:
                    print("Total sample size is less than number of processors.")
                print("Switching to serial clustering.")
                print("Idling all non-primary processors.")
            if self.rank != self.primary:
                return [None, ]

        # Plotting or output options
        self.fileNameTemplate = self.options.get(
            "fileNameTemplate", "./clusterLossPlot")
        val_range = self.options.get("val_range", np.linspace(1e-4, 1.5, 30))
        # options are: 'mpl', 'png', 'numpy'
        outputFormat = self.options.get("outputFormat", 'mpl')

        if self.do_parallel:

            # Read in the data in parallel; each processor has its own data
            data, global_ind = _koshParallelReader_(inputs,
                                                    self.comm,
                                                    input_sizes,
                                                    self.primary,
                                                    self.data_source,
                                                    self.verbose)
            # Collect sample size and loss for each distance value
            sample_sizes = []
            loss_list = []
            for ival in tqdm(val_range):
                # Save eps value in options
                self.options["eps"] = ival

                [reduced_data, loss] = ParallelClustering(data, self.comm, global_ind, self.options)

                if self.non_dim_return:
                    # Estimate the max loss to calculate the non-dimensional information loss
                    [epsMax, maxLoss] = GetMaxLoss(data,
                                                   self.options,
                                                   parallel=True,
                                                   comm=self.comm,
                                                   indices=global_ind)
                    loss = loss/maxLoss

                # Check if any procs have None
                if reduced_data is not None:
                    reduced_size = reduced_data.shape[0]
                else:
                    reduced_size = 0

                # Gather data sizes together to get total data size
                final_size = self.comm.allreduce(reduced_size, MPI.SUM)
                sample_sizes.append(final_size)

                if self.rank == self.primary:
                    loss_list.append(loss)
                loss_list = self.comm.bcast(loss_list, root=self.primary)

        else:
            # Cluster and reduce data serially
            global_ind = None
            data = inputs[0][:]
            for input_ in inputs[1:]:
                data = np.append(data, input_[:], axis=0)

            # Collect sample size and loss for each distance value
            sample_sizes = []
            loss_list = []
            for ival in tqdm(val_range):
                # Save eps value in options
                self.options["eps"] = ival

                [out, loss, labels] = SerialClustering(data, self.options)

                if self.non_dim_return:
                    # Estimate the max loss to calculate the non-dimensional info loss
                    [epsMax, maxLoss] = GetMaxLoss(data, self.options)
                    if maxLoss > 0.0:
                        loss = loss/maxLoss
                    else:
                        loss = 0.0

                sample_sizes.append(out.shape[0])
                loss_list.append(loss)

        draw_plot = self.options.get("draw_plot",
                                     (outputFormat == 'mpl') or
                                     (outputFormat == 'png'))

        if draw_plot:
            if self.rank == self.primary:

                if isinstance(draw_plot, plt.Axes):
                    # user sent us where to plot
                    ax1 = draw_plot
                    fig = ax1.get_figure()
                else:
                    fig, ax1 = plt.subplots()

                color = 'tab:red'
                ax1.set_xlabel('Clustering Parameter')
                ax1.set_ylabel('Loss', color=color)
                ax1.plot(val_range, loss_list, color=color)
                ax1.tick_params(axis='y', labelcolor=color)

                # instantiate a second axes that shares the same x-axis
                ax2 = ax1.twinx()

                color = 'tab:blue'

                # we already handled the x-label with ax1
                ax2.set_ylabel('Sample Size', color=color)
                ax2.plot(val_range, sample_sizes, color=color)
                ax2.tick_params(axis='y', labelcolor=color)

                # otherwise the right y-label is slightly clipped
                fig.tight_layout()

            else:
                fig = None

            if outputFormat == 'png':
                distance_function = self.options.get("distance_function", "euclidean")
                fileName = "{}_{}_{:.2g}_{:.2g}.png".format(
                    self.fileNameTemplate,
                    distance_function, val_range[0], val_range[-1])

                if self.rank == self.primary:
                    fig.savefig(fileName)

                return fileName

            else:
                return fig

        else:  # return output for obj or array options
            return [val_range, loss_list, sample_sizes]

from kosh.sampling_methods.cluster_sampling import Cluster
from kosh.sampling_methods.cluster_sampling.Clustering import makeBatchClusterParallel
import numpy as np
import time
import pytest
from unittest import TestCase


class ClusteringTest(TestCase):
    @pytest.mark.mpi_skip
    def test_import_cluster(self):
        self.assertTrue(callable(Cluster))

    @pytest.mark.mpi_skip
    def test_subsample_2d(self):

        Nsamples = 100
        Ndims = 2

        data = np.random.random((Nsamples, Ndims))
        dataR = np.zeros((Nsamples, Ndims))

        dataR[:, :] = data[0, :]

        dataT = np.concatenate((data, dataR), axis=0)

        my_cluster = Cluster(dataT, method='HAC')
        my_cluster.makeCluster(HAC_distance_scaling=0.01)
        data_sub = my_cluster.subsample()

        self.assertLessEqual(data_sub.shape[0], Nsamples)

        # TODO: data_sub2 = my_cluster.makeBatchCluster()

    @pytest.mark.mpi_skip
    def test_subsample_2d_DBSCAN(self):

        Nsamples = 100
        Ndims = 2

        data = np.random.random((Nsamples, Ndims))
        dataR = np.zeros((Nsamples, Ndims))

        dataR[:, :] = data[0, :]

        dataT = np.concatenate((data, dataR), axis=0)

        my_cluster = Cluster(dataT, method='DBSCAN')
        my_cluster.makeCluster(eps=0.001)
        data_sub = my_cluster.subsample()

        self.assertLessEqual(data_sub.shape[0], Nsamples)

    @pytest.mark.mpi_skip
    def test_subsample_rank3_DBSCAN(self):

        Nsamples = 100
        Ndimsx = 2
        Ndimsy = 2

        data = np.random.random((Nsamples, Ndimsx, Ndimsy))
        dataR = np.zeros((Nsamples, Ndimsx, Ndimsy))

        dataR[:, :, :] = data[0, :, :]

        dataT = np.concatenate((data, dataR), axis=0)

        my_cluster = Cluster(dataT, method='DBSCAN', flatten=True)
        my_cluster.makeCluster(eps=0.001)
        data_sub = my_cluster.subsample()

        self.assertLessEqual(data_sub.shape[0], Nsamples)

    @pytest.mark.mpi_skip
    def test_subsample_2d_NHAC(self):

        Nsamples = 100
        Ndims = 2

        data = np.random.random((Nsamples, Ndims))
        dataR = np.zeros((Nsamples, Ndims))

        dataR[:, :] = data[0, :]

        dataT = np.concatenate((data, dataR), axis=0)

        # Test when N-samples are requested
        my_cluster = Cluster(dataT, method='HAC')
        my_cluster.makeCluster(Nclusters=30)
        data_sub = my_cluster.subsample()
        self.assertTrue((data_sub.shape[0] >= 28) & (data_sub.shape[0] <= 32))

    @pytest.mark.mpi_skip
    def test_subsample_2d_NDBSCAN(self):

        Nsamples = 100
        Ndims = 2

        data = np.random.random((Nsamples, Ndims))
        dataR = np.zeros((Nsamples, Ndims))

        dataR[:, :] = data[0, :]

        dataT = np.concatenate((data, dataR), axis=0)

        my_cluster = Cluster(dataT, method='DBSCAN')

        # var_scale = dataT.std()**2
        # bounds = [1.e-7 * var_scale, var_scale]

        # Ncluster = []
        # EPS = []
        # for eps in np.linspace(bounds[0],bounds[1],50):#
        # my_cluster.makeCluster( eps = eps )
        # Ncluster.append( my_cluster.original_clusters )
        # EPS.append(eps)

        # import matplotlib.pyplot as plt
        # plt.plot( EPS, Ncluster, 'b-o')
        # plt.pause(.1)

        my_cluster.makeCluster(Nclusters=5)

        self.assertTrue((my_cluster.original_clusters >= 4) & (my_cluster.original_clusters <= 6))

    @pytest.mark.mpi_skip
    def test_hopkins(self):

        Nsamples = 100
        Ndims = 4

        data = np.random.random((Nsamples, Ndims))
        dataR = np.zeros((Nsamples, Ndims))

        dataR[:, :] = data[0, :]

        dataT = np.concatenate((data, dataR), axis=0)

        my_cluster = Cluster(dataT, scaling_function='min_max')
        hop_test = my_cluster.hopkins()
        self.assertTrue((hop_test > 0) & (hop_test < 1))

    @pytest.mark.mpi_skip
    def test_loss_plot(self):

        Nsamples = 100
        Ndims = 4

        data = np.random.random((Nsamples, Ndims))
        dataR = np.zeros((Nsamples, Ndims))

        dataR[:, :] = data[0, :]

        dataT = np.concatenate((data, dataR), axis=0)

        my_cluster = Cluster(dataT)

        vr = np.linspace(1e-6, 2, 50)
        results = my_cluster.lossPlot(
            val_range=vr, distance_function='euclidean')

        self.assertEqual(len(results[0]), len(results[1]))

    @pytest.mark.mpi_skip
    def test_batch_subsample_2d(self):

        Nsamples = 2200
        Ndims = 2

        data = np.random.random((Nsamples, Ndims))
        dataR = np.zeros((Nsamples, Ndims))

        dataR[:, :] = data[0, :]

        dataT = np.concatenate((data, dataR), axis=0)
        tic = time.time()
        my_cluster = Cluster(dataT, method='DBSCAN')

        data_sub = my_cluster.makeBatchCluster(batch_size=800,
                                               verbose=False,
                                               eps=.003,
                                               output='indices')
        toc = time.time()
        print("Total data: " + str(dataT.shape[0]))
        print("Batch time: " + str(toc - tic))
        self.assertLessEqual(data_sub.shape[0], Nsamples)

    @pytest.mark.mpi(min_size=2)
    def test_batch_parallel(self):
        import numpy as np
        from mpi4py import MPI
        from sklearn.datasets import make_blobs

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        nsamples = 50000

        x1 = np.arange(13.63636, 136.63636, 13.63636)
        x2 = np.arange(13.63636, 136.63636, 13.63636)
        centers = []
        for n in x1:
            for m in x2:
                centers.append((n, m))

        data, y = make_blobs(
            n_samples=nsamples, centers=centers, random_state=rank)

        # Make global array indices for all procs
        global_ind = np.arange(nsamples) + nsamples * rank

        n_samples = data.shape[0]
        total_samples = comm.allreduce(n_samples, op=MPI.SUM)

        rdata = makeBatchClusterParallel(data, global_ind, comm, batch_size=10000,
                                         convergence_num=20, scaling_function='min_max',
                                         output='samples', core_sample=True,
                                         distance_function="euclidean", verbose=True,
                                         eps=.05, min_samples=2)

        # Since retained data << 50k, all data will be on rank=0, and rdata
        # will be None for all other procs
        total_subsamples = 100
        if rank == 0:
            total_subsamples = rdata.shape[0]
            print("Original data size %s" % total_samples)
            print("New data size %s" % total_subsamples)

        self.assertLess(total_subsamples, 110)

    # @pytest.mark.mpi
    # #def test_batch_parallel(self):
    # from sampling_methods.cluster_sampling.Clustering import makeBatchClusterParallel, numpyParallelReader
    # import numpy as np
    # from mpi4py import MPI
    # from sklearn.datasets import make_blobs
    # import glob

    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # nprocs = comm.Get_size()

    # nsamples = 50000

    # x1 = np.arange(13.63636, 136.63636, 13.63636)
    # x2 = np.arange(13.63636, 136.63636, 13.63636)
    # centers = []
    # for n in x1:
    # 	for m in x2:
    # 		centers.append((n,m))

    # if rank%2 == 0:
    # 	filename = "p_data_%s.npy" % rank
    # 	data, y = make_blobs(n_samples=nsamples, centers=centers, random_state=rank)
    # 	np.save(filename, data)

    # comm.barrier()

    # input_sizes = []
    # data_files = []
    # if rank ==0:
    # 	data_files = glob.glob("p_data_*.npy")
    # 	input_sizes = []
    # 	for d in data_files:
    # 		f_data = np.load(d)
    # 		input_sizes.append(f_data.shape[0])

    # input_sizes = comm.bcast(input_sizes, root=0)
    # data_files = comm.bcast(data_files, root=0)

    # data = numpyParallelReader(data_files, input_sizes, comm)

    # n_samples = data.shape[0]
    # total_samples = comm.allreduce(n_samples, op=MPI.SUM)

    # rdata = makeBatchClusterParallel(data, 10000, comm, input_sizes,
    # 				convergence_num=20, output='samples', core_sample=True,
    # 				scaling_function='min_max', distance_function="euclidean",
    # 				verbose=True, eps=.05, min_samples=2)

    # n_subsamples = rdata.shape[0]
    # total_subsamples = comm.allreduce(n_subsamples, op=MPI.SUM)
    # if rank == 0:
    # 	print("Original data size %s" % total_samples )
    # 	print("New data size %s" % total_subsamples )

    # 	#assert( total_subsamples < 110 )

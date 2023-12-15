from kosh.sampling_methods.cluster_sampling import Cluster
from kosh.sampling_methods.cluster_sampling.Clustering import makeBatchClusterParallel
import numpy as np
import pytest
from unittest import TestCase


class ClusteringTest(TestCase):
    @pytest.mark.mpi_skip
    def test_import_cluster(self):
        self.assertTrue(callable(Cluster))

    @pytest.mark.mpi_skip
    def test_subsample_HAC(self):

        Nsamples = 100
        Ndims = 2

        data = np.random.random((Nsamples, Ndims))
        dataR = np.zeros((Nsamples, Ndims))

        dataR[:, :] = data[0, :]

        dataT = np.concatenate((data, dataR), axis=0)

        my_cluster = Cluster(dataT, method='HAC')
        my_cluster.makeCluster(HAC_distance_scaling=0.01)
        data_sub = my_cluster.subsample()

        self.assertLessEqual(data_sub.shape[0], dataT.shape[0])

    @pytest.mark.mpi_skip
    def test_subsample_DBSCAN(self):

        Nsamples = 100
        Ndims = 2

        data = np.random.random((Nsamples, Ndims))
        dataR = np.zeros((Nsamples, Ndims))

        dataR[:, :] = data[0, :]

        dataT = np.concatenate((data, dataR), axis=0)

        my_cluster = Cluster(dataT, method='DBSCAN')
        my_cluster.makeCluster(eps=0.001)
        data_sub = my_cluster.subsample()

        self.assertLessEqual(data_sub.shape[0], dataT.shape[0])

    @pytest.mark.mpi_skip
    def test_subsample_flatten_DBSCAN(self):

        Nsamples = 100
        Ndimsx = 2
        Ndimsy = 2
        total_features = Ndimsx + Ndimsy

        data = np.random.random((Nsamples, Ndimsx, Ndimsy))
        dataR = np.zeros((Nsamples, Ndimsx, Ndimsy))

        dataR[:, :, :] = data[0, :, :]

        dataT = np.concatenate((data, dataR), axis=0)

        my_cluster = Cluster(dataT, method='DBSCAN', flatten=True)
        my_cluster.makeCluster(eps=0.001)
        data_sub = my_cluster.subsample()

        self.assertEqual(data_sub.shape[1], total_features)

    @pytest.mark.mpi_skip
    def test_subsample_NHAC(self):

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
    def test_subsample_NDBSCAN(self):

        Nsamples = 100
        Ndims = 2

        data = np.random.random((Nsamples, Ndims))
        dataR = np.zeros((Nsamples, Ndims))

        dataR[:, :] = data[0, :]

        dataT = np.concatenate((data, dataR), axis=0)

        my_cluster = Cluster(dataT, method='DBSCAN')

        my_cluster.makeCluster(Nclusters=5)

        self.assertTrue(
            (my_cluster.original_clusters >= 4) & (
                my_cluster.original_clusters <= 6))

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
        results = my_cluster.lossPlot(val_range=vr,
                                      distance_function='euclidean')

        self.assertEqual(len(results[0]), len(results[1]))

    @pytest.mark.mpi_skip
    def test_batch_subsample(self):

        Nsamples = 2200
        Ndims = 2

        data = np.random.random((Nsamples, Ndims))
        dataR = np.zeros((Nsamples, Ndims))

        dataR[:, :] = data[0, :]

        dataT = np.concatenate((data, dataR), axis=0)

        my_cluster = Cluster(dataT, method='DBSCAN')

        data_sub = my_cluster.makeBatchCluster(batch_size=800,
                                               verbose=False,
                                               eps=.003,
                                               output='indices')

        self.assertLessEqual(data_sub.shape[0], dataT.shape[0])

    @pytest.mark.mpi_skip
    def test_convergence_float(self):

        Nsamples = 1000
        Ndims = 2

        data = np.random.random((Nsamples, Ndims))
        dataR = np.zeros((Nsamples, Ndims))

        dataR[:, :] = data[0, :]

        dataT = np.concatenate((data, dataR), axis=0)

        # Test convergence float works
        my_cluster2 = Cluster(dataT, method='DBSCAN')
        data_sub2 = my_cluster2.makeBatchCluster(
            eps=0.001, batch_size=50, convergence_num=.01)
        self.assertLessEqual(data_sub2.shape[0], dataT.shape[0])

    @pytest.mark.mpi_skip
    def test_convergence_int(self):

        Nsamples = 1000
        Ndims = 2

        data = np.random.random((Nsamples, Ndims))
        dataR = np.zeros((Nsamples, Ndims))

        dataR[:, :] = data[0, :]

        dataT = np.concatenate((data, dataR), axis=0)

        # Test convergence int works
        my_cluster1 = Cluster(dataT, method='DBSCAN')
        data_sub1 = my_cluster1.makeBatchCluster(
            eps=0.001, batch_size=50, convergence_num=30)
        self.assertLessEqual(data_sub1.shape[0], dataT.shape[0])

    @pytest.mark.mpi_skip
    def test_wrong_convergence_input(self):

        Nsamples = 1000
        Ndims = 2

        data = np.random.random((Nsamples, Ndims))
        dataR = np.zeros((Nsamples, Ndims))

        dataR[:, :] = data[0, :]

        dataT = np.concatenate((data, dataR), axis=0)

        my_cluster1 = Cluster(dataT, method='DBSCAN')

        # Test 2.3 input raises error
        with pytest.raises(AssertionError):
            my_cluster1.makeBatchCluster(
                eps=0.001, batch_size=50, convergence_num=2.3)

    @pytest.mark.mpi_skip
    def test_numpy_convergence_input(self):

        Nsamples = 1000
        Ndims = 2

        data = np.random.random((Nsamples, Ndims))
        dataR = np.zeros((Nsamples, Ndims))

        dataR[:, :] = data[0, :]

        dataT = np.concatenate((data, dataR), axis=0)

        my_cluster1 = Cluster(dataT, method='DBSCAN')

        # Test numpy array will work
        cv = np.array(2, dtype=int)
        data_sub1 = my_cluster1.makeBatchCluster(
            eps=0.001, batch_size=50, convergence_num=cv)
        self.assertLessEqual(data_sub1.shape[0], dataT.shape[0])

    @pytest.mark.mpi(min_size=2)
    def test_batch_parallel(self):
        import numpy as np
        from mpi4py import MPI
        from sklearn.datasets import make_blobs

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        nsamples = 5000

        x1 = np.arange(13.63636, 136.63636, 13.63636)
        x2 = np.arange(13.63636, 136.63636, 13.63636)
        centers = []
        for n in x1:
            for m in x2:
                centers.append((n, m))

        data, y = make_blobs(n_samples=nsamples,
                             centers=centers,
                             random_state=rank)

        # Make global array indices for all procs
        global_ind = np.arange(nsamples) + nsamples * rank

        rdata = makeBatchClusterParallel(data,
                                         comm,
                                         global_ind=global_ind,
                                         batch_size=3000,
                                         convergence_num=3,
                                         scaling_function='min_max',
                                         output='samples',
                                         eps=.05)[:]
        samp = rdata[0]

        # Since retained data << 50k, all data will be on rank=0, and rdata
        # will be None for all other procs
        total_subsamples = 100
        if rank == 0:
            total_subsamples = samp.shape[0]

        self.assertLess(total_subsamples, 110)

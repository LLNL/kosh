import random
import numpy as np
import string
from kosh.operators import KoshCluster, KoshClusterLossPlot, KoshHopkins
import h5py
import pytest
from os.path import exists
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # noqa
from koshbase import KoshTest  # noqa

# size of random string
rand_n = 7


class KoshTestClusters(KoshTest):
    @pytest.mark.mpi_skip
    def test_HACsubsample_kosh(self):

        Nsamples = 100
        Ndims = 2

        # generate random strings
        res = ''.join(random.choices(string.ascii_uppercase +
                                     string.digits, k=rand_n))
        fileName = 'data_' + str(res) + '.h5'

        res2 = ''.join(random.choices(string.ascii_uppercase +
                                      string.digits, k=rand_n))
        fileName2 = 'data_' + str(res2) + 'h5'

        # Create random data, add redundant data
        data = np.random.random((Nsamples, Ndims))
        dataR = np.zeros((Nsamples, Ndims))
        dataR[:, :] = data[0, :]
        dataT = np.concatenate((data, dataR), axis=0)

        data2 = np.random.random((Nsamples, Ndims))
        dataR = np.zeros((Nsamples, Ndims))
        dataR[:, :] = data2[0, :]
        dataT2 = np.concatenate((data2, dataR), axis=0)

        h5f_1 = h5py.File(fileName, 'w')
        h5f_1.create_dataset('dataset_1', data=dataT)
        h5f_1.close()

        h5f_2 = h5py.File(fileName2, 'w')
        h5f_2.create_dataset('dataset_2', data=dataT2)
        h5f_2.close()

        # Create a new store (erase if exists)
        store, uri = self.connect()

        # Add "dataT" to store...
        dataset = store.create("kosh_example1")
        dataset.associate([fileName, fileName2], "hdf5")

        # use Kosh operator to subsample data based off of clustering
        data_subsample = KoshCluster(dataset["dataset_1"], dataset["dataset_2"],
                                     method="HAC", HAC_distance_scaling=.01,
                                     output="samples")[:]
        samp = data_subsample[0]

        self.assertLessEqual(samp.shape[0], Nsamples * 2)

        # Cleanup
        store.close()
        os.remove(fileName)
        os.remove(fileName2)
        os.remove(uri)

    @pytest.mark.mpi_skip
    def test_DBSCANsubsample_kosh(self):

        Nsamples = 100
        Ndims = 2

        # generate random strings
        res = ''.join(random.choices(string.ascii_uppercase +
                                     string.digits, k=rand_n))
        fileName = 'data_' + str(res) + '.h5'

        dataL = np.random.random((Nsamples, Ndims)) * .1
        dataR = np.random.random((Nsamples, Ndims)) * .1
        dataR[:, 0] += 1.0
        dataR[:, 1] += 1.0
        dataT = np.concatenate((dataL, dataR), axis=0)

        h5f = h5py.File(fileName, 'w')
        h5f.create_dataset('dataset_1', data=dataT)
        h5f.close()

        # Create a new store (erase if exists)
        store, uri = self.connect()

        # Add "dataT" to store...

        dataset = store.create("kosh_example1")
        dataset.associate(fileName, "hdf5")

        # use Kosh operator to subsample data based off of clustering
        data_subsample = KoshCluster(
            dataset["dataset_1"],
            method="DBSCAN",
            eps=.1,
            output="samples")[:]
        samp = data_subsample[0]

        self.assertLessEqual(samp.shape[0], Nsamples)

        # Cleanup
        os.remove(fileName)
        store.close()
        os.remove(uri)

    @pytest.mark.mpi_skip
    def test_HDBSCANsubsample_kosh(self):

        Nsamples = 100
        Ndims = 2

        # generate random strings
        res = ''.join(random.choices(string.ascii_uppercase +
                                     string.digits, k=rand_n))
        fileName = 'data_' + str(res) + '.h5'

        dataL = np.random.random((Nsamples, Ndims)) * .1
        dataR = np.random.random((Nsamples, Ndims)) * .1
        dataR[:, 0] += 1.0
        dataR[:, 1] += 1.0
        dataT = np.concatenate((dataL, dataR), axis=0)

        h5f = h5py.File(fileName, 'w')
        h5f.create_dataset('dataset_1', data=dataT)
        h5f.close()

        # Create a new store (erase if exists)
        store, uri = self.connect()

        # Add "dataT" to store...
        dataset = store.create("kosh_example1")
        dataset.associate(fileName, "hdf5")

        # use Kosh operator to subsample data based off of clustering
        data_subsample = KoshCluster(dataset["dataset_1"], method="HDBSCAN",
                                     min_cluster_size=2, output="samples")[:]
        samp = data_subsample[0]

        self.assertLessEqual(samp.shape[0], Nsamples)

        # Cleanup
        os.remove(fileName)
        store.close()
        os.remove(uri)

    @pytest.mark.mpi_skip
    def test_koshHopkins(self):

        from sklearn import datasets

        X = datasets.load_iris().data
        h5f = h5py.File('iris_data.h5', 'w')
        h5f.create_dataset('dataset_1', data=X)
        h5f.close()

        # Create a new store (erase if exists)
        store, uri = self.connect()

        dataset = store.create("kosh_example1")
        dataset.associate('iris_data.h5', "hdf5")

        hop_stat = KoshHopkins(
            dataset["dataset_1"],
            scaling_function='standard')[:]

        self.assertTrue((hop_stat > .78) & (hop_stat < .92))

        # Cleanup
        os.remove('iris_data.h5')
        store.close()
        os.remove(uri)

    @pytest.mark.mpi_skip
    def test_clusterLossPlot(self):

        import matplotlib
        matplotlib.use("agg", force=True)
        import matplotlib.pyplot as plt

        try:
            os.remove("clusterLossPlot.png")
        except BaseException:
            pass

        Nsamples = 1000
        Ndims = 2

        # generate random strings
        res = ''.join(random.choices(string.ascii_uppercase +
                                     string.digits, k=rand_n))
        fileName = 'data_' + str(res) + '.h5'

        dataL = np.random.random((Nsamples, Ndims)) * .1
        dataR = np.random.random((Nsamples, Ndims)) * .1
        dataR[:, 0] += 1.0
        dataR[:, 1] += 1.0
        dataT = np.concatenate((dataL, dataR), axis=0)

        h5f = h5py.File(fileName, 'w')
        h5f.create_dataset('dataset_1', data=dataT)
        h5f.close()

        # Create a new store (erase if exists)
        store, uri = self.connect()

        # Add "dataT" to store...

        dataset = store.create("kosh_example1")
        dataset.associate(fileName, "hdf5")

        vr = np.linspace(1e-4, .008, 10)

        # Test outputFormat=mpl/png
        lossPlotFile = KoshClusterLossPlot(dataset["dataset_1"], val_range=vr,
                                           scaling_function='standard',
                                           outputFormat='mpl/png')[:]
        self.assertTrue(exists(lossPlotFile))

        # Test outputFormat=mpl
        lossPlot = KoshClusterLossPlot(dataset["dataset_1"], val_range=vr,
                                       scaling_function='standard',
                                       outputFormat='mpl')[:]
        self.assertEqual(type(lossPlot), type(plt.figure()))

        # Test outputFormat=numpy
        lossPlotData = KoshClusterLossPlot(dataset["dataset_1"], val_range=vr,
                                           scaling_function='standard',
                                           outputFormat='numpy')[:]
        self.assertEqual(len(lossPlotData), 3)

        # Test passing a mpl plot to it.
        fig = plt.figure(figsize=(25, 20))
        axes = fig.subplots(nrows=2, ncols=2)
        for i in range(4):
            lossPlotData = KoshClusterLossPlot(dataset["dataset_1"], val_range=vr,
                                               scaling_function='standard',
                                               outputFormat='mpl',
                                               draw_plot=axes[i // 2, i % 2])[:]
        fig.savefig(lossPlotFile)

        # Cleanup
        os.remove(fileName)
        os.remove(lossPlotFile)
        store.close()
        os.remove(uri)

    @pytest.mark.mpi_skip
    def test_batchClusteringSubsamples_kosh(self):

        Nsamples = 500
        Ndims = 2

        # generate random strings
        res = ''.join(random.choices(string.ascii_uppercase +
                                     string.digits, k=rand_n))
        fileName = 'data_' + str(res) + '.h5'

        res2 = ''.join(random.choices(string.ascii_uppercase +
                                      string.digits, k=rand_n))
        fileName2 = 'data_' + str(res2) + '.h5'

        # Create random data, add redundant data
        data = np.random.random((Nsamples, Ndims))
        dataR = np.zeros((Nsamples, Ndims))
        dataR[:, :] = data[0, :]
        dataT = np.concatenate((data, dataR), axis=0)

        data2 = np.random.random((Nsamples, Ndims))
        dataR = np.zeros((Nsamples, Ndims))
        dataR[:, :] = data2[0, :]
        dataT2 = np.concatenate((data2, dataR), axis=0)

        h5f_1 = h5py.File(fileName, 'w')
        h5f_1.create_dataset('dataset_1', data=dataT)
        h5f_1.close()

        h5f_2 = h5py.File(fileName2, 'w')
        h5f_2.create_dataset('dataset_2', data=dataT2)
        h5f_2.close()

        # Create a new store (erase if exists)
        store, uri = self.connect()

        # Add "dataT" to store...
        dataset = store.create("kosh_example1")
        dataset.associate([fileName, fileName2], "hdf5")

        # Test DBSCAN batching
        data_subsample2 = KoshCluster(dataset["dataset_1"], dataset["dataset_2"],
                                      method="DBSCAN", eps=.001, output="indices",
                                      batch=True, batch_size=250, convergence_num=2)[:]
        samp = data_subsample2[0]

        self.assertLessEqual(samp.shape[0], dataT.shape[0]*2)

        # Cleanup
        os.remove(fileName)
        os.remove(fileName2)
        store.close()
        os.remove(uri)

    @pytest.mark.mpi_skip
    def test_auto_eps_kosh(self):
        Nsamples = 100
        Ndims = 2

        # generate random strings
        res = ''.join(random.choices(string.ascii_uppercase +
                                     string.digits, k=rand_n))
        fileName = 'data_' + str(res) + '.h5'

        dataL = np.random.random((Nsamples, Ndims)) * .1
        dataR = np.random.random((Nsamples, Ndims)) * .1
        dataR[:, 0] += 1.0
        dataR[:, 1] += 1.0
        dataT = np.concatenate((dataL, dataR), axis=0)

        h5f = h5py.File(fileName, 'w')
        h5f.create_dataset('dataset_1', data=dataT)
        h5f.close()

        # Create a new store (erase if exists)
        store, uri = self.connect()

        # Add "dataT" to store...

        dataset = store.create("kosh_example1")
        dataset.associate(fileName, "hdf5")

        # use Kosh operator to subsample data based off of clustering
        data_subsample = KoshCluster(
            dataset["dataset_1"],
            method="DBSCAN",
            auto_eps=True,
            eps_0=.1,
            output="samples")[:]

        data = data_subsample[0]

        self.assertLessEqual(data.shape[0], dataT.shape[0])

        # Cleanup
        os.remove(fileName)
        store.close()
        os.remove(uri)

    @pytest.mark.mpi(min_size=2)
    def test_parallel_clustering(self):
        from sklearn.datasets import make_blobs
        from mpi4py import MPI

        comm = MPI.COMM_WORLD

        rank = comm.Get_rank()

        x1 = np.arange(13.63636, 136.63636, 13.63636)
        x2 = np.arange(13.63636, 136.63636, 13.63636)
        centers = []
        for n in x1:
            for m in x2:
                centers.append((n, m))

        data, y = make_blobs(n_samples=1000, centers=centers, random_state=0)

        # generate random strings
        fileName = ""
        if rank == 0:
            res = ''.join(random.choices(string.ascii_uppercase +
                                         string.digits, k=rand_n))
            fileName = 'data_' + str(res) + '.h5'

            h5f_1 = h5py.File(fileName, 'w')
            h5f_1.create_dataset('dataset_1', data=data)
            h5f_1.close()

        fileName = comm.bcast(fileName, root=0)

        # Create a new store (erase if exists)
        store, uri = self.connect()

        # Add "dataT" to store...
        dataset = store.create("kosh_example1")

        dataset.associate(fileName, "hdf5")

        # Test parallel DBSCAN
        data_subsample = KoshCluster(dataset["dataset_1"], method="DBSCAN",
                                     eps=.04, output="samples", scaling_function='min_max',
                                     batch=True, batch_size=500, convergence_num=5)[:]
        samp = data_subsample[0]

        if rank == 0:
            self.assertEqual(samp.shape[0], 100)
        else:
            self.assertIsNone(samp)

        comm.Barrier()
        if rank == 0:
            # Cleanup
            os.remove(fileName)
        store.close()
        if rank == 0:
            os.remove(uri)

    @pytest.mark.mpi(min_size=4)
    def test_gather_to_rank1(self):
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        fileName = ""

        if rank == 0:
            data = np.random.rand(2, 2)

            # size of random string
            rand_n = 7

            # generate random strings
            fileName = ""

            res = ''.join(random.choices(string.ascii_uppercase +
                                         string.digits, k=rand_n))
            fileName = 'data_' + str(res) + '.h5'

            h5f_1 = h5py.File(fileName, 'w')
            h5f_1.create_dataset('dataset_1', data=data)
            h5f_1.close()

        fileName = comm.bcast(fileName, root=0)

        # Create a new store (erase if exists)
        store, uri = self.connect()

        # Add "dataT" to store...
        dataset = store.create("kosh_example1")
        dataset.associate(fileName, "hdf5")

        # Test parallel DBSCAN
        data_subsample = KoshCluster(dataset["dataset_1"], method="DBSCAN",
                                     eps=.04, output="samples", gather_to=1,
                                     scaling_function='min_max',
                                     batch=True, batch_size=3000)[:]

        if rank == 1:
            self.assertIsInstance(data_subsample[0], np.ndarray)
        else:
            self.assertIsNone(data_subsample[0], None)

        comm.Barrier()
        # Cleanup
        if rank == 0:
            os.remove(fileName)
        store.close()
        if rank == 0:
            os.remove(uri)

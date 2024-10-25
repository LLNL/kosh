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
        fileName2 = 'data_' + str(res2) + '.h5'

        # Create random data
        data = np.random.random((Nsamples, Ndims))

        data2 = np.random.random((Nsamples, Ndims))

        h5f_1 = h5py.File(fileName, 'w')
        h5f_1.create_dataset('dataset_1', data=data)
        h5f_1.close()

        h5f_2 = h5py.File(fileName2, 'w')
        h5f_2.create_dataset('dataset_2', data=data2)
        h5f_2.close()

        # Create a new store (erase if exists)
        store, uri = self.connect()

        # Add "dataT" to store...
        dataset = store.create("kosh_example1")
        dataset.associate([fileName, fileName2], "hdf5")

        # use Kosh operator to subsample data based off of clustering
        data_subsample = KoshCluster(dataset["dataset_1"],
                                     dataset["dataset_2"],
                                     method="HAC",
                                     HAC_distance_scaling=.01,
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

        np.random.seed(3)
        data = np.random.random((Nsamples, Ndims))

        h5f = h5py.File(fileName, 'w')
        h5f.create_dataset('dataset_1', data=data)
        h5f.close()

        # Create a new store (erase if exists)
        store, uri = self.connect()

        # Add "dataT" to store...

        dataset = store.create("kosh_example1")
        dataset.associate(fileName, "hdf5")

        # use Kosh operator to subsample data based off of clustering
        data_subsample = KoshCluster(dataset["dataset_1"],
                                     method="DBSCAN",
                                     eps=.1,
                                     output="samples")[:]
        samp = data_subsample[0]
        loss = data_subsample[1]

        self.assertEqual(samp.shape[0], 18)
        self.assertAlmostEqual(loss, 11.508393200414897)

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

        data = np.random.random((Nsamples, Ndims))
        dataR = np.zeros((Nsamples, Ndims))
        dataR[:, :] = data[0, :]
        dataT = np.concatenate((data, dataR), axis=0)

        h5f = h5py.File(fileName, 'w')
        h5f.create_dataset('dataset_1', data=dataT)
        h5f.close()

        # Create a new store (erase if exists)
        store, uri = self.connect()

        # Add "dataT" to store...
        dataset = store.create("kosh_example1")
        dataset.associate(fileName, "hdf5")

        # use Kosh operator to subsample data based off of clustering
        output = KoshCluster(dataset["dataset_1"],
                             method="HDBSCAN",
                             min_cluster_size=2,
                             output="samples")[:]
        samp = output[0]

        self.assertLessEqual(samp.shape[0], Nsamples*2)

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

        Nsamples = 2000
        Ndims = 2

        # generate random strings
        res = ''.join(random.choices(string.ascii_uppercase +
                                     string.digits, k=rand_n))
        fileName = 'data_' + str(res) + '.h5'

        data = np.random.random((Nsamples, Ndims))

        h5f = h5py.File(fileName, 'w')
        h5f.create_dataset('dataset_1', data=data)
        h5f.close()

        # Create a new store (erase if exists)
        store, uri = self.connect()

        # Add "dataT" to store...

        dataset = store.create("kosh_example1")
        dataset.associate(fileName, "hdf5")

        vr = np.linspace(1e-4, .008, 10)

        # Test outputFormat=png
        lossPlotFile = KoshClusterLossPlot(dataset["dataset_1"],
                                           val_range=vr,
                                           scaling_function='standard',
                                           outputFormat='png')[:]
        self.assertTrue(exists(lossPlotFile))

        # Test outputFormat=mpl
        lossPlot = KoshClusterLossPlot(dataset["dataset_1"],
                                       val_range=vr,
                                       scaling_function='standard',
                                       outputFormat='mpl')[:]
        self.assertEqual(type(lossPlot), type(plt.figure()))

        # Test outputFormat=numpy
        lossPlotData = KoshClusterLossPlot(dataset["dataset_1"],
                                           val_range=vr,
                                           scaling_function='standard',
                                           outputFormat='numpy')[:]
        self.assertEqual(len(lossPlotData), 3)

        # Test passing a mpl plot to it.
        fig = plt.figure(figsize=(25, 20))
        axes = fig.subplots(nrows=2, ncols=2)
        for i in range(4):
            lossPlotData = KoshClusterLossPlot(dataset["dataset_1"],
                                               val_range=vr,
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

        Nsamples = 1000
        Ndims = 2

        # generate random strings
        res = ''.join(random.choices(string.ascii_uppercase +
                                     string.digits, k=rand_n))
        fileName = 'data_' + str(res) + '.h5'

        res2 = ''.join(random.choices(string.ascii_uppercase +
                                      string.digits, k=rand_n))
        fileName2 = 'data_' + str(res2) + '.h5'

        # Create random data, add redundant data
        np.random.seed(3)
        data = np.random.random((Nsamples, Ndims))

        data2 = np.random.random((Nsamples, Ndims))

        h5f_1 = h5py.File(fileName, 'w')
        h5f_1.create_dataset('dataset_1', data=data)
        h5f_1.close()

        h5f_2 = h5py.File(fileName2, 'w')
        h5f_2.create_dataset('dataset_2', data=data2)
        h5f_2.close()

        # Create a new store (erase if exists)
        store, uri = self.connect()

        # Add "dataT" to store...
        dataset = store.create("kosh_example1")
        dataset.associate([fileName, fileName2], "hdf5")

        # Test DBSCAN batching
        output = KoshCluster(dataset["dataset_1"],
                             dataset["dataset_2"],
                             method="DBSCAN",
                             eps=.01,
                             output="indices",
                             batch=True,
                             batch_size=250,
                             convergence_num=2,
                             non_dim_return=True)[:]
        samp = output[0]
        loss = output[1]

        self.assertEqual(samp.shape[0], 1512)
        self.assertAlmostEqual(loss, 0.00425026866881826)

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
        np.random.seed(3)
        data = np.random.random((Nsamples, Ndims))

        h5f = h5py.File(fileName, 'w')
        h5f.create_dataset('dataset_1', data=data)
        h5f.close()

        # Create a new store (erase if exists)
        store, uri = self.connect()

        # Add "dataT" to store...

        dataset = store.create("kosh_example1")
        dataset.associate(fileName, "hdf5")

        # Specify information loss we will test for
        target_loss = .01

        # use Kosh operator to subsample data based off of clustering
        output = KoshCluster(dataset["dataset_1"],
                             method="DBSCAN",
                             auto_eps=True,
                             target_loss=target_loss,
                             eps_0=.1,
                             output="samples",
                             non_dim_return=True)[:]

        data = output[0]
        actual_loss = output[1]
        eps_found = output[2]

        self.assertEqual(data.shape[0], 84)
        self.assertAlmostEqual(target_loss, round(actual_loss, 2))
        self.assertEqual(eps_found, 0.035010144321470385)

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

        # generate random strings
        fileName = ""
        if rank == 0:
            data, y = make_blobs(n_samples=1000, centers=centers, random_state=0)

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
        output = KoshCluster(dataset["dataset_1"],
                             method="DBSCAN",
                             eps=.04,
                             output="samples",
                             scaling_function='min_max',
                             batch=True,
                             batch_size=500,
                             convergence_num=5,
                             non_dim_return=True)[:]

        if rank == 0:
            samp = output[0]
            loss = output[1]

            self.assertEqual(samp.shape[0], 100)
            self.assertAlmostEqual(loss, 0.0180640359336712)
        else:
            self.assertIsNone(output[0])

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

            # generate random strings
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
        output = KoshCluster(dataset["dataset_1"],
                             method="DBSCAN",
                             eps=.04,
                             output="samples",
                             gather_to=1,
                             scaling_function='min_max',
                             batch=True,
                             batch_size=3000,
                             non_dim_return=True)[:]

        if rank == 1:
            self.assertIsInstance(output[0], np.ndarray)
        else:
            self.assertIsNone(output[0], None)

        comm.Barrier()
        # Cleanup
        if rank == 0:
            os.remove(fileName)
        store.close()
        if rank == 0:
            os.remove(uri)

    @pytest.mark.mpi(min_size=2)
    def test_parallel_data_source(self):
        import kosh
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        Nsamples = 100
        Ndims = 2

        if rank == 0:
            # generate random strings
            res = ''.join(random.choices(string.ascii_uppercase +
                                         string.digits, k=rand_n))
            fileName = 'data_' + str(res) + '.h5'
            np.random.seed(3)
            data = np.random.random((Nsamples, Ndims))

            f1 = h5py.File(fileName, 'w')
            f1.create_dataset('dataset_1', data=data)
            f1.close()
        elif rank == 1:
            data2 = [[0]]

            f2 = h5py.File("none_data.h5", 'w')
            f2.create_dataset('dataset_2', data=data2)
            f2.close()

        # Create a new store (erase if exists)
        store, uri = self.connect()

        # Add "dataT" to store...
        if rank == 0:
            dataset = store.create("kosh_example1")
            dataset.associate(fileName, "hdf5")
        else:
            dataset = store.create("kosh_example2")
            dataset.associate("none_data.h5", "hdf5")

        @kosh.numpy_operator
        def fake_op(*inputs):
            new_col = inputs[0][:, 0] + 1.0
            new_col = new_col.reshape(-1, 1)
            return np.hstack(tup=(inputs[0][:], new_col))

        if rank == 0:
            processed_data = fake_op(dataset['dataset_1'])
        else:
            processed_data = dataset['dataset_2']

        # use Kosh operator to subsample data based off of clustering
        output = KoshCluster(processed_data,
                             method="DBSCAN",
                             eps=.16,
                             output="indices",
                             scaling_function='min_max',
                             batch=True,
                             batch_size=25,
                             data_source=0,
                             non_dim_return=True)[:]

        indices = output[0]
        actual_loss = output[1]

        all_indices = comm.allgather(indices)
        flat_indices = np.sort(np.concatenate(all_indices))

        indices_expected = np.array([0, 2, 4, 10, 15, 17, 18, 22, 23,
                                     25, 27, 29, 32, 34, 38, 40, 41,
                                     47, 51, 52, 54, 63, 64, 65, 71,
                                     76, 87, 89, 91])
        loss_expected = 0.1538495698951048

        self.assertEqual(flat_indices.all(), indices_expected.all())
        self.assertAlmostEqual(actual_loss, loss_expected)

        # Cleanup
        if rank == 0:
            os.remove(fileName)
        elif rank == 1:
            os.remove("none_data.h5")
        store.close()
        os.remove(uri)

    @pytest.mark.mpi(min_size=2)
    def test_LossPlot_parallel(self):
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        import matplotlib
        matplotlib.use("agg", force=True)
        import matplotlib.pyplot as plt

        try:
            os.remove("clusterLossPlot.png")
        except BaseException:
            pass

        Nsamples = 2000
        Ndims = 2

        fileName = ""

        if rank == 0:
            # generate random strings
            res = ''.join(random.choices(string.ascii_uppercase +
                          string.digits, k=rand_n))
            fileName = 'data_' + str(res) + '.h5'

            data = np.random.random((Nsamples, Ndims))

            h5f = h5py.File(fileName, 'w')
            h5f.create_dataset('dataset_1', data=data)
            h5f.close()

        fileName = comm.bcast(fileName, root=0)

        # Create a new store (erase if exists)
        store, uri = self.connect()

        # Add "dataT" to store...

        dataset = store.create("kosh_example1")
        dataset.associate(fileName, "hdf5")

        vr = np.linspace(1e-4, .008, 10)

        # Test outputFormat=png
        lossPlotFile = KoshClusterLossPlot(dataset["dataset_1"],
                                           val_range=vr,
                                           scaling_function='standard',
                                           outputFormat='png')[:]

        comm.Barrier()
        if rank == 0:
            self.assertTrue(exists(lossPlotFile))

        # # Test outputFormat=mpl
        lossPlot = KoshClusterLossPlot(dataset["dataset_1"],
                                       val_range=vr,
                                       scaling_function='standard',
                                       outputFormat='mpl')[:]

        if rank == 0:
            self.assertEqual(type(lossPlot), type(plt.figure()))

        # Test outputFormat=numpy
        lossPlotData = KoshClusterLossPlot(dataset["dataset_1"],
                                           val_range=vr,
                                           scaling_function='standard',
                                           outputFormat='numpy')[:]
        if rank == 0:
            self.assertEqual(len(lossPlotData), 3)

        # Test passing a mpl plot to it.
        fig = plt.figure(figsize=(25, 20))
        axes = fig.subplots(nrows=2, ncols=2)

        for i in range(4):
            lossPlot = KoshClusterLossPlot(dataset["dataset_1"],
                                           val_range=vr,
                                           scaling_function='standard',
                                           outputFormat='mpl',
                                           draw_plot=axes[i // 2, i % 2])[:]
        if rank == 0:
            self.assertEqual(type(lossPlot), type(plt.figure()))

        comm.Barrier()
        # Cleanup
        if rank == 0:
            os.remove(fileName)
            os.remove(lossPlotFile)
        store.close()
        if rank == 0:
            os.remove(uri)

    @pytest.mark.mpi(min_size=2)
    def test_LossPlot_data_source(self):
        from mpi4py import MPI
        import kosh
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        import matplotlib
        matplotlib.use("agg", force=True)

        try:
            os.remove("clusterLossPlot.png")
        except BaseException:
            pass

        Nsamples = 2000
        Ndims = 2

        fileName = ""

        if rank == 0:
            # generate random strings
            res = ''.join(random.choices(string.ascii_uppercase +
                                         string.digits, k=rand_n))
            fileName = 'data_' + str(res) + '.h5'

            data = np.random.random((Nsamples, Ndims))

            h5f = h5py.File(fileName, 'w')
            h5f.create_dataset('dataset_1', data=data)
            h5f.close()
        elif rank == 1:
            data2 = [[0]]

            f2 = h5py.File("none_data.h5", 'w')
            f2.create_dataset('dataset_2', data=data2)
            f2.close()

        # Create a new store (erase if exists)
        store, uri = self.connect()

        # Add "data" to store...
        if rank == 0:
            dataset = store.create("kosh_example1")
            dataset.associate(fileName, "hdf5")
        else:
            dataset = store.create("kosh_example2")
            dataset.associate("none_data.h5", "hdf5")

        @kosh.numpy_operator
        def fake_op(*inputs):
            new_col = inputs[0][:, 0] + 1.0
            new_col = new_col.reshape(-1, 1)
            return np.hstack(tup=(inputs[0][:], new_col))

        if rank == 0:
            processed_data = fake_op(dataset['dataset_1'])
        else:
            processed_data = dataset['dataset_2']

        vr = np.linspace(1e-4, .008, 10)

        # Test outputFormat=png
        lossPlotFile = KoshClusterLossPlot(processed_data,
                                           val_range=vr,
                                           scaling_function='min_max',
                                           data_source=0,
                                           batch=True,
                                           batch_size=250,
                                           fileNameTemplate='doo_wop',
                                           outputFormat='png')[:]

        if rank == 0:
            assert exists(lossPlotFile)

        comm.Barrier()
        # Cleanup
        if rank == 0:
            os.remove(fileName)
            os.remove(lossPlotFile)
            os.remove("none_data.h5")
        store.close()
        if rank == 0:
            os.remove(uri)

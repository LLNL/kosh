try:
    import conduit
    has_conduit = True
except ImportError:
    has_conduit = False
from .utils import get_ids_for_rank, get_mpi_tools
from .core import KoshTransformer
import numpy


class SidreFeatureMetrics(KoshTransformer):
    """Numpy-based Metrics
    this returns a dictionary of metrics for a feature (sent as input)
    """
    types = {"sidre/path": ["dict"]}

    def __init__(self, *args, **kargs):
        if not has_conduit:
            raise RuntimeError(
                "Could not import conduit, Conduit-based transformers are not available")
        super(SidreFeatureMetrics, self).__init__(*args, **kargs)

    def transform(self, input_, format):
        rank, size, comm = get_mpi_tools()
        from mpi4py import MPI
        if rank != 0:
            stats = None
        else:
            stats = {}
        ioh, pth = input_

        sp0 = pth.split("/fields")[0]

        mesh = sp0.split("/")[-1]
        # First let's figure out the number of domains
        ndoms = conduit.Node()
        ioh.read(
            ndoms,
            "root/blueprint_index/{}/state/number_of_domains".format(mesh))
        ndoms = int(ndoms.value())

        # which domains do I read?
        my_domains = get_ids_for_rank(ndoms)

        bp_path = conduit.Node()
        ioh.read(bp_path, pth)
        bp_path = bp_path.value()
        vals = conduit.Node()
        # now read the data for this rank
        sum = 0
        N = 0
        mn = 1.e999
        mx = -1.e999
        rank_vals = None
        dtype = -1
        for domain in my_domains:
            dom_path = "{}/".format(domain)
            dom_path += bp_path + "/values"
            ioh.read(vals, dom_path)
            data = vals.value()
            mn = min(data.min(), mn)
            mx = max(data.max(), mx)
            N += len(data.flat)
            sum += data.sum()
            if rank_vals is None:
                rank_vals = data
            else:
                rank_vals = numpy.concatenate((rank_vals, data))
                dtype = rank_vals.dtype

        if rank == 0:
            for i in range(1, size):
                N += comm.recv(source=i, tag=1)
                sum += comm.recv(source=i, tag=2)
                mn = min(mn, comm.recv(source=i, tag=3))
                mx = max(mx, comm.recv(source=i, tag=4))
                comm.send(dtype, dest=i, tag=5)
        else:
            comm.send(N, dest=0, tag=1)
            comm.send(sum, dest=0, tag=2)
            comm.send(mn, dest=0, tag=3)
            comm.send(mx, dest=0, tag=4)
            dtype = comm.recv(source=0, tag=5)

        # Histogram from: https://ascent.readthedocs.io/en/latest/Tutorial_CloverLeaf_Demos.html
        # compute bins on global extents
        bins = numpy.linspace(mn, mx)

        if rank_vals is not None:  # DAta on this rank
            # get histogram counts for local data
            hist, bin_edges = numpy.histogram(rank_vals, bins=bins)
        else:
            hist = numpy.zeros((len(bins) - 1), dtype=dtype)

        # declare var for reduce results
        hist_all = numpy.zeros_like(hist)
        # sum histogram counts with MPI to get final histogram
        comm.Allreduce(hist, hist_all, op=MPI.SUM)

        if rank == 0:
            stats["histogram"] = hist
            stats["min"] = float(mn)
            stats["max"] = float(mx)
            stats["mean"] = float(sum) / float(N)

        # Now we can compute the std dev
        std = 0.
        if stats is not None:
            # We have data on this rank
            for domain in my_domains:
                dom_path = "{}/".format(domain)
                dom_path += bp_path + "/values"
                ioh.read(vals, dom_path)
                std += ((vals.value() - stats["mean"])**2).sum()

        if rank == 0:
            for i in range(1, size):
                std += comm.recv(source=i, tag=6)
            stats["std"] = float(numpy.sqrt(std / N))
        else:
            comm.send(std, dest=0, tag=6)

        stats = comm.bcast(stats, root=0)
        return stats

from __future__ import print_function, division
import sys
import numpy


def get_mpi_tools():
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    except ImportError:
        # no mpi
        # we'll fake it
        class Comm():
            def Get_size(self):
                return 1

            def Get_rank(self):
                return 0
        comm = Comm()
        MPI = False

    rank = comm.Get_rank()
    size = comm.Get_size()
    return rank, size, comm


def get_ids_for_rank(total):
    rank, size, comm = get_mpi_tools()
    if isinstance(total, int):
        total = list(range(total))
    if len(total) == 1:
        if rank == 0:
            return total
        else:
            return []
    n_per_rank = len(total) // size
    offset = int(rank * n_per_rank)
    if rank == (comm.size - 1):
        ids = total[offset: len(total)]
    else:
        ids = total[offset: offset + n_per_rank]
    return ids


def send_or_gather(data, dest=0, tag=0, verbose=False):
    out = None
    rank, size, comm = get_mpi_tools()
    if rank != dest:
        if isinstance(data, numpy.ndarray):
            comm.send(data.shape, dest=dest, tag=tag)
            comm.Send(data, dest=dest, tag=tag+1)
        else:
            comm.send(data, dest=dest)
    else:
        for rk in range(size):
            if rk == dest:
                empty = data
            else:
                if verbose:
                    MPIPrint("receiving from rank: {}".format(rk))
                sh = comm.recv(source=rk, tag=tag)
                empty = numpy.empty(sh, dtype=data.dtype)
                comm.Recv(empty, source=rk, tag=tag+1)
            if out is None:
                out = [empty, ]
            else:
                out = numpy.concatenate((out, empty), axis=0)
    return out


def MPIPrint(s, pre=""):
    rank, _, _ = get_mpi_tools()
    print("{}{}: {}".format(pre, rank, s), file=sys.stderr)
    sys.stderr.flush()

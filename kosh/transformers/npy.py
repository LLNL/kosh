from .core import KoshTransformer, kosh_cache_dir
import os
import numpy
from .utils import comm, rank, size, get_ids_for_rank, MPIPrint
import time


def make_slices_args(ndims, axis, start, end):
    if axis < 0:
        pos = ndims + axis
    else:
        pos = axis
    args = ()
    for i in range(ndims):
        if i == pos:
            args += (slice(start, end),)
        else:
            args += (slice(0, None),)
    return args


class KoshSimpleNpCache(KoshTransformer):
    def save(self, signature, *arrays):
        cache_file = os.path.join(self.cache_dir, signature)
        numpy.savez(cache_file, *arrays)

    def load(self, signature):
        cache_file = os.path.join(self.cache_dir, signature) + ".npz"
        npz = numpy.load(cache_file)
        out = [npz[x] for x in npz.files]
        if len(out) == 1:
            out = out[0]
        return out

    def transform(self, input):
        return input


class Take(KoshSimpleNpCache):
    """Equivalent of numpy's take, MPI enbabled"""
    types = {"numpy": ["numpy", ]}

    def __init__(self,
                 cache_dir=kosh_cache_dir,
                 cache=True,
                 indices=[],
                 axis=0,
                 verbose=False):

        super(Take, self).__init__(
            cache_dir, cache, indices=indices, axis=axis)
        self.indices = indices
        self.axis = axis
        self.verbose = verbose

    def transform(self, input, format):
        my_ids = get_ids_for_rank(self.indices)

        if self.verbose and rank == 0:
            t1 = time.time()

        data = numpy.take(input, my_ids, axis=self.axis).astype('f')

        if rank != 0:
            comm.send(data.shape, dest=0, tag=10)
            comm.Send(numpy.ascontiguousarray(data), dest=0, tag=11)
            out = None
        else:
            sh = list(data.shape)
            total = sh[self.axis]
            shapes = [sh, ]
            for rk in range(1, size):
                shp = comm.recv(source=rk, tag=10)
                shapes.append(shp)
                total += shp[self.axis]
            sh[self.axis] = total
            out = numpy.empty(sh, data.dtype)

            start = data.shape[1]
            for rk in range(1, size):
                sh = shapes[rk]
                if sh is None:
                    continue
                empty = numpy.empty(sh, dtype=data.dtype)
                comm.Recv(empty, source=rk, tag=11)
                args = make_slices_args(
                    data.ndim, self.axis, start, start + sh[1])
                out[args] = empty

            if self.verbose and rank == 0:
                t2 = time.time()
                MPIPrint("Time loading single metric: %f" % (t2 - t1))
        return out


class Delta(KoshSimpleNpCache):
    """Computes delta between two consecutive slices over a given axis
    Possibly pads the ends with a value"""

    types = {"numpy": ["numpy", ]}

    def __init__(self,
                 cache_dir=kosh_cache_dir,
                 cache=False,
                 axis=0,
                 pad=None,
                 pad_value=0,
                 verbose=False):

        super(Delta, self).__init__(
            cache_dir, cache, axis=axis, pad=pad, pad_value=pad_value)
        self.axis = axis
        self.pad = pad
        self.pad_value = pad_value
        self.verbose = verbose

    def transform(self, input, format):
        args1 = make_slices_args(input.ndim, self.axis, 0, -1)
        args2 = make_slices_args(input.ndim, self.axis, 1, None)
        delta = input[args2] - input[args1]
        if self.pad == "start":
            sh = list(delta.shape)
            sh[self.axis] = 1
            delta = numpy.concatenate(
                (numpy.ones(sh) * self.pad_value, delta), axis=self.axis)
        elif self.pad == "end":
            sh = list(delta.shape)
            sh[self.axis] = 1
            delta = numpy.concatenate(
                (delta, numpy.ones(sh) * self.pad_value), axis=self.axis)
        elif self.pad is not None:
            raise RuntimeError(
                "Unknown pad value ('{}'), acceptable values are (None, 'start', 'end')".format(
                    self.pad))
        return delta

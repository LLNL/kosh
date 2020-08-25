from .core import KoshTransformer, kosh_cache_dir
import os
import numpy
from .utils import comm, rank, size, get_ids_for_rank, MPIPrint
import time


def make_slices_args(ndims, axis, start, end):
    """given the number of total dimenions return the slice(start, end) at the correct postion for axis
    :param ndims: Total number of dimensions
    :type ndims: int
    :param axis: axis where to position the slice
    :type axis: int
    :param start: start index of the slice we want
    :type start: int
    :param end: end indexof the slice we want
    :type end: int
    :return: list of slice objects to pass to numpy to operate on slice(start, end) on axis
    :rtype: list
    """
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
        """some data to a numpy cache file
        :param cache_file: name of cache file, will be joined with self.cache_dir
        :type cache_file: str
        :param content: content to save to cache
        :type content: object
        """
        cache_file = os.path.join(self.cache_dir, signature)
        numpy.savez(cache_file, *arrays)

    def load(self, signature):
        """loads content from numpy cache
        :param cache_file: name of cache file, will be joined with self.cache_dir
        :type cache_file: str
        :return: data
        :rtpye: object
        """
        cache_file = os.path.join(self.cache_dir, signature) + ".npz"
        npz = numpy.load(cache_file)
        out = [npz[x] for x in npz.files]
        if len(out) == 1:
            out = out[0]
        return out

    def transform(self, input, format):
        """does absolutely nothing but is used as base class to cache a numpy array
        :param input: numpy array(s) to cache
        :type input: ndarray
        :param format: desired format (numpy)
        :type format: str
        :return: same input
        :rtype: ndarray
        """
        return input


class Shuffle(KoshSimpleNpCache):
    """Shuffles data along an axis"""
    types = {"numpy": ["numpy", ]}

    def __init__(self,
                 cache_dir=kosh_cache_dir,
                 cache=False,
                 axis=0,
                 random_state=None,
                 verbose=False):
        """
        :param cache_dir: directory to save cachd files
        :type cache_dir: str
        :param cache: do we use cache?
        :type cache: bool
        :param axis: axis over with to take
        :type axis: int
        :param random_state: random state for reproducibility
                             Controls the randomness of the training and
                             testing indices produced.
                             Pass an int for reproducible output across
                             multiple function calls.
        :type random_state: int
        :param verbose: verbose or not
        :type verbose: bool
        """

        super(Shuffle, self).__init__(
            cache_dir, cache,
            axis=axis,
            random_state=random_state)
        self.axis = axis
        self.random_state = random_state

    def transform(self, input, format):
        """Shuffles data over the transformer's axis
        :param input: array from previous loader or transformer
        :type input: ndarray
        :param format: output format
        :type format: str
        :return: shuffled input over transformer's axis
        :rtype: ndarray
        """

        numpy.random.seed = self.random_state
        return numpy.take(input, numpy.random.permutation(input.shape[self.axis]),
                          axis=self.axis)


class Take(KoshSimpleNpCache):
    """Equivalent of numpy's take, MPI enbabled"""
    types = {"numpy": ["numpy", ]}

    def __init__(self,
                 cache_dir=kosh_cache_dir,
                 cache=False,
                 indices=[],
                 axis=0,
                 verbose=False):
        """
        :param cache_dir: directory to save cachd files
        :type cache_dir: str
        :param cache: do we use cache?
        :type cache: bool
        :param indices: indices to send to take
        :type indices: list
        :param axis: axis over with to take
        :type axis: int
        :param verbose: verbose or not
        :type verbose: bool
        """

        super(Take, self).__init__(
            cache_dir, cache, indices=indices, axis=axis)
        self.indices = indices
        self.axis = axis
        self.verbose = verbose

    def transform(self, input, format):
        """Perform take over transformer's axis and indices
        Can take advantage of MPI if present
        :param input: array from previous loader or transformer
        :type input: ndarray
        :param format: output format
        :type format: str
        :return: input taken over transformer's axis and indices
        :rtype: ndarray
        """
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
        """
        :param cache_dir: directory to save cachd files
        :type cache_dir: str
        :param cache: do we use cache?
        :type cache: bool
        :param axis: axis over with to take
        :type axis: int
        :param pad: Do we pad and i so where? None, "start", "end"
        :type pad: str or None
        :param pad_value: Value to use for padding
        :type pad_value: float
        :param verbose: verbose or not
        :type verbose: bool
        """

        super(Delta, self).__init__(
            cache_dir, cache, axis=axis, pad=pad, pad_value=pad_value)
        self.axis = axis
        self.pad = pad
        self.pad_value = pad_value
        self.verbose = verbose

    def transform(self, input, format):
        """Computes delta between two consecutive slices over a given axis
        Possibly pads the ends with a value
        :param input: array from previous loader or transformer
        :type input: ndarray
        :param format: output format
        :type format: str
        :return: input taken over transformer's axis and indices
        :rtype: ndarray
        """
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

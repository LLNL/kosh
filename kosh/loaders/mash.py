from __future__ import print_function, division
import os
import sys
import numpy
from kosh.arrays import KoshAxis
from .core import KoshLoader
sys.path.append(os.path.expanduser("~/git/mashextract/tools"))  # noqa
import ExtractReader  # noqa
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    hasMpi = True
except ImportError:
    # no mpi
    # we'll fake it
    class Comm():
        def Get_size(self):
            return 1

        def Get_rank(self):
            return 0
    comm = Comm()
    hasMPI = False


class MashReader(object):
    def __init__(self, path):
        """MashReader is a layer on top of MASHExtract Extracter

        :param path: Path to directory where extract was done
        :type path: str
        :raises RuntimeError: Input directory does not exists
        """
        if not os.path.exists(path):
            raise RuntimeError("bad input dir {}".format(path))
        self.reader = ExtractReader.MASHExtractReader(path)
        self.cycles = None
        self.elements = None
        self.metrics_avail = {}
        self.proc_ids = {}
        self.ids = {}
        for elt in ["zone", "node", "corner", "scalarRlxData", "dimRlxData", "srd", "drd"]:
            try:
                metrics_avail, proc_ids = self.__query(elt)
                self.proc_ids[elt] = proc_ids
                self.metrics_avail[elt] = metrics_avail
                ids = []
                for lst in proc_ids:
                    ids += lst
                self.ids[elt] = ids
            except Exception:
                pass

    def __query(self, elt_type):
        """__query Retrieve certain cycle/metrics

        :param elt_type: The elements to query (node/zone/etc..)
        :type elt_type: str
        :raises RuntimeError: Unknown element
        :return: list of metrics available for this element and processors ids
        :rtype: tuple
        """
        # Metrics available
        if elt_type in ["zone", "node", "corner"]:
            metrics_avail = getattr(self.reader, "{}_metrics".format(elt_type))
        elif elt_type == "scalarRlxData":
            metrics_avail = self.reader.scalar_rlx
        elif elt_type == "srd":  # old name
            metrics_avail = self.reader.scalar_rlx
        elif elt_type == "dimRlxData":
            metrics_avail = self.reader.dim_rlx
        elif elt_type == "drd":  # old name
            metrics_avail = self.reader.dim_rlx
        else:
            raise RuntimeError("unknow elt type:", elt_type)

        # map node/zone to processors
        processors = range(self.reader.num_procs)
        proc_ids = []
        n_elements = 0
        for proc in processors:
            if elt_type in ["scalarRlxData", "dimRlxData", "node", "srd", "drd", "corner"]:
                get_proc_ids = "Node"
            else:
                get_proc_ids = "Zone"
            if elt_type == "corner":
                proc_ids.append(list(range(n_elements, n_elements + self.reader.num_proc_corners[proc])))
            else:
                proc_ids.append(
                    getattr(
                        self.reader,
                        "getGlobal{}Ids".format(get_proc_ids))(proc).tolist())
            n_elements += len(proc_ids[-1])
        return metrics_avail, proc_ids

    def get_elements(self, elt_type, cycles=None,
                     elements=None, metrics=None, processors=[]):
        """get_elements fetches desired elements

        :param elt_type: The type of element desired (node, zone, ...)
        :type elt_type: str
        :param cycles: cycles to retrieve, defaults to None which means all
        :type cycles: list, optional
        :param elements: list of elements desired (node ids), defaults to None which means all
        :type elements: list, optional
        :param metrics: metrics to retrieve, defaults to None which means all
        :type metrics: list, optional
        :param processors: processors to retrieve, defaults to [] which means all
        :type processors: list, optional
        :raises RuntimeError: Element not available
        :raises RuntimeError: Metric not available
        :return: array of shape (cycles, elements, metrics)
        :rtype: numpy.ndarray
        """
        metrics_avail = self.metrics_avail[elt_type]
        proc_ids = self.proc_ids[elt_type]
        n_metrics_avail = len(metrics_avail)
        n_elements = len(self.ids[elt_type])

        if cycles is None:
            cycles = self.cycles

        if elements is None:
            elements = self.elements

        # Make sure we have valid elements
        if processors == []:
            processors = range(len(proc_ids))
        if elements is not None:
            for e in elements:
                missing = True
                for proc in processors:
                    if e in proc_ids[proc]:
                        missing = False
                if missing:
                    raise RuntimeError(
                        "Missing {} element # {}".format(
                            elt_type, e))
            n_elements = len(elements)

        kargs_core = {}
        if cycles is not None:
            kargs_core["cycles"] = cycles
            n_cycles = len(cycles)
        else:
            n_cycles = self.reader.num_cycles

        metrics_indices = []
        if metrics is not None:
            for m in metrics:
                if m not in metrics_avail:
                    raise RuntimeError("Metrics {} not available".format(m))
                else:
                    metrics_indices.append(metrics_avail.index(m))
        else:
            metrics = metrics_avail

        data = None
        retrieved_elements = []
        # ok let's split along available processor
        size = comm.Get_size()
        rank = comm.Get_rank()
        slices = len(processors) // size
        if len(processors) % size != 0:
            slices += 1
        for proc in processors[rank*slices:min((rank+1)*slices, len(processors))]:
            kargs = kargs_core.copy()
            if proc == slices:
                print("Rank {} reading {} space: ({} ->  {})".format(rank,
                                                                     proc, rank * slices,
                                                                     min((rank+1)*slices, len(processors))))
                sys.stdout.flush()
            if "elements" in kargs:
                del(kargs["elements"])
            if elements is not None:
                # Not all elements are on a processor
                elt = []
                for e in elements:
                    if e in proc_ids[proc]:
                        elt.append(proc_ids[proc].index(e))
                n_elements = len(elt)
                if n_elements == 0:
                    continue
                retrieved_elements += elt
                kargs["elements"] = elt
                if cycles is None:  # need to create cycles
                    cycles = self.getStateVariables()["cycle"]
                    kargs["cycles"] = cycles
            else:
                n_elements = len(proc_ids[proc])
            if "metrics" in kargs:
                n_metrics_avail == len(metrics)
            # Final shape for one processor
            if elt_type in ["zone", "node", "scalarRlxData", "srd", "corner"]:
                sh = [n_cycles, n_elements, n_metrics_avail]
            elif elt_type in ["dimRlxData", "drd"]:
                sh = [n_cycles, n_elements, 2, n_metrics_avail]
            if elt_type in ["zone", "node", "corner"]:
                use_ext = "{} metric".format(elt_type)
            elif elt_type == "scalarRlxData":
                use_ext = "scalar rlx"
            elif elt_type == "srd":
                use_ext = "srd"
            elif elt_type == "dimRlxData":
                use_ext = "dim rlx"
            elif elt_type == "drd":
                use_ext = "drd"
            else:
                use_ext = elt_type
            if cycles is not None and len(
                    cycles) == self.reader.num_cycles and "cycles" in kargs:
                del(kargs["cycles"])
            elt = kargs.get("elements", [])
            if len(elt) == len(proc_ids[proc]) and sorted(elt) == elt and "elements" in kargs:
                del(kargs["elements"])
            # if len(kargs.keys()) == 1 and "cycles" in kargs:
            #    # right now passing just cycles is not implemented yet
            #    kargs["elements"] = list(range(len(proc_ids[proc])))
            if len(metrics_indices) != 0 and len(kargs) == 0:
                kargs["metrics"] = metrics
                metrics_indices = range(len(metrics_indices))
                sh[-1] = len(metrics)
            tmp = self.reader.request(ext_type=use_ext,
                                      proc=proc,
                                      **kargs)
            tmp.shape = sh
            if len(metrics_indices) != 0:
                # We need to return only the metrics wanted
                out = None
                for indx in metrics_indices:
                    if out is None:
                        out = tmp[..., indx:indx + 1]
                    else:
                        out = numpy.concatenate(
                            (out, tmp[..., indx:indx + 1]), axis=-1)
                tmp = out
            if data is None:  # fist time
                data = tmp
            else:
                data = numpy.concatenate((data, tmp), axis=1)
        return data

    get = get_elements

    def gather_mpi_processors(self, data):
        """After a get was issued accross multiple processor, this function gathers them all on rk 0"""
        size = comm.Get_size()
        rank = comm.Get_rank()
        if rank != 0:
            # we need to send the sahpe so we can prepare the receive on rk 0
            if data is not None:
                print("sending array of shape", data.shape, "and type:", data.dtype, "from rank:", rank)
                sys.stdout.flush()
                comm.send(data.shape, dest=0, tag=10)
                comm.Send(data, dest=0, tag=11)
            else:
                print("Rk:", rank, "Sending back None")
                sys.stdout.flush()
                comm.send(data, dest=0, tag=10)
        else:
            sh = list(data.shape)
            shapes = [sh, ]
            total = sh[1]
            for rk in range(1, size):
                shp = comm.recv(source=rk, tag=10)
                print("Received", shp, "from rank", rk)
                sys.stdout.flush()
                shapes.append(shp)
                if shp is not None:
                    total += shp[1]
            # We are on first proc let's concatenenate all
            sh[1] = total
            out = numpy.empty(sh, data.dtype)
            out[:, :data.shape[1]] = data[:]
            sys.stdout.flush()
            start = data.shape[1]
            for rk in range(1, size):
                sh = shapes[rk]
                if sh is None:
                    continue
                empty = numpy.empty(sh, dtype=data.dtype)
                comm.Recv(empty, source=rk, tag=11)
                out[:, start:start+sh[1]] = empty
        if rank == 0:
            return out

    def getStateVariables(self):
        """getStateVariables return a dictionary of all state variables
        usually cycles and time

        return: dictionary containing var:array
        rtype: dict
        """
        state = os.path.join(self.reader.ext_path, "state.bin")
        data = numpy.fromfile(state, dtype=self.reader.state_dtype)
        state_vars = {}
        nvars = len(self.reader.state_vars)
        for i, v in enumerate(self.reader.state_vars):
            state_vars[v] = data[i::nvars]
        return state_vars

    def getAxis(self, axis, elt):
        """getAxis get an axis (dimension info) for an element

        :param axis: name of axis to retrieve
        :type axis: str
        :param elt: element type for which axis is requested
        :type elt: str
        :raises RuntimeError: [description]
        :raises RuntimeError: [description]
        :return: axis
        :rtype: KoshAxis
        """
        good_axes = ["cycles", "elements", "metrics", "direction"]
        if axis not in good_axes:
            raise RuntimeError(
                "Invalid axis {}, available axes are: {}".format(
                    axis, good_axes))
        if axis == "cycles":
            return KoshAxis(axis, self.getStateVariables()["cycle"])
        elif axis == "elements":
            return KoshAxis(axis, self.ids[elt])
        elif axis == "metrics":
            return KoshAxis(axis, self.metrics_avail[elt])
        elif axis == "direction" and elt == "dimRlxData":
            return KoshAxis(axis, [0, 1])
        else:
            raise RuntimeError(
                "Invalid axis {} for element type {}".format(
                    axis, elt))

    def getAxisList(self, elt):
        """getAxisList returns all axes information for an element type

        :param elt: element type (node, zone, ...)
        :type elt: str
        :return: list of Kosh axes
        :rtype: type
        """
        axes_ids = ["cycles", "elements", "metrics"]
        if elt in ["dimRlxData"]:
            axes_ids.insert(-1, "direction")
        axes = []
        for axis in axes_ids:
            axes.append(self.getAxis(axis, elt))
        return axes

    def get(self, feature, *args, **kargs):
        """get a metric out of the reader

        :param feature: element/metric to retrieve
        :type feature: str
        :return: data
        :rtype: numpy.ndarray
        """
        sp = feature.split("/")
        if len(sp) == 1:
            elt = sp[0]
            feature = None
        else:
            elt, feature = sp[:2]
        return self.get_elements(elt, metrics=[feature, ], *args, **kargs)


class MashLoader(KoshLoader):
    types = {"mash": ["numpy", ]}

    def __init__(self, obj):
        """MashLoader for Kosh to be able to read in MASHExtract files

        :param KoshLoader: Kosh loaders base class
        :type KoshLoader: KoshLoader
        :param obj: Kosh obj reference
        """
        super(MashLoader, self).__init__(obj)

    def open(self, mode="r"):
        """open the mash reader

        :return: MashReader
        :rtype: MashReader
        """
        return MashReader(str(self.obj.uri))

    def extract(self):
        """get a feature

        feature and format come from "self"
        :return: numpy array
        :rtype: numpy.ndarray
        """
        args, kargs = self._user_passed_parameters
        reader = self.open()
        return reader.get(self.feature, *args, **kargs)

    def list_features(self):
        """list_features lists features available

        :return: list of features you can retrieve
        :rtype: list
        """
        reader = self.open()
        out = []
        for elt in ["zone", "node", "scalarRlxData", "dimRlxData", "corner"]:
            try:
                metrics_avail = reader.metrics_avail[elt]
                for m in metrics_avail:
                    out.append("{}/{}".format(elt, m))
            except Exception:
                pass
        return out

    def describe_feature(self, feature):
        """describe a feature

        :param feature: feature (variable) to read, defaults to None
        :type feature: str, optional if loader does not require this
        :return: dictionary describing the feature
        :rtype: dict
        """
        if feature not in self.list_features():
            raise ValueError("feature {feature} is not available".format(feature=feature))
        reader = self.open()
        sp = feature.split("/")
        axes = reader.getAxisList(sp[0])
        sh = []
        dims = []
        info = {"format": "mash"}
        for ax in axes[:-1]:  # last one is the feature
            specs = {}
            sh.append(len(ax))
            specs["name"] = ax.id
            specs["length"] = len(ax)
            specs["first"] = ax[0]
            specs["last"] = ax[-1]
            dims.append(specs)
        info["dimensions"] = dims
        info["size"] = sh
        return info

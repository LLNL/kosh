import sys
sys.path.append("/g/g19/cdoutrix/git/mashextract/tools")  # noqa
import ExtractReader
import os
import numpy
from collections import OrderedDict

class Axis(object):
    def __init__(self, id, values):
        self.id = id
        self.__values = values
    def __getitem__(self, key):
        return self.__values[key]
    def __setitem__(self, key, value):
        self.__values[key] = value
    def __len__(self):
        return len(self.__values)
    
class KullReader(object):
    def __init__(self, path):
        if not os.path.exists(path):
            raise RuntimeError("bad input dir {}".format(path))
        self.reader = ExtractReader.MASHExtractReader(path)
        self.cycles = None
        self.elements = None
        self.metrics_avail = {}
        self.proc_ids = {}
        self.ids = {}
        for elt in ["zone", "node", "srd", "drd"]:
            metrics_avail, proc_ids = self.query(elt)
            self.proc_ids[elt] = proc_ids
            self.metrics_avail[elt] = metrics_avail 
            ids = []
            for lst in proc_ids:
                ids += lst
            self.ids[elt] = ids

    def query(self, elt_type):
        """Retrieve certain cycle/metrics 
        """
        # Metrics available
        if elt_type in ["zone", "node"]:
            key = "zone"  # for now because of bug in reader
            metrics_avail = getattr(self.reader,"{}_metrics".format(key))
            print("Metrics avail:", elt_type, metrics_avail, len(metrics_avail))
        elif elt_type == "srd":
            metrics_avail = self.reader.SRD
        elif elt_type == "drd":
            metrics_avail = self.reader.DRD
        else:
            raise RuntimeError("unknow elt type:", elt_type)

        # map node/zone to processors
        processors = range(self.reader.num_procs)
        proc_ids = []
        n_elements = 0
        for proc in processors:
            if elt_type in ["srd", "drd", "node"]:
                get_proc_ids = "Node"
            else:
                get_proc_ids = "Zone"
            proc_ids.append(getattr(self.reader, "getGlobal{}Ids".format(get_proc_ids))(proc).tolist())
            n_elements += len(proc_ids[-1])
        return metrics_avail, proc_ids

    def get_elements(self, elt_type, cycles=None, elements=None, metrics=None, processors=[]):
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
                    raise RuntimeError("Missing {} element # {}".format(node_or_zone, e))
            n_elements = len(elements)

        kargs = {}
        if cycles is not None:
            kargs["cycles"] = cycles
            n_cycles = len(cycles)
            #if elements is None:  # We need both cycle and elements at the moment....
            #    elements = []
            #    for ids in processors:
            #        elements += proc_ids[ids]
        else:
            n_cycles = self.reader.num_cycles

        metrics_indices = []
        if metrics is not None:
            n_metrics = len(metrics)
            for m in metrics:
                if m not in metrics_avail:
                    raise RuntimeError("Metrics {} not available".format(m))
                else:
                    metrics_indices.append(metrics_avail.index(m))
        else:
            n_metrics = n_metrics_avail
            metrics = metrics_avail
            

        data = None
        retrieved_elements = []
        for proc in processors:
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
                if cycles is None: # need to create cycles
                    cycles = list(range(self.reader.num_cycles))
                    kargs["cycles"] = cycles
            else:
                n_elements = len(proc_ids[proc])
            # Final shape for one processor
            if elt_type in ["zone", "node", "srd"]:
                sh = (n_cycles, n_elements, n_metrics_avail)
            elif elt_type == "drd":
                sh = (n_cycles, n_elements, 2, n_metrics_avail)
            if elt_type in ["zone", "node"]:
                use_ext = "{} metric".format(elt_type)
            else:
                use_ext = elt_type
            if len(cycles) == self.reader.num_cycles and "cycles" in kargs:
                del(kargs["cycles"])
            elt = kargs.get("elements", [])
            if len(elt) == len(proc_ids[proc]) and sorted(elt) == elt:
                del(kargs["elements"])
            print("KARGS:", list(kargs.keys()), proc, use_ext)
            tmp = self.reader.request(ext_type=use_ext,
                                      proc=proc,
                                      **kargs)
            tmp.shape=sh
            print("TMP:", tmp.dtype, tmp.shape)
            if data is None:  # fist time
                data = tmp
            else:
                data = numpy.concatenate((data, tmp), axis=1)
        if len(metrics_indices) !=0:
            # We need to return only the metrics wanted
            out = None
            for indx in metrics_indices:
                if out is None:
                    out = data[...,indx:indx+1]
                else:
                    out = numpy.concatenate((out, data[...,indx:indx+1]))
            data = out
        return data

    def getAxis(self, axis, elt):
        good_axes = ["cycles", "elements", "metrics", "direction"]
        if not axis in good_axes:
            raise RuntimeError("Invalid axis {}, available axes are: {}".format(axis, good_axes))
        if axis == "cycles":
            return Axis(axis, list(range(self.reader.num_cycles)))
        elif axis == "elements":
            return Axis(axis, self.ids[elt])
        elif axis == "metrics":
            return Axis(axis, self.metrics_avail[elt])
        elif axis == "direction" and elt == "drd":
            return Axis(axis, [0,1])
        else:
            raise RuntimeError("Invalid axis {} for element type {}".format(axis, elt))

    def getAxisList(self, elt):
        axes_ids = ["cycles", "elements", "metrics"]
        if elt in ["drd"]:
            axes_ids.insert(-1, "direction")
        axes = []
        for axis in axes_ids:
            axes.append(self.getAxis(axis, elt))
        return axes


import h5py
import re
from .core import KoshLoader
import numpy


def walk_hdf5(d, prefix=""):
    """Walk through hdf5 groups to find all datsets and return their paths
    return generator
    """
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, h5py._hl.dataset.Dataset):
            yield prefix+"/"+k+"***"
        else:
            if prefix == "":
                yield "/".join(walk_hdf5(v, prefix=k))
            else:
                yield "/".join(walk_hdf5(v, prefix=prefix+"/"+k))


def list_hdf5(obj):
    """walk hdf5 and return list of path to all datasets
    """
    nest = list(walk_hdf5(obj))
    out = []
    for l in nest:
        for d in l.split("***"):
            if len(d) > 0:
                if d[0] == "/":
                    out.append(d[1:])
                else:
                    out.append(d)
    return out


class KoshHDF5Loader(KoshLoader):
    types = {"hdf5": ["numpy", ]}

    def __init__(self, obj):
        self._restart_features = None
        self._no_restart_features = None
        super(KoshHDF5Loader, self).__init__(obj)
        self._restart_features = self.list_features(restarts=True)
        self._no_restart_features = self.list_features(restarts=False)

    def open(self, mode='r'):
        """open/load  matching Kosh SIna File

        :param mode: mode to open the file in, defaults to 'r'
        :type mode: str, optional
        :return: Kosh File object
        """
        return h5py.File(self.obj.uri, mode)

    def preprocess(self):
        """Preprocess to indentify desired feature and possible restart
        """
        args, kargs = self._user_passed_parameters
        feat = self.feature
        if feat[-9:] == "restarts)":
            # Let's remove the comment
            feat = " ".join(feat.split(" ")[:-2])
        restart = kargs.pop("restart", None)
        if restart is not None:
            if f"{restart:03d}/" not in feat:
                feat = feat.split("/")
                if len(feat) > 1:
                    feat = "/".join([feat[0], f"{restart:03d}"] + feat[1:])
                else:
                    feat = f"{restart:03d}/{feat[0]}"

        self.feature = feat

    def get(self, feature=None, format=None, Id=None, loader=None, *args, **kargs):
        """get extracts a feature from an hdf5 file

        This loader can handle df5 files with custom restart in them.
        pass `restart=N` to extract the Nth cycle

        w/o arguments only the restarted cycles will be extracted,
        if a slice is passed to the `cycles` keyword then the loader will extract
        all the necessary cycles backward from the desired restart up to the original restart (000)
        
        :param feature: feature to extract, defaults to None which means all features
        :type feature: str, optional
        :param format: output format
        :type format: str, optional
        :param Id: dataset Id, defaults to None
        :type Id: str, optional
        :param loader: loader to use, defaults to None
        :return: extracted feature
        :rtype: numpy.ndarray
        """
        if feature not in self.list_features() and feature in self._restart_features:
            feature = " ".join(feature.split(" ")[:-2])
        return super(KoshHDF5Loader, self).get(feature, format, Id, loader, *args, **kargs)

    def extract(self):
        """extract return a feature from the loaded object.

        :param feature: variable to read from file
        :type feature: str
        :param format: desired output format
        :type format: str
        :param restart: desired restart (if applicable)
        :type restart: int or None
        :return: data
        """
        args, kargs = self._user_passed_parameters
        if "restart" in kargs:
            kargs.pop("restart")  # it's for preprocess
        f = h5py.File(self.obj.uri, "r")
        feat = f[self.feature]
        if len(kargs) != 0:  # probably requested dims
            feat_dims = [x.label for x in feat.dims]
            if feat_dims == [""]:
                # Probably a dimension (cycle?)
                feat_dims = [self.feature, ]
            user_dims = {}
            for k in list(kargs.keys()):
                if k in feat_dims:
                    user_dims[k] = kargs.pop(k)
            select = {}
            for dim in user_dims:
                user_selection = user_dims[dim]
                if isinstance(user_selection, slice):
                    indices = user_selection
                else:  # User passed a value or values
                    values = f[dim][:].tolist()
                    indices = [values.index(x) for x in user_selection]
                select[feat_dims.index(dim)] = indices
            selectors = []
            for i in range(len(feat.shape)):
                if i in select:
                    selectors.append(select[i])
                else:
                    selectors.append(slice(0, None))
            if "cycles" in kargs:
                # ok let's make sure it's not a restart!
                restart = re.search("/\d\d\d/", self.feature)  # noqa
                if restart is None:  # we need to search dims as well
                    restart = re.search("\d\d\d/", self.feature)  # noqa
                if restart is not None:
                    # Ok it's a restart we need to match cycles/restart file
                    my_restart = restart.group()
                    if my_restart[0] != "/":
                        my_restart = f"/{my_restart}"
                    restarts = {}
                    restart = int(my_restart[1:-1])
                    cycles = f[f"{my_restart}/cycles"]
                    restarts[restart] = {"first": cycles[0],
                                         "cycles": cycles[:],
                                         "feature": self.feature}
                    last_valid_restart = restart
                    restart -= 1
                    while restart > 0:
                        feature = self.feature.replace(
                            my_restart, f"/{restart:03d}/")
                        cycles = f[f"{restart:03d}/cycles"]
                        if cycles[0] < restarts[last_valid_restart]["first"]:
                            restarts[restart] = {"first": cycles[0],
                                                 "cycles": cycles[:],
                                                 "feature": feature}
                            last_valid_restart = restart
                        restart -= 1
                    # Original run
                    cycles = f["cycles"]
                    fnm = self.feature if "cycles" not in self.feature else "cycles"
                    restarts[0] = {"first": cycles[0],
                                   "cycles": cycles[:],
                                   "feature": fnm}
                    keys = sorted(restarts.keys())
                    cycles = numpy.array(())
                    start_indx = 0
                    for indx, key in enumerate(keys[:-1]):
                        last = int(numpy.argwhere(restarts[key]["cycles"] == restarts[keys[indx+1]]["first"])[0])
                        restarts[key]["indices"] = (start_indx, last + start_indx)
                        start_indx += last
                        cycles = numpy.concatenate((cycles, restarts[key]["cycles"][:last]))
                    cycles = numpy.concatenate(
                        (cycles, restarts[keys[-1]]["cycles"]))
                    restarts[keys[-1]]["indices"] = (start_indx, len(cycles))
                    user_cycles = kargs["cycles"]
                    if not isinstance(user_cycles, slice):
                        # User wants a value range but we want indices
                        start = int(numpy.argwhere(
                            cycles == user_cycles[0])[0])
                        stop = int(numpy.argwhere(
                            cycles == user_cycles[-1])[0]) + 1
                    else:
                        step = user_cycles.step
                        if step is None:
                            step = 1
                        start = user_cycles.start
                        if start is None:
                            start = 0
                            if step < 0:
                                start = len(cycles)
                        elif start < 0:
                            start = len(cycles) + start
                        stop = user_cycles.stop
                        if stop is None:
                            stop = len(cycles)
                            if step < 0:
                                stop = None
                        elif stop < 0:
                            stop = len(cycles) + stop
                    flip = False
                    if stop is None or start > stop:
                        flip = True
                        tmp = cycles[start:stop:step]
                        start = int(numpy.argwhere(cycles == tmp[-1])[0])
                        stop = int(numpy.argwhere(cycles == tmp[0])[0]) + 1
                        step = -step

                    user_cycles = slice(start, stop, step)
                    # Ok at this point we have a slice selection

                    start = current_start = user_cycles.start
                    for cycles_index, fd in enumerate(feat_dims):
                        if fd[-6:] == "cycles":
                            break
                    feat = None
                    for key in sorted(keys):
                        rs = restarts[key]
                        range_key = rs["indices"]
                        if range_key[0] <= current_start < range_key[1]:
                            # Ok we have data intersection
                            if user_cycles.stop >= range_key[1]:
                                # goes past this restart
                                stop = range_key[1] - range_key[0]
                            else:
                                # We are done here
                                stop = user_cycles.stop - range_key[0]
                            start = current_start - range_key[0]
                            cycle_slice = slice(start, stop, user_cycles.step)
                            selectors[cycles_index] = cycle_slice
                            if feat is None:
                                feat = f[rs["feature"]][tuple(selectors)]
                            else:
                                feat = numpy.concatenate(
                                    (feat, f[rs["feature"]][tuple(selectors)]), axis=cycles_index)
                            j = 0
                            while current_start + j*step < range_key[1]:
                                j += 1
                            current_start += j*step
                # Reversed order
                if flip:
                    selectors = []
                    for j in range(len(feat.shape)):
                        if j == cycles_index:
                            selectors += [slice(None, None, -1), ]
                        else:
                            selectors += [slice(0, None)]
                    feat = feat[tuple(selectors)]
            else:
                feat = feat[tuple(selectors)]
        return feat

    def list_features(self, restarts=False, **kargs):
        """list_features list features in file,
        for hdf5 you can pass extra argument to navigate groups.

        :return: list of features available in file
        :rtype: list
        """
        if restarts and self._restart_features is not None:
            # Saves time it's already done
            return self._restart_features
        if not restarts and self._no_restart_features is not None:
            # Saves time again
            return self._no_restart_features

        with h5py.File(self.obj.uri, "r") as f:
            features = list_hdf5(f)
        if restarts:
            # Ok we want a cleaner output just mentioning how many restarts
            n = 0
            feat = []  # Features to return (w/o restart)
            restart_features = set()  # Features that have a restart
            for f in features:
                s = re.search("/\d\d\d/", f)  # noqa
                s2 = re.search("\d\d\d/", f)  # noqa
                if s is not None:
                    # We found a restart
                    st = s.group()
                    n = max(n, int(st[1:-1]))
                    restart_features.add(f.replace(st, "/"))
                elif s2 is not None:
                    st = s2.group()
                    restart_features.add(f.replace(st, ""))
                else:
                    feat.append(f)
            for dup in restart_features:
                try:
                    indx = feat.index(dup)
                    feat[indx] = feat[indx] + f" ({n} restarts)"
                except Exception:
                    # weird case when original run neg node relaxer
                    feat.append(dup+f" ({n} restarts but not used on original set)")
            features = feat
        return features

    def describe_feature(self, feature):
        """describe a feature

        :param feature: feature (variable) to read, defaults to None
        :type feature: str, optional if loader does not require this
        :return: dictionary describing the feature
        :rtype: dict
        """
        features = self.list_features()
        if feature not in features and feature not in self.list_features(restarts=True):
            raise ValueError(f"feature {feature} is not available")

        info = {}
        with h5py.File(self.obj.uri, "r") as f:
            try:
                feature = f[feature]
            except Exception:
                sp = feature.split(" ")
                feature = " ".join(sp[:-2])  # trying to remove the restart comment
                feature = f[feature]
                info["restarts"] = int(sp[-2][1:])
            info["size"] = feature.shape
            info["format"] = "hdf5"
            info["type"] = feature.dtype
            if hasattr(feature, "dims"):
                dims = []
                for d in feature.dims.keys():
                    specs = {}
                    specs["name"] = d.label
                    try:
                        specs["first"] = f[d.label][0]
                        specs["last"] = f[d.label][-1]
                        specs["length"] = len(f[d.label])
                    except Exception:
                        pass
                    dims.append(specs)
                info["dimensions"] = dims
        return info

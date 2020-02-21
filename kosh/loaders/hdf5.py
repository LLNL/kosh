import h5py
import re
from .core import KoshLoader


def walk_hdf5(d, prefix=""):
    """Walk through hdf5 groups to find all datsets and return their paths
    return generator
    """
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, h5py._hl.dataset.Dataset):
            yield prefix+"/"+k+"***"
        else:
            if prefix=="":
                yield "/".join(walk_hdf5(v, prefix=k))
            else:
                yield "/".join(walk_hdf5(v, prefix=prefix+"/"+k))

def list_hdf5(obj):
    """walk hdf5 and return list of path to all datasets
    """
    nest = list(walk_hdf5(obj))
    out =[]
    for l in nest:
        for d in l.split("***"):
            if len(d)>0:
                if d[0] == "/":
                    out.append(d[1:])
                else:
                    out.append(d)
    return out


class KoshHDF5Loader(KoshLoader):
    types = {"hdf5": ["numpy", ]}

    def __init__(self, obj):
        super(KoshHDF5Loader, self).__init__(obj)

    def open(self, mode='r'):
        """open/load the matching Kosh SIna File

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
            if f"/{restart:03d}/" not in feat:
                feat = feat.split("/")
                feat = "/".join([feat[0], f"{restart:03d}"] + feat[1:])
        self.feature = feat

    def get(self, feature=None, format=None, Id=None, loader=None, *args, **kargs):
        if feature not in self.list_features() and feature in self.list_features(restarts=True):
            feature = " ".join(feature.split(" ")[:-2])
        super(KoshHDF5Loader, self).get(feature, format, Id, loader, *args, **kargs)

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
        print("WANT TO READ:", self.feature)
        feat = f[self.feature]
        if len(kargs) != 0:  # probably requested dims
            feat_dims = [x.label for x in feat.dims]
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
            feat = feat[tuple(selectors)]
        return feat

    def list_features(self, restarts=False, **kargs):
        """list_features list features in file,
        for hdf5 you can pass extra argument to navigate groups.

        :return: list of features available in file
        :rtype: list
        """
        with h5py.File(self.obj.uri, "r") as f:
            features = list_hdf5(f)
        if restarts:
            # Ok we want a cleaner output just mentioning how many restarts
            n = 0
            feat = []  # Features to return (w/o restart)
            restart_features = set()  # Features that have a restart
            for f in features:
                s = re.search("/\d\d\d/", f)
                s2 = re.search("\d\d\d/", f)
                if s is not None:
                    # We found a restart
                    st = s.group()
                    n = max(n, int(st[1:-1]))
                    restart_features.add(f.replace(st,"/"))
                elif s2 is not None:
                    st = s2.group()
                    restart_features.add(f.replace(st,""))
                else:
                    feat.append(f)
            for dup in restart_features:
                try:
                    indx = feat.index(dup)
                    feat[indx] = feat[indx] + f" ({n} restarts)"
                except:
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

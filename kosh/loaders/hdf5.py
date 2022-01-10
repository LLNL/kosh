from __future__ import print_function, division
from .core import KoshLoader


def walk_hdf5(dataset, prefix=""):
    """Walk through hdf5 groups to find all datasets and return their paths
    return generator
    :param dataset: hdf5 dataset to start walking from
    :type dataset: h5py._hl.dataset.Dataset
    :param prefix: prefix to use when walking hdf5 paths
    :return: hdf5 dataset structure
    :rtype: generator
    """
    import h5py
    for key in sorted(dataset.keys()):
        value = dataset[key]
        if isinstance(value, h5py._hl.dataset.Dataset):
            yield prefix+"/"+key+"***"
        else:
            if prefix == "":
                yield "/".join(walk_hdf5(value, prefix=key))
            else:
                yield "/".join(walk_hdf5(value, prefix=prefix+"/"+key))


def list_hdf5(dataset):
    """walk hdf5 and return list of path to all datasets
    :param dataset: hdf5 dataset to start walking from
    :type dataset: h5py._hl.dataset.Dataset
    :return: hdf5 dataset structure
    :rtype: list
    """
    nest = list(walk_hdf5(dataset))
    out = []
    for p in nest:
        for d in p.split("***"):
            if len(d) > 0:
                # Removes leading /
                while d[0] == "/":
                    d = d[1:]
                out.append(d)
    return out


class HDF5Loader(KoshLoader):
    """ Kosh loader to load HDF5 data"""
    types = {"hdf5": ["numpy", ]}

    def open(self, mode='r'):
        """open/load  matching Kosh Sina File

        :param mode: mode to open the file in, defaults to 'r'
        :type mode: str, optional
        :return: Kosh File object
        """
        import h5py
        return h5py.File(self.obj.uri, mode)

    def extract(self):
        """extract return a feature from the loaded object.

        :param feature: variable to read from file
        :type feature: str
        :param format: desired output format
        :type format: str
        :return: data
        """
        import h5py
        args, kargs = self._user_passed_parameters
        f = h5py.File(self.uri, "r")
        features = self.feature
        if not isinstance(features, list):
            features = [self.feature, ]

        out = []
        for feature in features:
            feat = f[feature]
            if len(kargs) != 0:  # probably requested dims
                feat_dims = [x.label for x in feat.dims]
                if feat_dims == [""]:
                    # Probably a dimension (cycle?)
                    feat_dims = [feature, ]
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
            out.append(feat)
        if not isinstance(self.feature, list):
            return out[0]
        else:
            return out

    def list_features(self, group=None, **kargs):
        """list_features list features in file,
        for hdf5 you can pass extra argument to navigate groups.

        :return: list of features available in file
        :rtype: list
        """
        import h5py
        with h5py.File(self.uri, "r") as f:
            features = list_hdf5(f)
        if group is not None:
            feats = []
            for f in features:
                if f[:len(group)] == group:
                    feats.append(f[len(group)+1:])
            features = feats
        # At this point we need to allow to select a group itself
        feats = set()
        for feature in features:
            sp = feature.split("/")
            for i in range(len(sp)):
                feats.add("/".join(sp[:i+1]))
        features = sorted(feats)
        return features

    def describe_feature(self, feature):
        """describe a feature

        :param feature: feature (variable) to read, defaults to None
        :type feature: str, optional if loader does not require this
        :return: dictionary describing the feature
        :rtype: dict
        """
        import h5py
        features = self._list_features()
        if feature not in features:
            raise ValueError("feature {feature} is not available".format(feature=feature))

        info = {}
        with h5py.File(self.uri, "r") as f:
            feature = f[feature]
            info["size"] = feature.shape
            info["format"] = "hdf5"
            info["type"] = feature.dtype
            if hasattr(feature, "dims"):
                dims = []
                for d in feature.dims:
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

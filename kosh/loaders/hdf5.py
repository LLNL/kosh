import h5py
from .core import KoshLoader


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

    def extract(self, feature, format):
        """extract return a feature from the loaded object.

        :param feature: variable to read from file
        :type feature: str
        :param format: desired output format
        :type format: str
        :return: data
        """
        args, kargs = self._user_passed_parameters
        f = h5py.File(self.obj.uri, "r")
        feat = f[feature]
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

    def list_features(self, *args):
        """list_features list features in file,
        for hdf5 you can pass extra argument to navigate groups.

        :return: list of features available in file
        :rtype: list
        """
        with h5py.File(self.obj.uri, "r") as f:
            keys = []
            if len(args) == 0:
                for k in f.keys():
                    if hasattr(f[k], "keys"):
                        for k2 in f[k].keys():
                            keys.append(f"{k}/{k2}")
                    else:
                        keys.append(k)
                return keys
            else:
                return list(f[args[0]].keys())

    def describe_feature(self, feature):
        """describe a feature

        :param feature: feature (variable) to read, defaults to None
        :type feature: str, optional if loader does not require this
        :return: dictionary describing the feature
        :rtype: dict
        """
        if feature not in self.list_features():
            raise ValueError(f"feature {feature} is not available")
        info = {}
        with h5py.File(self.obj.uri, "r") as f:
            feature = f[feature]
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

# Ultra files loader contributed by Josh Kallman 5/6/2020
from .core import KoshLoader
import sys
sys.path.append("/usr/gapps/pydv/current")  # noqa
try:
    import pydvpy as pydvif
except ImportError:
    import pydv
    sys.path.append(pydv.__path__[0])
    import pydv.pydvpy as pydvif


class UltraLoader(KoshLoader):
    types = {"ultra": ["numpy", ]}

    def load_from_ultra(self, variable):
        if not isinstance(variable, (list, tuple)):  # only one variable requested
            variable = [variable, ]

        variables = [{}, ] * len(variable)

        # curve.x is time, curve.y is data
        for c in self.curves:
            name = c.name.split()[0]
            if name in variable:
                variables[variable.index(name)]['time'] = c.x
                variables[variable.index(name)]['val'] = c.y

        if len(variables) > 1:
            return variables
        else:  # only one variable read in
            return variables[0]

    def __init__(self, obj):
        super(UltraLoader, self).__init__(obj)
        self.curves = pydvif.read(self.obj.uri)

    def extract(self, *args, **kargs):
        return self.load_from_ultra(self.feature)

    def list_features(self):
        variables = []
        for curve in self.curves:
            variables.append(curve.name.split()[0])
        return variables

    def describe_feature(self, feature):
        info = {"name": feature}
        for c in self.curves:
            if c.name.split()[0] == feature:
                info["size"] = len(c.x)
                info["first_time"] = c.x[0]
                info["last_time"] = c.x[-1]
                info["min"] = min(c.y)
                info["max"] = max(c.y)
                info["type"] = c.y.dtype
        return info

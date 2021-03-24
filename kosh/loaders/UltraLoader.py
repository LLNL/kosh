# Ultra files loader contributed by Josh Kallman 5/6/2020
from .core import KoshLoader
import sys
import os
sys.path.append("/usr/gapps/pydv/current")  # noqa
# pydv import matplotlib.pyplot
# on some systems with no X forwarding this causes
# an uncatchable error.
# Setting the matplotlib backend to a windowless
# backend fixes this.
if "DISPLAY" not in os.environ or os.environ["DISPLAY"] == "":
    import matplotlib
    matplotlib.use("agg", force=True)
try:
    import pydvpy as pydvif
except ImportError:
    import pydv
    sys.path.append(pydv.__path__[0])
    import pydv.pydvpy as pydvif


class UltraLoader(KoshLoader):
    """Kosh Loader for ultra files"""
    types = {"ultra": ["numpy", ]}

    def __init__(self, obj):
        super(UltraLoader, self).__init__(obj)
        self.curves = pydvif.read(self.obj.uri)

    def load_from_ultra(self, variable):
        """Load variables from an ultra file
        :param variable: variables to load
        :type variable: list or str
        :return list of dictionary conatining 'time and 'val' for each variable
        :rtype: list of dict or dict
        """
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

    def extract(self, *args, **kargs):
        """Extract a feature"""
        return self.load_from_ultra(self.feature)

    def list_features(self):
        """List features available in ultra file"""
        variables = []
        for curve in self.curves:
            variables.append(curve.name.split()[0])
        return variables

    def describe_feature(self, feature):
        """Describe a feature

        :param feature: feature to describe
        :type feature: str
        :return: dictionary with attributes describing the feature:
                 'name', 'size', 'first_time', 'last_time', 'min', 'max', 'type'
        :rtype: dict
        """
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

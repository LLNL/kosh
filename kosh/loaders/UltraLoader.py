# Ultra files loader contributed by Josh Kallman 5/6/2020
from .core import KoshLoader
import sys
import os
import numpy
sys.path.append("/usr/gapps/pydv/current")  # noqa


class UltraLoader(KoshLoader):
    """Kosh Loader for ultra files"""
    types = {"ultra": ["curves", "curves/pydv", "curves/pdv", "pydv", "pdv",
                       "dict", "numpy"]}

    def __init__(self, obj, **kargs):
        super(UltraLoader, self).__init__(obj, **kargs)
        self.curves = None

    def load_curves(self):
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
        self.curves = pydvif.read(self.uri)

    def load_from_ultra(self, variable):
        """Load variables from an ultra file
        :param variable: variables to load
        :type variable: list or str
        :return Dictionary containing 'x-axis' and 'y-axis' for each variable
        :rtype: dict
        """
        if self.curves is None:
            self.load_curves()
        if not isinstance(variable, (list, tuple)) and variable is not None:  # only one variable requested
            variable = [variable, ]

        pydv_format = self.format in ["curves", "curves/pydv", "curves/pdv", "pydv", "pdv"]

        if pydv_format or self.format == "numpy":
            variables = []
        else:
            variables = {}

        if variable is None:  # all curves
            if pydv_format:
                return self.curves
            elif self.format == "numpy":
                return numpy.array([[c.x, c.y] for c in self.curves])
            else:
                for c in self.curves:
                    name = c.name
                    variables[name] = {}
                    variables[name]['x-axis'] = c.x
                    variables[name]['y-axis'] = c.y
        else:
            for var in variable:
                for c in self.curves:
                    name = c.name
                    if name == var:
                        if pydv_format:
                            variables.append(c)
                        elif self.format == "numpy":
                            variables.append(numpy.array([c.x, c.y]))
                        else:
                            variables[name] = {}
                            variables[name]['x-axis'] = c.x
                            variables[name]['y-axis'] = c.y
                        break

        if self.format == "numpy":
            variables = numpy.array(variables)

        if len(variable) == 1:
            return variables[0]
        else:
            return variables

    def extract(self, *args, **kargs):
        """Extract a feature"""
        return self.load_from_ultra(self.feature)

    def open(self):
        """open/load matching ultra file

        :return: Dictionary containing 'x-axis' and 'y-axis' for each variable
        """
        return self.load_from_ultra(None)

    def list_features(self):
        """List features available in ultra file"""
        variables = []
        if self.curves is None:
            self.load_curves()
        for curve in self.curves:
            variables.append(curve.name)
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
        if self.curves is None:
            self.load_curves()
        for c in self.curves:
            if c.name == feature:
                info["size"] = len(c.x)
                info["first_time"] = c.x[0]
                info["last_time"] = c.x[-1]
                info["min"] = min(c.y)
                info["max"] = max(c.y)
                info["type"] = c.y.dtype
                break
        return info

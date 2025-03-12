from .core import KoshOperator


class KoshLNorm(KoshOperator):
    """Interpolates Two 2-D Arrays [x0, y0], [xy, y1] or four 1-D arrays x0, y0, x1, y1
    to a shared domain and calculates the `LNorm = np.sum(abs(y0-y1)**power)**1/power`.
    Returns the shared x-axis, the interpolated y-axis for both features, and the LNorm.
    """
    types = {"numpy": ["numpy", ],
             "dict": ["numpy", ],
             "hdf5": ["numpy", ]}

    def __init__(self, *args, **options):
        """
        :param inputs: Two 2-D Arrays [x0, y0], [xy, y1] or four 1-D arrays x0, y0, x1, y1 features or data
        :type inputs: kosh feature or data
        :param power: The power of the `LNorm = np.sum(abs(y0-y1)**power)**1/power`
        :type power: int
        :param overlap_only: Only interpolate within the overlap of both x-axis.
                             Eliminates the need for `left`, `right`, and `period` parameters below.
        :param overlap_only: bool
        :param left: The `left` parameter in `numpy.interp()`.
                     "Value to return for x < xp[0], default is fp[0]."
        :type left: float or complex
        :param right: The `right` parameter in `numpy.interp()`.
                      "Value to return for x > xp[-1], default is fp[-1]."
        :type right: float or complex
        :param period: The `period` parameter in `numpy.interp()`.
                       "A period for the x-coordinates. This parameter allows the proper interpolation of angular
                       x-coordinates. Parameters left and right are ignored if period is specified."
        :type period: None or float
        :returns:
            - x (:py:class:`ndaray`) - The shared domain x-axis
            - y0 (:py:class:`ndarray`) - The first shared domain interpolated y-axis
            - y1 (:py:class:`ndarray`) - The second shared domain interpolated y-axis
            - LNorm (:py:class:`float`) - The LNorm of the overlapping data points
        """
        super(KoshLNorm, self).__init__(*args, **options)
        self.options = options
        loaders_used = {}
        for i, arg in enumerate(args):
            loaders_used[i] = {}
            loaders_used[i]['start_nodes'] = arg.__dict__['start_nodes']
            loaders_used[i]['end_nodes'] = arg.__dict__['end_nodes']
        self.loaders_used = loaders_used

    def operate(self, *inputs, **kargs):
        import numpy as np

        features = {}
        features[0] = {}
        features[1] = {}

        # Gather data from inputs
        if len(inputs) == 2:

            for i in range(2):

                # UltraLoader dict
                if (self.loaders_used[i]['start_nodes'][0][0] == 'ultra' and
                        self.loaders_used[i]['end_nodes'][0][0] == 'dict'):

                    for key, value in inputs[i].items():
                        features[i]['x-axis'] = value['x-axis']
                        features[i]['y-axis'] = value['y-axis']

                # HDF5Loader HDF5
                elif (self.loaders_used[i]['start_nodes'][0][0] == 'hdf5' and
                      self.loaders_used[i]['end_nodes'][0][0] == 'numpy'):
                    features[i]['x-axis'] = inputs[i][0]
                    features[i]['y-axis'] = inputs[i][1]

                # 2D Numpy
                elif (self.loaders_used[i]['end_nodes'][0][0] == 'numpy'):
                    features[i]['x-axis'] = inputs[i][0]
                    features[i]['y-axis'] = inputs[i][1]

        elif len(inputs) == 4:

            # HDF5Loader HDF5 and 2D Numpy
            features[0]['x-axis'] = inputs[0]
            features[0]['y-axis'] = inputs[1]
            features[1]['x-axis'] = inputs[2]
            features[1]['y-axis'] = inputs[3]

        else:
            raise ValueError("Inputs must be two 2-D Arrays [x0, y0], [xy, y1] or four 1-D arrays x0, y0, x1, y1")

        # Acquire common domain
        if self.options.get("overlap_only"):
            x = list(features[0]['x-axis'][np.where(
                np.logical_and(features[0]['x-axis'] >= features[1]['x-axis'][0],
                               features[0]['x-axis'] <= features[1]['x-axis'][-1]))])
            x.extend(list(features[1]['x-axis'][np.where(
                np.logical_and(features[1]['x-axis'] >= features[0]['x-axis'][0],
                               features[1]['x-axis'] <= features[0]['x-axis'][-1]))]))
            x = list(set(x))
            x.sort()
        else:
            x = []
            x.extend(features[0]['x-axis'])
            x.extend(features[1]['x-axis'])
            x = list(set(x))
            x.sort()

        left = self.options.get("left")
        right = self.options.get("right")
        period = self.options.get("period")
        y0 = np.interp(x, features[0]['x-axis'], features[0]['y-axis'],
                       left=left, right=right, period=period)
        y1 = np.interp(x, features[1]['x-axis'], features[1]['y-axis'],
                       left=left, right=right, period=period)

        power = self.options.get("power", 2)
        LNorm = np.sum(abs(y0-y1)**power)**1/power
        return np.array(x), np.array(y0), np.array(y1), LNorm

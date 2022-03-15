import numpy
from .core import KoshLoader
from io import open
import six


class NpyLoader(KoshLoader):
    types = {"npy": ["numpy", ]}

    def open(self, mode='c'):
        return numpy.load(self.uri, mmap_mode=mode)

    def extract(self, feature='ndarray', format='numpy'):
        return self.open()

    def list_features(self, *args, **kargs):
        return ["ndarray", ]

    def describe_feature(self, feature):
        info = {}
        feature = self.open()
        info["size"] = feature.shape
        info["format"] = "numpy"
        info["type"] = feature.dtype
        return info


def blocks(file_, size=65536):
    while True:
        b = file_.read(size)
        if not b:
            break
        yield b


def number_of_lines(file_):
    from_string = False
    if isinstance(file_, six.string_types):
        file_ = open(file_, "r", encoding="utf-8", errors='ignore')
        from_string = True
    nlines = sum(bl.count("\n") for bl in blocks(file_))
    if from_string:
        file_.close()
    return nlines


class NumpyTxtLoader(KoshLoader):
    types = {"numpy/txt": ["numpy", ]}

    def _setup_via_metadata(self):
        # use metadata to identify
        self.skiprows = getattr(self.obj, "skiprows", 0)
        self.features_at_line = getattr(self.obj, "features_line", None)
        self.features_separator = getattr(self.obj, "features_separator", None)
        self.columns_width = getattr(self.obj, "columns_width", None)

    def list_features(self, *args, **kargs):
        self._setup_via_metadata()
        if self.features_at_line is None:
            return ["features", ]
        else:
            with open(self.obj.uri) as f:
                line = -1
                while line < self.features_at_line:
                    st = f.readline()
                    line += 1
                if self.columns_width is not None:
                    features = [st[i:i + self.columns_width].strip()
                                for i in range(0, len(st), self.columns_width)]
                else:
                    while st[0] == "#":
                        st = st[1:]
                    st = st.strip()
                    features = st.split(self.features_separator)
            return features

    def extract(self):
        self._setup_via_metadata()
        return self[:]

    def __getitem__(self, key):
        self._setup_via_metadata()
        original_key = key
        if isinstance(key, tuple):
            # double subset
            key, key2 = key[:2]  # ignore if more is sent
        else:
            key2 = None
        if isinstance(key, int):
            key = slice(key, key + 1)

        if isinstance(key, slice):
            start = key.start
            stop = key.stop
            step = key.step
            if step is None:
                step = 1
            if (start is not None and start < 0) or \
                    (stop is not None and stop < 0):
                # Doh we need to count lines
                nlines = number_of_lines(self.obj.uri)
                if start is not None and start < 0:
                    start = nlines + start
                if stop is not None and stop < 0:
                    stop = nlines + stop
            # ok if it's neg step we need to flip these two
            # it has to do with numpy loader starting at a row for n row
            # not reading a range
            if step < 0:
                start, stop = stop, start
                if stop is not None:
                    stop += 1  # because slice if exclusive on the end
            if start is None:
                start = self.skiprows
            else:
                start += self.skiprows
            if stop is not None:
                max_rows = stop - start + self.skiprows
            else:
                max_rows = None

            if max_rows is None or max_rows > 0:
                # , usecols=numpy.arange(key2.start, key2.stop, key2.step))
                data = numpy.loadtxt(
                    self.obj.uri, skiprows=start, max_rows=max_rows)
            else:
                # , usecols=numpy.arange(key2.start, key2.stop, key2.step))
                data = numpy.loadtxt(
                    self.obj.uri,
                    skiprows=start,
                    max_rows=2)[
                    0:2:-1]
            if data.ndim > 1:
                if key2 is not None:
                    data = data[::step, key2]
                elif step != 1:  # useless if step is 1
                    data = data[::step]
            else:
                if key.step is not None and key.step != 1:
                    data = data[::step]
                else:
                    if key2 is not None:
                        data = data[key2]
                    else:
                        data = data
        else:
            raise KeyError("Invalid key value: {}".format(original_key))
        if self.features_at_line is None:
            return data
        else:
            feature_index = self.list_features().index(self.feature)
            return data[:, feature_index]

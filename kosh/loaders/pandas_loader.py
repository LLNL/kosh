from .core import KoshLoader


class PandasLoader(KoshLoader):
    """ Kosh loader to load data using Pandas"""

    types = {"pandas/csv": ["DataFrame", ],
             "pandas/excel": ["DataFrame", ],
             "pandas/pickle": ["DataFrame", ],
             "pandas/table": ["DataFrame", ],
             "pandas/fwf": ["DataFrame", ],
             "pandas/clipboard": ["DataFrame", ],
             "pandas/json": ["DataFrame", ],
             "pandas/html": ["DataFrame", ],
             "pandas/xml": ["DataFrame", ],
             "pandas/hdf": ["DataFrame", ],
             "pandas/feather": ["DataFrame", ],
             "pandas/parquet": ["DataFrame", ],
             "pandas/orc": ["DataFrame", ],
             "pandas/sas": ["DataFrame", ],
             "pandas/spss": ["DataFrame", ],
             "pandas/sql_table": ["DataFrame", ],
             "pandas/sql_query": ["DataFrame", ],
             "pandas/sql": ["DataFrame", ],
             "pandas/gbq": ["DataFrame", ],
             "pandas/stata": ["DataFrame", ]}

    def _load_dataframe(self):

        import pandas as pd

        kwargs = self.obj.loader_kwargs

        if self._mime_type == 'pandas/csv':
            df = pd.read_csv(self.obj.uri, **kwargs)
        elif self._mime_type == 'pandas/excel':
            df = pd.read_excel(self.obj.uri, **kwargs)
        elif self._mime_type == 'pandas/pickle':
            df = pd.read_pickle(self.obj.uri, **kwargs)
        elif self._mime_type == 'pandas/table':
            df = pd.read_table(self.obj.uri, **kwargs)
        elif self._mime_type == 'pandas/fwf':
            df = pd.read_fwf(self.obj.uri, **kwargs)
        elif self._mime_type == 'pandas/clipboard':
            df = pd.read_clipboard(self.obj.uri, **kwargs)
        elif self._mime_type == 'pandas/json':
            df = pd.read_json(self.obj.uri, **kwargs)
        elif self._mime_type == 'pandas/html':
            df = pd.read_html(self.obj.uri, **kwargs)
        elif self._mime_type == 'pandas/xml':
            df = pd.read_xml(self.obj.uri, **kwargs)
        elif self._mime_type == 'pandas/hdf':
            df = pd.read_hdf(self.obj.uri, **kwargs)
        elif self._mime_type == 'pandas/feather':
            df = pd.read_feather(self.obj.uri, **kwargs)
        elif self._mime_type == 'pandas/parquet':
            df = pd.read_parquet(self.obj.uri, **kwargs)
        elif self._mime_type == 'pandas/orc':
            df = pd.read_orc(self.obj.uri, **kwargs)
        elif self._mime_type == 'pandas/sas':
            df = pd.read_sas(self.obj.uri, **kwargs)
        elif self._mime_type == 'pandas/spss':
            df = pd.read_spss(self.obj.uri, **kwargs)
        elif self._mime_type == 'pandas/sql_table':
            df = pd.read_sql_table(self.obj.uri, **kwargs)
        elif self._mime_type == 'pandas/sql_query':
            df = pd.read_sql_query(self.obj.uri, **kwargs)
        elif self._mime_type == 'pandas/sql':
            df = pd.read_sql(self.obj.uri, **kwargs)
        elif self._mime_type == 'pandas/gbq':
            df = pd.read_gbq(self.obj.uri, **kwargs)
        elif self._mime_type == 'pandas/stata':
            df = pd.read_stata(self.obj.uri, **kwargs)

        return df

    def open(self):
        """open/load matching file using `pandas.read_*()` method

        :return: Pandas DataFrame
        """
        df = self._load_dataframe()
        return df

    def extract(self):
        """extract return a feature from the loaded object.

        :param feature: variable to read from file
        :type feature: str
        :return: Pandas DataFrame
        """
        features = self.feature
        if not isinstance(features, list):
            features = [self.feature, ]

        df = self._load_dataframe()
        data = df[features]
        return data

    def list_features(self):
        """list_features list features in file,

        :return: list of features available in file
        :rtype: list
        """
        df = self._load_dataframe()
        features = df.columns.values.tolist()
        return features

    def describe_feature(self, feature):
        """describes the features

        :return: dictionary describing the feature
        :rtype: dict
        """
        df = self._load_dataframe()
        info = df[feature].describe()
        return info

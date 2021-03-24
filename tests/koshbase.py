import unittest
import os
from kosh import KoshStore
import kosh
import uuid
import logging

# Turn off sina logging
for name in ["sina.datastores.sql", "sina.model", "sina.utils", "sina.dao",
             "matplotlib", "matplotlib.font", "matplotlib.pyplot"]:
    logger = logging.getLogger(name)
    logger.setLevel(logging.CRITICAL)
    logger.disabled = True


class KoshTest(unittest.TestCase):
    def init_db(self, engine=None, db_uri=None):
        if engine is None:
            engine = os.environ.get("KOSH_ENGINE", "sina")
        if engine == "sina":
            # Make sure local file is new sql file
            if db_uri is None:
                kosh_test_sql_file = "kosh_test_{}.sql".format(
                    uuid.uuid1().hex)
            else:
                kosh_test_sql_file = db_uri
            if db_uri is None and os.path.exists(kosh_test_sql_file):
                os.remove(kosh_test_sql_file)
            if db_uri is None:
                kosh.utils.create_new_db(kosh_test_sql_file[:-4], verbose=False)
            return kosh_test_sql_file

    def connect(self, engine=None, db_uri=None, sync=True,
                dataset_record_type="blah"):
        if engine is None:
            engine = os.environ.get("KOSH_ENGINE", "sina")
        kosh_db = self.init_db(engine, db_uri)
        if engine == "sina":
            # os.getlogin does not work on my WSL
            store = KoshStore(engine="sina", username=os.environ["USER"], db='sql',
                              db_uri=kosh_db, sync=sync, dataset_record_type=dataset_record_type, verbose=False)
        return store, os.path.abspath(kosh_db)

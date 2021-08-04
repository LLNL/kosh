import unittest
import os
from kosh import connect
import uuid
import logging

# Turn off sina logging
for name in ["sina.datastores.sql", "sina.model", "sina.utils", "sina.dao",
             "matplotlib", "matplotlib.font", "matplotlib.pyplot"]:
    logger = logging.getLogger(name)
    logger.setLevel(logging.CRITICAL)
    logger.disabled = True


class KoshTest(unittest.TestCase):
    def connect(self, db_uri=None, sync=True,
                dataset_record_type="blah"):
        if db_uri is None:
            kosh_db = "kosh_test_{}.sql".format(uuid.uuid1().hex)
        else:
            kosh_db = db_uri
        # os.getlogin does not work on my WSL
        store = connect(database=kosh_db, sync=sync, dataset_record_type=dataset_record_type, verbose=False)
        if db_uri is None:
            store.delete_all_contents(force="SKIP PROMPT")
        return store, os.path.abspath(kosh_db)

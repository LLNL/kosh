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
    dbname = os.environ.get("KOSH_TEST_MARIADB", "cz-kosh-testkoshdb.apps.czapps.llnl.gov:30637")
    dbcnf = os.environ.get("KOSH_TEST_MARIACNF", '~/.my.kosh.testdb.cnf')
    dbcnf = os.path.expanduser(dbcnf)
    mariadb = f"mysql+mysqlconnector://{dbname}/?read_default_file={dbcnf}"

    def connect(self, db_uri=None, sync=True,
                dataset_record_type="blah",
                delete_all_contents=False, **kwargs):
        if db_uri is None:
            kosh_db = "kosh_test_{}.sql".format(uuid.uuid1().hex)
        else:
            kosh_db = db_uri
        # os.getlogin does not work on my WSL
        store = connect(database=kosh_db, sync=sync, dataset_record_type=dataset_record_type, verbose=False, **kwargs)
        if db_uri is None or delete_all_contents:
            store.delete_all_contents(force="SKIP PROMPT")
        return store, os.path.abspath(kosh_db) if "://" not in kosh_db else kosh_db

    def cleanup_store(self, store):
        store.close()
        os.remove(store.db_uri)

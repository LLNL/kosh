import unittest
import os
import sys
import shlex
from subprocess import PIPE, Popen
from kosh import KoshStore
import uuid
import logging


# Turn off sina logging
for name in ["sina.datastores.sql", "sina.model", "sina.utils"]:
    logger = logging.getLogger(name)
    logger.setLevel(logging.ERROR)


class KoshTest(unittest.TestCase):
    def init_db(self, engine=None):
        if engine is None:
            engine = os.environ.get("KOSH_ENGINE", "sina")
        if engine == "sina":
            # Make sure local file is new sql file
            kosh_test_sql_file = "kosh_test_{}.sql".format(uuid.uuid1().hex)
            if os.path.exists(kosh_test_sql_file):
                os.remove(kosh_test_sql_file)

            cmd = "{}/bin/python scripts/init_sina.py --sina_db={}".format(
                sys.prefix, kosh_test_sql_file)
            p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
            o, e = p.communicate()
            return kosh_test_sql_file

    def connect(self, engine=None):
        if engine is None:
            engine = os.environ.get("KOSH_ENGINE", "sina")
        kosh_db = self.init_db(engine)
        if engine == "sina":
            # os.getlogin does not work on my WSL
            store = KoshStore(engine="sina", username=os.environ["USER"], sql='sql',
                      db_path=kosh_db)
        return store, kosh_db

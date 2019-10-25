import unittest
import os
import sys
import shlex
from subprocess import PIPE, Popen
from kosh import KoshStore

kosh_test_sql_file = "kosh_test.sql"


class KoshTest(unittest.TestCase):
    def setUp(self):
        self.engine = os.environ.get("KOSH_ENGINE", "sina")

    def test_connect(self):
        self.init_db()
        if self.engine == "sina":
            KoshStore(engine="sina", username=os.getlogin(), sql='sql',
                      db_path=kosh_test_sql_file)

    def init_db(self):
        if self.engine == "sina":
            # Make sure local file is new sql file
            if os.path.exists(kosh_test_sql_file):
                os.remove(kosh_test_sql_file)

            cmd = "{}/bin/python scripts/init_sina.py --sina_db={}".format(
                sys.prefix, kosh_test_sql_file)
            p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
            o, e = p.communicate()

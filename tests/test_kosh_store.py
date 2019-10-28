import os
from koshbase import KoshTest


class KoshTestStore(KoshTest):
    def test_connect(self):
        store, kosh_test_sql_file = self.connect()
        os.remove(kosh_test_sql_file)

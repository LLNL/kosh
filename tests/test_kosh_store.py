import os
import kosh
from koshbase import KoshTest


class KoshTestStore(KoshTest):
    def test_connect(self):
        store, kosh_test_sql_file = self.connect()
        os.remove(kosh_test_sql_file)

    def test_create(self):
        self.assertIsInstance(
            kosh.create_new_db("blah_blah_blah.sql"),
            kosh.sina.KoshSinaStore)

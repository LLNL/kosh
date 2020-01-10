import os
from koshbase import KoshTest
import kosh
from sina.utils import DataRange

def g5(value):
    assert(value > 5)
    return True

def g5b(value):
    if value>5:
        return True
    else:
        return False


class KoshTestDataset(KoshTest):
    def test_dataset_schema(self):
        store, kosh_db = self.connect()
        # Create dataset

        schema = kosh.KoshSchema({"req1":None,
                                  "req_int": lambda x: isinstance(x, int),
                                  "req_list": [1, 2, 3],
                                  "req_list_comb": ["a", g5]},
                                 {"opt1": None, "opt_g5": g5b})

        meta = {"req1":"blah", 
                "req_int": 6,
                "req_list": 1,
                "req_list_comb": 67,

                "opt1": "blah",
                "opt_g5": 33
                }
        ds = store.create(schema=schema, metadata=meta)

        with self.assertRaises(ValueError):
            del(meta["req1"])
            meta["opt_g5"] = 3.
            meta["req_list"] = 6
            ds = store.create(schema=schema, metadata=meta)
        os.remove(kosh_db)


import os
from koshbase import KoshTest
import kosh
from kosh.schema import KoshSchema


def g5(value):
    assert(value > 5)
    return True


def g5b(value):
    if value > 5:
        return True
    else:
        return False


class KoshTestDataset(KoshTest):
    def test_store_schema(self):
        store, kosh_db = self.connect()
        schema = kosh.KoshSchema(
            {"req1": float,
             "req_int": int,
             "req_list": [1, 2, 3],
             "req_list_comb": ["a", g5]},
            {"opt1": None, "opt_g5": g5b})
        schema2 = kosh.KoshSchema(
            {"req1": None})
        # Create dataset
        ds = store.create()
        self.assertEqual(ds.schema, None)
        with self.assertRaises(ValueError):
            # We do not have all the required attributes yet
            ds.schema = schema2
        # No validation func on req1 can be anything
        ds.req1 = "blah"
        ds.schema = schema2
        self.assertTrue(isinstance(ds.schema, KoshSchema))
        self.assertTrue("req1" in ds.schema.required)
        self.assertEqual(len(ds.schema.required), 1)
        self.assertEqual(len(ds.schema.optional), 0)
        with self.assertRaises(ValueError):
            # We do not have all the required attributes yet
            ds.schema = schema
        # Add attributes
        ds.req_int = 5
        ds.req_list = 1
        ds.req_list_comb = 7
        with self.assertRaises(ValueError):
            # req1 no longer valid under that schema
            ds.schema = schema
        ds.req1 = 4.5
        ds.schema = schema
        self.assertTrue("req_int" in ds.schema.required)
        self.assertEqual(len(ds.schema.required), 4)
        self.assertEqual(len(ds.schema.optional), 2)

        store2, _ = self.connect(db_uri=kosh_db)
        ds2 = store2.open(ds.id)
        self.assertTrue(isinstance(ds2.schema, KoshSchema))
        self.assertTrue("req_int" in ds2.schema.required)
        self.assertEqual(len(ds2.schema.required), 4)
        self.assertEqual(len(ds2.schema.optional), 2)
        os.remove(kosh_db)

    def test_dataset_schema(self):
        store, kosh_db = self.connect()
        # Create dataset

        schema = kosh.KoshSchema(
            {"req1": None,
             "req_int": lambda x: isinstance(x, int),
             "req_list": [1, 2, 3],
             "req_list_comb": ["a", g5]},
            {"opt1": None, "opt_g5": g5b})

        meta = {"req1": "blah",
                "req_int": 6,
                "req_list": 1,
                "req_list_comb": 67,

                "opt1": "blah",
                "opt_g5": 33
                }
        store.create(schema=schema, metadata=meta)

        with self.assertRaises(ValueError):
            del(meta["req1"])
            meta["opt_g5"] = 3.
            meta["req_list"] = 6
            store.create(schema=schema, metadata=meta)
        os.remove(kosh_db)

    def test_validation(self):
        # Autovalidation
        goods = {int: 5, float: 6., str: "7"}
        values = [5, 6., "7"]
        for typ in goods.keys():
            for v in values:
                if goods[typ] == v:
                    self.assertTrue(kosh.schema.validate_value(v, typ))
                else:
                    with self.assertRaises(ValueError):
                        self.assertTrue(kosh.schema.validate_value(v, typ))

        # function based
        def greater_than_5(value):
            if value > 5:
                return True
            else:
                raise ValueError("Invalid value")

        self.assertTrue(kosh.schema.validate_value(78., greater_than_5))
        with self.assertRaises(ValueError):
            self.assertTrue(kosh.schema.validate_value(.5, greater_than_5))

        # list of validation
        self.assertTrue(kosh.schema.validate_value(
            78., [greater_than_5, float]))
        self.assertTrue(kosh.schema.validate_value(78., [greater_than_5, int]))
        self.assertTrue(kosh.schema.validate_value(2, [greater_than_5, int]))
        with self.assertRaises(ValueError):
            self.assertTrue(kosh.schema.validate_value(
                2, [greater_than_5, float]))


if __name__ == "__main__":
    A = KoshTestDataset()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

from __future__ import print_function
from koshbase import KoshTest
import kosh
from collections import OrderedDict
import os


class KoshTestScriptWrapper(KoshTest):
    def test_wrap_script(self):
        store, uri = self.connect()

        # Let's create two datasets with different values for param1 and 2

        ds1 = store.create(
            "first",
            metadata={
                "param1": "one",
                "param2": "two"})
        ds2 = store.create("second", metadata={"param1": "1", "param2": "2"})

        # Let's run our wrapper on ds1
        wrapper = kosh.utils.KoshScriptWrapper(
            "python tests/baselines/scripts/dummy.py", ["param1", "param2"])

        # Should use values from kosh dataset here
        o, e = wrapper(ds1)
        print(o.decode())
        self.assertTrue("P1:one" in o.decode())
        self.assertTrue("P2:two" in o.decode())
        o, e = wrapper(ds2)
        self.assertTrue("P1:1" in o.decode())
        self.assertTrue("P2:2" in o.decode())

        # Overwriting datasrt attibute!
        o, e = wrapper(ds1, param2="BLAH")
        self.assertTrue("P1:one" in o.decode())
        self.assertTrue("P2:BLAH" in o.decode())
        # Now going to let it now there's a combined param but no let it know
        # about param2

        wrapper = kosh.utils.KoshScriptWrapper(
            "python tests/baselines/scripts/dummy.py", {
                "param1": "use_default", "combined": "COMBINED"})
        o, e = wrapper(ds1)
        self.assertTrue("P1:one" in o.decode())
        self.assertTrue("P2:None" in o.decode())
        self.assertTrue("C:COMBINED" in o.decode())

        # Now changing combined to some other value
        o, e = wrapper(ds2, combined="new")
        self.assertTrue("P1:1" in o.decode())
        self.assertTrue("P2:None" in o.decode())
        self.assertTrue("C:new" in o.decode())

        # Ok now let's remap some parameter to param1 and use evaluator for
        # param2 and combined

        # Create a dataset with the mapped params and associate data with it
        ds3 = store.create("d3", metadata={"p1": "p1", "p2_a": 2, "p2_b": 3})
        ds3.associate("setup.py", mime_type="py", metadata={"combined": "CO"})

        wrapper = kosh.utils.KoshScriptWrapper("python tests/baselines/scripts/dummy.py",
                                               ["param1", "param2", "combined"],
                                               {"param1": "p1",
                                                   "param2": lambda x: getattr(x, "p2_a") + getattr(x, "p2_b"),
                                                   "combined":
                                                lambda x: getattr(
                                                    x.search(mime_type="py")[0], "combined"),
                                                }
                                               )

        o, e = wrapper(ds3)
        print("O:", o)
        self.assertTrue("P1:p1" in o.decode())
        self.assertTrue("P2:5" in o.decode())
        self.assertTrue("C:CO" in o.decode())

        # Ok now let's test for postional args
        # We will first ensure to pass two extra args w/o mapping in kosh
        # object
        pos = OrderedDict()
        pos["optional_one"] = "OPT"
        pos["opt2"] = "O2"
        wrapper = kosh.utils.KoshScriptWrapper("python tests/baselines/scripts/dummy.py",
                                               ["param1", "param2", "combined"],
                                               {"param1": "p1",
                                                   "param2": lambda x: getattr(x, "p2_a") + getattr(x, "p2_b"),
                                                   "combined":
                                                lambda x: getattr(
                                                    x.search(mime_type="py")[0], "combined"),
                                                },
                                               pos
                                               )

        o, e = wrapper(ds3)
        self.assertTrue("P1:p1" in o.decode())
        self.assertTrue("P2:5" in o.decode())
        self.assertTrue("C:CO" in o.decode())
        self.assertTrue("['OPT', 'O2']" in o.decode())

        # let's map one to the dataset and the other to some eval function
        wrapper = kosh.utils.KoshScriptWrapper("python tests/baselines/scripts/dummy.py",
                                               ["param1", "param2", "combined"],
                                               {"param1": "p1",
                                                   "optional_one": "opt1",
                                                   "param2": lambda x: getattr(x, "p2_a") + getattr(x, "p2_b"),
                                                   "combined":
                                                   lambda x: getattr(
                                                       x.search(mime_type="py")[0], "combined"),
                                                   "opt2": lambda x: getattr(x, "p2_a") - getattr(x, "p2_b"),
                                                },
                                               pos
                                               )
        ds3.opt1 = "OOO1"
        o, e = wrapper(ds3)
        self.assertTrue("P1:p1" in o.decode())
        self.assertTrue("P2:5" in o.decode())
        self.assertTrue("C:CO" in o.decode())
        self.assertTrue("['OOO1', '-1']" in o.decode())

        # Finally let s overwrite a named and an optinoal param
        o, e = wrapper(ds3, param1="NEW_P1", opt2="OVER2")
        self.assertTrue("P1:NEW_P1" in o.decode())
        self.assertTrue("P2:5" in o.decode())
        self.assertTrue("C:CO" in o.decode())
        self.assertTrue("['OOO1', 'OVER2']" in o.decode())

        # let's also test single dash parameters and double dash passed with
        # double dashes
        wrapper = kosh.utils.KoshScriptWrapper("python tests/baselines/scripts/dummy.py",
                                               ["param1", "param2",
                                                "--combined", "-r"],
                                               {"param1": "p1",
                                                   "r": "name",
                                                   "optional_one": "opt1",
                                                   "param2": lambda x: getattr(x, "p2_a") + getattr(x, "p2_b"),
                                                   "combined":
                                                   lambda x: getattr(
                                                       x.search(mime_type="py")[0], "combined"),
                                                   "opt2": lambda x: getattr(x, "p2_a") - getattr(x, "p2_b"),
                                                },
                                               pos
                                               )
        ds3.opt1 = "OOO1"
        o, e = wrapper(ds3)
        self.assertTrue("P1:p1" in o.decode())
        self.assertTrue("P2:5" in o.decode())
        self.assertTrue("C:CO" in o.decode())
        self.assertTrue("['OOO1', '-1']" in o.decode())
        self.assertTrue("Run:d3" in o.decode())

        # Finally let s overwrite a named and an optinoal param
        o, e = wrapper(ds3, param1="NEW_P1", opt2="OVER2", r="d4")
        self.assertTrue("P1:NEW_P1" in o.decode())
        self.assertTrue("P2:5" in o.decode())
        self.assertTrue("C:CO" in o.decode())
        self.assertTrue("['OOO1', 'OVER2']" in o.decode())
        self.assertTrue("Run:d4" in o.decode())

        # Cleanup
        os.remove(uri)

from __future__ import print_function
from koshbase import KoshTest
import kosh
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
            "python tests/baselines/scripts/dummy.py")
        wrapper.add_argument("--param1")
        wrapper.add_argument("--param2")

        # Should use values from kosh dataset here
        o, e = wrapper(ds1)
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
            "python tests/baselines/scripts/dummy.py")
        wrapper.add_argument("--param1")
        wrapper.add_argument("--combined", default="COMBINED")

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

        wrapper = kosh.utils.KoshScriptWrapper(
            "python tests/baselines/scripts/dummy.py")
        wrapper.add_argument("--param1", default="p1")
        wrapper.add_argument(
            "--combined",
            mapper=lambda x,
            y: getattr(
                next(x.find(
                    mime_type="py")),
                "combined"))
        wrapper.add_argument(
            "--param2",
            mapper=lambda x,
            y: getattr(
                x,
                "p2_a") +
            getattr(
                x,
                "p2_b"))

        o, e = wrapper(ds3)
        self.assertTrue("P1:p1" in o.decode())
        self.assertTrue("P2:5" in o.decode())
        self.assertTrue("C:CO" in o.decode())

        # Ok now let's test for postional args
        # We will first ensure to pass two extra args w/o mapping in kosh
        # object
        wrapper = kosh.utils.KoshScriptWrapper(
            "python tests/baselines/scripts/dummy.py")
        wrapper.add_argument("--param1", default="p1")
        wrapper.add_argument(
            "--param2",
            mapper=lambda x,
            y: getattr(
                x,
                "p2_a") +
            getattr(
                x,
                "p2_b"))
        wrapper.add_argument(
            "--combined",
            mapper=lambda x,
            y: getattr(
                next(x.find(
                    mime_type="py")),
                "combined"))
        # Pos arg 1
        wrapper.add_argument("", feed_attribute="opt1", default="OPT")
        # Pos arg 2
        wrapper.add_argument("", feed_attribute="opt2", default="O2")

        o, e = wrapper(ds3)
        self.assertTrue("P1:p1" in o.decode())
        self.assertTrue("P2:5" in o.decode())
        self.assertTrue("C:CO" in o.decode())
        self.assertTrue("['OPT', 'O2']" in o.decode())

        # let's map one to the dataset and the other to some eval function
        wrapper = kosh.utils.KoshScriptWrapper(
            "python tests/baselines/scripts/dummy.py")
        wrapper.add_argument("--param1", default="p1")
        wrapper.add_argument(
            "--param2",
            mapper=lambda x,
            y: getattr(
                x,
                "p2_a") +
            getattr(
                x,
                "p2_b"))
        wrapper.add_argument(
            "--combined",
            mapper=lambda x,
            y: getattr(
                next(x.find(
                    mime_type="py")),
                "combined"))
        # Pos arg 1
        wrapper.add_argument("", feed_attribute="opt1", default="OPT")
        # Pos arg 2
        wrapper.add_argument(
            "",
            feed_attribute="opt2",
            default="O2",
            mapper=lambda x,
            y: getattr(
                x,
                "p2_a") -
            getattr(
                x,
                "p2_b"))

        ds3.opt1 = "OOO1"
        o, e = wrapper(ds3)
        self.assertTrue("P1:p1" in o.decode())
        self.assertTrue("P2:5" in o.decode())
        self.assertTrue("C:CO" in o.decode())
        self.assertTrue("['OOO1', '-1']" in o.decode())

        # Finally let s overwrite a named and an optional param
        o, e = wrapper(ds3, param1="NEW_P1", opt2="OVER2")
        self.assertTrue("P1:NEW_P1" in o.decode())
        self.assertTrue("P2:5" in o.decode())
        self.assertTrue("C:CO" in o.decode())
        self.assertTrue("['OOO1', 'OVER2']" in o.decode())

        # let's also test single dash parameters and double dash passed with
        # double dashes
        wrapper = kosh.utils.KoshScriptWrapper(
            "python tests/baselines/scripts/dummy.py")
        wrapper.add_argument("--param1", default="p1")
        wrapper.add_argument(
            "--param2",
            mapper=lambda x,
            y: getattr(
                x,
                "p2_a") +
            getattr(
                x,
                "p2_b"))
        wrapper.add_argument(
            "--combined",
            mapper=lambda x,
            y: getattr(
                next(x.find(
                    mime_type="py")),
                "combined"))
        wrapper.add_argument("-r", feed_attribute="name")
        # Pos arg 1
        wrapper.add_argument("", feed_attribute="opt1", default="OPT")
        # Pos arg 2
        wrapper.add_argument(
            "",
            feed_attribute="opt2",
            default="O2",
            mapper=lambda x,
            y: getattr(
                x,
                "p2_a") -
            getattr(
                x,
                "p2_b"))
        ds3.opt1 = "OOO1"
        o, e = wrapper(ds3)
        self.assertTrue("P1:p1" in o.decode())
        self.assertTrue("P2:5" in o.decode())
        self.assertTrue("C:CO" in o.decode())
        self.assertTrue("['OOO1', '-1']" in o.decode())
        self.assertTrue("Run:d3" in o.decode())

        # Finally let s overwrite a named and an optional param
        o, e = wrapper(ds3, param1="NEW_P1", opt2="OVER2", r="d4")
        self.assertTrue("P1:NEW_P1" in o.decode())
        self.assertTrue("P2:5" in o.decode())
        self.assertTrue("C:CO" in o.decode())
        self.assertTrue("['OOO1', 'OVER2']" in o.decode())
        self.assertTrue("Run:d4" in o.decode())

        # At this point trying to get parameters for multiple fed objects
        wrapper = kosh.utils.KoshScriptWrapper(
            "python tests/baselines/scripts/dummy.py")
        wrapper.add_argument("--param1", feed_pos=0)
        wrapper.add_argument("--param2", feed_pos=1)

        # creating datasets with both values but because of feed_pos it should
        # know where to pick from
        ds1 = store.create(metadata={"param1": 1, "param2": 1})
        ds2 = store.create(metadata={"param1": 2, "param2": 2})
        # Dataset with just p1 and p3
        ds3 = store.create(metadata={"param1": 3, "param3": 3})
        o, e = wrapper.run(ds1, ds2)
        self.assertTrue("P1:1" in o.decode())
        self.assertTrue("P2:2" in o.decode())
        o, e = wrapper.run(ds2, ds1)
        self.assertTrue("P1:2" in o.decode())
        self.assertTrue("P2:1" in o.decode())

        # Ok by default as soon as an object match it will keep it
        wrapper = kosh.utils.KoshScriptWrapper(
            "python tests/baselines/scripts/dummy.py")
        wrapper.add_argument("--param1")
        wrapper.add_argument("--param2")
        o, e = wrapper.run(ds1, ds2)
        self.assertTrue("P1:1" in o.decode())
        self.assertTrue("P2:1" in o.decode())
        o, e = wrapper.run(ds2, ds1)
        self.assertTrue("P1:2" in o.decode())
        self.assertTrue("P2:2" in o.decode())

        # ok we can also force it to override in case the next object as the
        # value
        wrapper = kosh.utils.KoshScriptWrapper(
            "python tests/baselines/scripts/dummy.py")
        wrapper.add_argument("--param1", feed_pos=-1)
        wrapper.add_argument("--param2", feed_pos=-1)
        o, e = wrapper.run(ds1, ds2)
        self.assertTrue("P1:2" in o.decode())
        self.assertTrue("P2:2" in o.decode())
        o, e = wrapper.run(ds2, ds1)
        self.assertTrue("P1:1" in o.decode())
        self.assertTrue("P2:1" in o.decode())
        o, e = wrapper.run(ds1, ds2, ds3)
        self.assertTrue("P1:3" in o.decode())
        self.assertTrue("P2:2" in o.decode())

        # Ok now make sure if we say the feed_pos and not in obj then use
        # default
        wrapper = kosh.utils.KoshScriptWrapper(
            "python tests/baselines/scripts/dummy.py")
        wrapper.add_argument("--param1", default=0, feed_pos=0)
        wrapper.add_argument("--param2", feed_pos=-1)
        del(ds1.param1)
        o, e = wrapper.run(ds1, ds2, ds3)
        self.assertTrue("P1:0" in o.decode())
        self.assertTrue("P2:2" in o.decode())

        # Finally let's do similar things but with pos param
        wrapper = kosh.utils.KoshScriptWrapper(
            "python tests/baselines/scripts/dummy.py")
        wrapper.add_argument("", feed_attribute="opt1", feed_pos=0)
        wrapper.add_argument("", feed_attribute="opt2", feed_pos=1)
        # if not present in any should not even be on cmd line
        o, e = wrapper.run(ds1, ds2, ds3)
        self.assertTrue("[]" in o.decode())
        ds2.opt1 = "OPT1"
        ds1.opt2 = "OPT2"
        # Present but at wrong pos
        o, e = wrapper.run(ds1, ds2, ds3)
        self.assertTrue("[]" in o.decode())

        # Ok now back to default as soon as found use it
        wrapper = kosh.utils.KoshScriptWrapper(
            "python tests/baselines/scripts/dummy.py")
        wrapper.add_argument("", feed_attribute="opt1")
        wrapper.add_argument("", feed_attribute="opt2")
        ds1.opt1 = "opt1"
        del(ds1.opt2)
        o, e = wrapper.run(ds1, ds2, ds3)
        self.assertTrue("['opt1']" in o.decode())

        # Ok now the other way around, as long as you find it, overwrite
        wrapper = kosh.utils.KoshScriptWrapper(
            "python tests/baselines/scripts/dummy.py")
        wrapper.add_argument("", feed_attribute="opt1", feed_pos=-1)
        wrapper.add_argument("", feed_attribute="opt2", feed_pos=-1)
        ds1.opt1 = "opt1"
        o, e = wrapper.run(ds1, ds2, ds3)
        self.assertTrue("['OPT1']" in o.decode())

        # Let's also check that the order in which pos arg are defined matters
        wrapper = kosh.utils.KoshScriptWrapper(
            "python tests/baselines/scripts/dummy.py")
        wrapper.add_argument(
            "",
            feed_attribute="opt2",
            default="O2",
            feed_pos=-1)
        wrapper.add_argument("", feed_attribute="opt1", feed_pos=-1)
        ds1.opt1 = "opt1"
        o, e = wrapper.run(ds1, ds2, ds3)
        self.assertTrue("['O2', 'OPT1']" in o.decode())

        wrapper = kosh.utils.KoshScriptWrapper(
            "python tests/baselines/scripts/dummy.py")
        wrapper.add_argument("-r", feed_attribute="run", default="r2")
        ds3.run = 'd3'
        o, e = wrapper.run(ds1, ds2, ds3)
        self.assertTrue("Run:d3" in o.decode())

        # Test for single dash args
        # Cleanup
        os.remove(uri)


if __name__ == "__main__":
    A = KoshTestScriptWrapper()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()

from __future__ import print_function
import os
from koshbase import KoshTest


class KoshTestEnsembles(KoshTest):
    def test_create_and_print(self):
        store, db = self.connect()
        e1 = store.create_ensemble()
        e1.root = "foo"
        printTestResults = """\
KOSH ENSEMBLE
        id: {id}
        name: Unnamed Ensemble
        creator: {creator}

--- Attributes ---
        creator: {creator}
        name: Unnamed Ensemble
        root: foo
--- Associated Data ({n_data})---{data}
--- Member Datasets ({n_datasets})---
        {datasets}"""
        self.maxDiff = None
        e1_str = str(e1).replace("\t", "        ")
        self.assertEqual(
            e1_str,
            printTestResults.format(
                id=e1.id,
                creator=e1.creator,
                n_data=0,
                data="",
                n_datasets=0,
                datasets=[]))
        ds1 = e1.create()
        self.assertEqual(len(list(e1.get_members(ids_only=True))), 1)
        # test ds1 string
        ds1_str = str(ds1).replace("\t", "        ").strip()
        username = os.environ.get("USER", "default")
        good_ds1 = """KOSH DATASET
        id: {}
        name: Unnamed Dataset
        creator: {}

--- Attributes ---
        creator: {}
        name: Unnamed Dataset
--- Associated Data (0)---
--- Ensembles (1)---
        ['{}']
--- Ensemble Attributes ---
        --- Ensemble {} ---
                root: foo
""".format(str(ds1.id), username, username, str(e1.id), str(e1.id))
        self.assertEqual(ds1_str, good_ds1.strip())
        e1_str = str(e1).replace("\t", "        ")
        self.assertEqual(
            e1_str,
            printTestResults.format(
                id=e1.id,
                creator=e1.creator,
                n_data=0,
                data="",
                n_datasets=1,
                datasets="['{}']".format(
                    ds1.id)))
        with self.assertRaises(ValueError):
            # Cannot create an ensemble from an ensemble
            e1.create(sina_type=store._ensembles_type)
        ds2 = store.create()
        ds2.child = "bar"
        e1.add(ds2)
        self.assertEqual(len(list(e1.get_members(ids_only=True))), 2)
        e2 = store.create(
            sina_type=store._ensembles_type,
            metadata={
                "bar": "closed"})
        self.assertEqual(e2.bar, "closed")
        bad_ds = store.create(metadata={"root": "bad"})
        with self.assertRaises(ValueError):
            e1.add(bad_ds)
        ok_ds = store.create(metadata={"root": "foo"})
        ok_ds.join_ensemble(e1)
        e1.root = "foobar"
        # check it propagates to all members
        self.assertEqual([x.root for x in e1.get_members()],
                         ["foobar", "foobar", "foobar"])
        e1.new_att = "bar"
        # check it propagates
        self.assertEqual([x.new_att for x in e1.get_members()], [
                         "bar", "bar", "bar"])
        # check we can't set root from members
        with self.assertRaises(KeyError):
            ds1.root = "some value"
        e1.associate(
            r"tests/baselines/node_extracts2/node_extracts2.hdf5",
            "hdf5")
        self.assertEqual(e1.list_features(), ['cycles', 'direction', 'elements', 'node',
                                              'node/metrics_0', 'node/metrics_1', 'node/metrics_10',
                                              'node/metrics_11', 'node/metrics_12', 'node/metrics_2', 'node/metrics_3',
                                              'node/metrics_4', 'node/metrics_5', 'node/metrics_6', 'node/metrics_7',
                                              'node/metrics_8', 'node/metrics_9', 'zone', 'zone/metrics_0',
                                              'zone/metrics_1', 'zone/metrics_2', 'zone/metrics_3', 'zone/metrics_4'])
        self.assertEqual(ds1.list_features(), ['cycles', 'direction', 'elements', 'node', 'node/metrics_0',
                                               'node/metrics_1', 'node/metrics_10', 'node/metrics_11',
                                               'node/metrics_12', 'node/metrics_2', 'node/metrics_3',
                                               'node/metrics_4', 'node/metrics_5', 'node/metrics_6', 'node/metrics_7',
                                               'node/metrics_8', 'node/metrics_9', 'zone', 'zone/metrics_0',
                                               'zone/metrics_1', 'zone/metrics_2', 'zone/metrics_3', 'zone/metrics_4'])
        self.assertEqual(len(list(e1.get_associated_data(ids_only=True))), 1)
        store.close()
        os.remove(db)

    def testImportEnsemble(self):
        s1, db1 = self.connect()
        s2, db2 = self.connect()

        e1 = s1.create_ensemble()
        e1.create()
        e1.create(name="blah")

        self.assertEqual(len(list(s1.find())), 2)
        self.assertEqual(len(list(s1.find(types=s1._ensembles_type))), 1)

        s2.import_dataset(e1)
        self.assertEqual(len(list(s2.find())), 2)
        self.assertEqual(len(list(s2.find(types=s2._ensembles_type))), 1)
        s1.close()
        s2.close()
        os.remove(db1)
        os.remove(db2)

    def testMultiEnsembles(self):
        s, db = self.connect()
        e1 = s.create_ensemble(metadata={"root": "/some/root1"})
        e2 = s.create_ensemble(metadata={"code": "/some/code"})
        d1 = e1.create(name="d1")
        self.assertEqual(len(list(e1.get_members())), 1)
        self.assertEqual(len(list(d1.get_ensembles())), 1)
        # now let's add to second ensemble
        e2.add(d1)
        self.assertEqual(len(list(e1.get_members())), 1)
        self.assertEqual(len(list(e2.get_members())), 1)
        self.assertEqual(len(list(d1.get_ensembles())), 2)
        # Now let's make sure its attribute are combined
        atts = d1.list_attributes()
        self.assertTrue("root" in atts)
        self.assertTrue("code" in atts)
        e1.root = "new root"
        self.assertEqual(d1.root, "new root")
        e2.code = "new code"
        self.assertEqual(d1.code, "new code")

        # ok now let's make sure we cannot add to incompatible ensemble
        e3 = s.create_ensemble(metadata={"root": "/some/root3"})
        with self.assertRaises(Exception) as err:
            d1.join_ensemble(e3)
            print(err)
        # ok now changing ensemble attribute on e2 should lead to error
        with self.assertRaises(NameError):
            e2.root = "blah"

        self.assertEqual(len(list(s.find_ensembles())), 3)
        self.assertEqual(len(list(s.find_ensembles("root"))), 2)
        self.assertEqual(len(list(s.find_ensembles("code"))), 1)
        s.close()
        os.remove(db)

    def test_search(self):
        s, db = self.connect()
        e1 = s.create_ensemble(metadata={"root": "/some/root1"})
        e1.create(metadata={"param1": 4})
        e1.create(metadata={"param1": 3})
        # Create datasets with same attributes but not in ensemble
        s.create(metadata={"param1": 4})
        s.create(metadata={"param1": 3})

        # Store should find 2 datasets
        self.assertEqual(len(list(s.find(param1=4))), 2)
        # Ensemble should find 1 datasets
        self.assertEqual(len(list(e1.find_datasets(param1=4))), 1)
        s.close()
        os.remove(db)

    def test_add_remove(self):

        store, db_uri = self.connect()

        ensemble = store.create_ensemble()
        _ = ensemble.create()

        self.assertEqual(len(list(ensemble.get_members(ids_only=True))), 1)

        d1 = store.create()
        d1.join_ensemble(ensemble)
        self.assertEqual(len(list(ensemble.get_members(ids_only=True))), 2)
        d1.leave_ensemble(ensemble)
        self.assertEqual(len(list(ensemble.get_members(ids_only=True))), 1)
        ensemble.add(d1)
        self.assertEqual(len(list(ensemble.get_members(ids_only=True))), 2)
        ensemble.delete(d1)
        self.assertEqual(len(list(ensemble.get_members(ids_only=True))), 1)
        ensemble.add(d1)
        ensemble.remove(d1)
        self.assertEqual(len(list(ensemble.get_members(ids_only=True))), 1)
        store.close()
        os.remove(db_uri)

    def test_import_creator(self):
        a, dba = self.connect()
        b, dbb = self.connect()
        a_en = a.create_ensemble('a_en')
        a_ds = a.create('a_ds', metadata={'attr1': 10})
        self.assertEqual(a_ds.attr1, 10)
        a_en.add(a_ds)
        self.assertTrue(a_ds.is_member_of(a_en))
        # Create a dataset with the same name as a_ds so that it will merge
        # when imported
        b_ds = b.create('a_ds', metadata={'attr1': 500})
        a.import_dataset(b.export_dataset(b_ds), merge_handler='overwrite')
        self.assertTrue(a_ds.is_member_of(a_en))
        self.assertEqual(a_ds.attr1, 500)

        a.close()
        b.close()
        os.remove(dba)
        os.remove(dbb)

    def test_cannot_clone_ensemble(self):
        a, dba = self.connect()

        a_en = a.create_ensemble('a_en')
        a_en.create()
        with self.assertRaises(NotImplementedError):
            a_en.clone()
        os.remove(dba)

    def test_ensemble_list_attribute(self):
        a, dba = self.connect()

        a_en = a.create_ensemble('a_en')
        a_en.root = "foo"
        a_en.bar = None
        self.assertEqual(sorted(a_en.list_attributes(
            no_duplicate=True)), ["bar", "root"])
        a.close()
        os.remove(dba)

    def test_is_ensemble_attribute(self):
        a, dba = self.connect()

        a_en = a.create_ensemble('a_en', Id="A")
        a_en.root = "foo"
        a_en.bar = None
        ds = a_en.create(metadata={"local": True})
        self.assertFalse(ds.is_ensemble_attribute("local"))
        self.assertTrue(ds.is_ensemble_attribute("root"))
        self.assertEqual(ds.is_ensemble_attribute("root", ensemble_id=True), a_en.id)
        self.assertTrue(ds.is_ensemble_attribute("root", a_en.id))
        with self.assertRaises(ValueError):
            self.assertTrue(ds.is_ensemble_attribute("root", ds.id))
        b_en = a.create_ensemble('b_en', Id="B")
        b_en.foo = "bar"
        self.assertFalse(ds.is_ensemble_attribute("foo"))
        self.assertEqual(ds.is_ensemble_attribute("foo", ensemble_id=True), "")
        # Weird but true you can pass an enemble you're not part of
        self.assertTrue(ds.is_ensemble_attribute("foo", b_en))
        ds.join_ensemble(b_en)
        self.assertTrue(ds.is_ensemble_attribute("foo"))
        self.assertEqual(ds.is_ensemble_attribute("root", ensemble_id=True), a_en.id)
        self.assertEqual(ds.is_ensemble_attribute("foo", ensemble_id=True), b_en.id)
        self.assertFalse(ds.is_ensemble_attribute("foo", a_en))
        a.close()
        os.remove(dba)

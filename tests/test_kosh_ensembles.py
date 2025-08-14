from __future__ import print_function
import os
from koshbase import KoshTest
import kosh


class KoshTestEnsembles(KoshTest):
    def test_create_dataset_with_ensemble_attributes(self):
        store, db = self.connect()
        e = store.create_ensemble()
        e.root = "foo"
        with self.assertRaises(ValueError):
            e.create(metadata={"root": "foo2"})
        with self.assertWarns(UserWarning):
            ds = e.create(metadata={"root": "foo"})
        self.assertEqual(ds.root, "foo")
        with self.assertRaises(KeyError):
            ds.root = "bar"
        self.assertEqual(ds.root, "foo")
        with self.assertWarns(UserWarning):
            ds.root = "foo"
        self.assertEqual(ds.root, "foo")

    def test_create_and_print(self):
        store, db = self.connect()
        e1 = store.create_ensemble()
        e1.root = "foo"
        printTestResults = """\
KOSH ENSEMBLE
        id: {id}
        name: Unnamed Ensemble
        creator: {creator}
        creation date: {creation_date}
        last modified date: {last_modified_date}

--- Attributes ---
        creation_date: {creation_date}
        creator: {creator}
        last_modified_date: {last_modified_date}
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
                creation_date=e1.creation_date,
                last_modified_date=e1.last_modified_date,
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
        creation date: {}
        last modified date: {}

--- Attributes ---
        creation_date: {}
        creator: {}
        last_modified_date: {}
        name: Unnamed Dataset
--- Associated Data (0)---
--- Ensembles (1)---
        ['{}']
--- Ensemble Attributes ---
        --- Ensemble {} ---
                ['root']
--- Alias Feature Dictionary ---
""".format(str(ds1.id), username, ds1.creation_date, ds1.last_modified_date,
           ds1.creation_date, username, ds1.last_modified_date, str(e1.id), str(e1.id))
        self.assertEqual(ds1_str, good_ds1.strip())
        e1_str = str(e1).replace("\t", "        ")
        self.assertEqual(
            e1_str,
            printTestResults.format(
                id=e1.id,
                creator=e1.creator,
                creation_date=e1.creation_date,
                last_modified_date=e1.last_modified_date,
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

    def test_ensemble_tags(self):
        store, kosh_db = self.connect()

        n_ensembles = 10
        n_datasets = 10

        datasets = []

        for i in range(n_datasets):
            metadata = {f"{ia}": f"dataset_{i}_attributes_{ia}" for ia in range(n_datasets)}
            metadata['same'] = 'dataset'
            ds = store.create(id=f"dataset_{i}", metadata=metadata)
            datasets.append(ds)

        for i in range(n_ensembles):
            metadata = {f"{ia}": f"ensemble_{i}_attributes_{ia}" for ia in range(n_ensembles)}
            metadata['same'] = 'ensemble'
            ens = store.create_ensemble(id=f"ensemble_{i}", metadata=metadata)
            for j, ds in enumerate(datasets):
                ensemble_tags = {}
                if j % 2 == 0:
                    ensemble_tags["eoo"] = "even"
                else:
                    ensemble_tags["eoo"] = "odd"

                if j % 5 == 0:
                    ensemble_tags["data_type"] = "test data"
                else:
                    ensemble_tags["data_type"] = "train data"

                ens.add(ds, inherit_attributes=False, ensemble_tags=ensemble_tags)
                ds.same = f'dataset {j}'

            ens.same = f'ensemble {i}'

            ds = list(ens.find_datasets(ensemble_tags={"eoo": "even"}))
            self.assertEqual(len(ds), 5)
            ds = list(ens.find_datasets(ensemble_tags={"data_type": "test data"}))
            self.assertEqual(len(ds), 2)
            ds = list(ens.find_datasets(ensemble_tags={"eoo": "even", "data_type": "test data"}))
            self.assertEqual(len(ds), 1)

        self.assertEqual(ens.same, 'ensemble 9')

        # Attributes and Tags
        ds_atts_and_tags = ds[0].list_attributes(ensemble_id=ens.id)
        self.assertEqual(ds_atts_and_tags, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                                            'creation_date', 'creator', 'id', 'last_modified_date', 'name',
                                            'same',
                                            'ensemble_9_ENSEMBLE_TAG_data_type', 'ensemble_9_ENSEMBLE_TAG_eoo'])

        ds_atts_and_tags = ds[0].list_attributes(ensemble_id=ens.id, obscure=False)
        self.assertEqual(ds_atts_and_tags, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                                            'creation_date', 'creator', 'id', 'last_modified_date', 'name',
                                            'same',
                                            'ensemble_9_ENSEMBLE_TAG_INHERIT_ATTRIBUTES',
                                            'ensemble_9_ENSEMBLE_TAG_data_type', 'ensemble_9_ENSEMBLE_TAG_eoo'])

        ds_atts_and_tags = ds[0].list_attributes(dictionary=True, ensemble_id=ens.id)
        self.assertDictEqual(ds_atts_and_tags, {'0': 'dataset_0_attributes_0',
                                                '1': 'dataset_0_attributes_1',
                                                '2': 'dataset_0_attributes_2',
                                                '3': 'dataset_0_attributes_3',
                                                '4': 'dataset_0_attributes_4',
                                                '5': 'dataset_0_attributes_5',
                                                '6': 'dataset_0_attributes_6',
                                                '7': 'dataset_0_attributes_7',
                                                '8': 'dataset_0_attributes_8',
                                                '9': 'dataset_0_attributes_9',
                                                'creation_date': ds_atts_and_tags['creation_date'],
                                                'creator': os.environ.get("USER", "default"),
                                                'id': 'dataset_0',
                                                'last_modified_date': ds_atts_and_tags['last_modified_date'],
                                                'name': 'Unnamed Dataset',
                                                'same': 'dataset 0',
                                                'ensemble_9_ENSEMBLE_TAG_data_type': 'test data',
                                                'ensemble_9_ENSEMBLE_TAG_eoo': 'even'})

        ds_atts_and_tags = ds[0].list_attributes(dictionary=True, ensemble_id=ens.id, obscure=False)
        self.assertDictEqual(ds_atts_and_tags, {'0': 'dataset_0_attributes_0',
                                                '1': 'dataset_0_attributes_1',
                                                '2': 'dataset_0_attributes_2',
                                                '3': 'dataset_0_attributes_3',
                                                '4': 'dataset_0_attributes_4',
                                                '5': 'dataset_0_attributes_5',
                                                '6': 'dataset_0_attributes_6',
                                                '7': 'dataset_0_attributes_7',
                                                '8': 'dataset_0_attributes_8',
                                                '9': 'dataset_0_attributes_9',
                                                'creation_date': ds_atts_and_tags['creation_date'],
                                                'creator': os.environ.get("USER", "default"),
                                                'id': 'dataset_0',
                                                'last_modified_date': ds_atts_and_tags['last_modified_date'],
                                                'name': 'Unnamed Dataset',
                                                'same': 'dataset 0',
                                                'ensemble_9_ENSEMBLE_TAG_INHERIT_ATTRIBUTES': False,
                                                'ensemble_9_ENSEMBLE_TAG_data_type': 'test data',
                                                'ensemble_9_ENSEMBLE_TAG_eoo': 'even'})

        ds_atts_and_tags = ds[0].list_attributes(ensemble_id=['ensemble_0', 'ensemble_9'])
        self.assertEqual(ds_atts_and_tags, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                                            'creation_date', 'creator', 'id', 'last_modified_date', 'name',
                                            'same',
                                            'ensemble_0_ENSEMBLE_TAG_data_type', 'ensemble_0_ENSEMBLE_TAG_eoo',
                                            'ensemble_9_ENSEMBLE_TAG_data_type', 'ensemble_9_ENSEMBLE_TAG_eoo'])

        # Only tags
        ds_tags = ds[0].list_ensemble_tags(ensemble_id=ens.id)
        self.assertEqual(ds_tags, ['ensemble_9_ENSEMBLE_TAG_data_type', 'ensemble_9_ENSEMBLE_TAG_eoo'])

        ds_tags = ds[0].list_ensemble_tags(dictionary=True, ensemble_id=ens.id)
        self.assertDictEqual(ds_tags, {'ensemble_9_ENSEMBLE_TAG_data_type': 'test data',
                                       'ensemble_9_ENSEMBLE_TAG_eoo': 'even'})

        ds_tags = ds[0].list_ensemble_tags(ensemble_id=['ensemble_0', 'ensemble_9'])
        self.assertEqual(ds_tags, ['ensemble_0_ENSEMBLE_TAG_data_type', 'ensemble_0_ENSEMBLE_TAG_eoo',
                                   'ensemble_9_ENSEMBLE_TAG_data_type', 'ensemble_9_ENSEMBLE_TAG_eoo'])

        ens.remove(ds[0])

        ds_ens_tags = ds[0].list_attributes(ensemble_id=ens.id)
        self.assertEqual(ds_ens_tags, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                                       'creation_date', 'creator', 'id', 'last_modified_date', 'name',
                                       'same'])

        # Testing Schema
        required = {"color": None}
        optional = {"number": [0, 10]}
        schema_ds = kosh.KoshSchema(required, optional)
        ds = store.create(id="dataset_schema", metadata={"color": "blue", "number": 0}, schema=schema_ds)

        required = {"color": None}
        optional = {"number": [10, 100]}
        schema_ens = kosh.KoshSchema(required, optional)
        ens = store.create_ensemble(id="ensemble_schema", metadata={"color": "red",  "number": 100}, schema=schema_ens)

        # Doesn't pass ensemble schema
        with self.assertRaises(ValueError):
            ens.add(ds, inherit_attributes=False, ensemble_tags={"color": "green",  "number": 0})

        # Doesn't pass dataset schema
        with self.assertRaises(ValueError):
            ens.add(ds, inherit_attributes=False, ensemble_tags={"color": "green",  "number": 100})

        # Passes both
        ens.add(ds, inherit_attributes=False, ensemble_tags={"color": "green",  "number": 10})

    def test_long_string_as_attribute_without_verbosity(self):
        store, db_uri = self.connect(verbose_attributes=False)
        ens = store.create_ensemble()
        long_string = (
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit, "
            "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
            "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris "
            "nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in "
            "reprehenderit in voluptate velit esse cillum dolore eu fugiat "
            "nulla pariatur. Excepteur sint occaecat cupidatat non proident, "
            "sunt in culpa qui officia deserunt mollit anim id est laborum."
        ) * 10
        ens.long_string = long_string
        self.assertLess(len(ens.__str__()), len(long_string))
        self.assertEqual(len(ens.long_string), len(long_string))
        store.close()
        os.remove(db_uri)

    def test_long_string_as_attribute_with_verbosity(self):
        store, db_uri = self.connect(verbose_attributes=True)
        ens = store.create_ensemble()
        long_string = (
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit, "
            "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
            "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris "
            "nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in "
            "reprehenderit in voluptate velit esse cillum dolore eu fugiat "
            "nulla pariatur. Excepteur sint occaecat cupidatat non proident, "
            "sunt in culpa qui officia deserunt mollit anim id est laborum."
        ) * 10
        ens.long_string = long_string
        self.assertGreater(len(ens.__str__()), len(long_string))
        self.assertEqual(len(ens.long_string), len(long_string))
        store.close()
        os.remove(db_uri)

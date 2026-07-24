from __future__ import print_function

import os

from koshbase import KoshTest

from kosh.parameter_store import StepRequest, apply_step


class KoshTestParameterStore(KoshTest):
    def test_apply_step_init_then_update_with_zero_tolerances(self):
        store, uri = self.connect()
        store.create_ensemble(name="ens")
        store.close()

        init_params = {"p0": 1.0, "p1": 2.0}

        init_ids = apply_step(
            StepRequest(
                store_uri=uri,
                ensemble_name="ens",
                step="init",
                init_step_value="init",
                init_params=init_params,
                connect_kwargs={"dataset_record_type": "blah"},
            )
        )
        self.assertEqual(len(init_ids), 1)

        update_ids = apply_step(
            StepRequest(
                store_uri=uri,
                ensemble_name="ens",
                step="create",
                init_step_value="init",
                init_params=init_params,
                rtol=0.0,
                atol=0.0,
                metadata_updates={"foo": "bar"},
                connect_kwargs={"dataset_record_type": "blah"},
            )
        )
        self.assertEqual(update_ids, init_ids)

        store, _ = self.connect(db_uri=uri)
        ds = list(store.find(id_pool=init_ids, ids_only=False))[0]
        self.assertEqual(getattr(ds, "workflow_step"), "create")
        self.assertEqual(getattr(ds, "foo"), "bar")
        store.close()
        os.remove(uri)

    def test_apply_step_upsert_init_does_not_create_duplicate(self):
        store, uri = self.connect()
        store.create_ensemble(name="ens")
        store.close()

        init_params = {"p0": 1.0, "p1": 2.0}

        first_ids = apply_step(
            StepRequest(
                store_uri=uri,
                ensemble_name="ens",
                step="init",
                init_step_value="init",
                init_params=init_params,
                connect_kwargs={"dataset_record_type": "blah"},
            )
        )
        self.assertEqual(len(first_ids), 1)

        # Slightly different, but within atol/rtol => should upsert into the same dataset id
        second_ids = apply_step(
            StepRequest(
                store_uri=uri,
                ensemble_name="ens",
                step="init",
                init_step_value="init",
                init_params={"p0": 1.0000000005, "p1": 2.0},
                rtol=1e-5,
                atol=1e-8,
                upsert_init=True,
                metadata_updates={"duplicate_attempt": "true"},
                connect_kwargs={"dataset_record_type": "blah"},
            )
        )
        self.assertEqual(second_ids, first_ids)

        store, _ = self.connect(db_uri=uri)
        ensemble = list(store.find_ensembles(name="ens"))[0]
        self.assertEqual(len(list(ensemble.find_datasets())), 1)
        ds = list(store.find(id_pool=first_ids, ids_only=False))[0]
        self.assertEqual(getattr(ds, "duplicate_attempt"), "true")
        store.close()
        os.remove(uri)

    def test_apply_step_without_dataset_record_type_searches_all_types(self):
        store, uri = self.connect(dataset_record_type="some_new_type")
        store.create(metadata={"p0": 1.0, "workflow_step": "init"})
        store.close()

        update_ids = apply_step(
            StepRequest(
                store_uri=uri,
                ensemble_name=None,
                step="create",
                init_step_value="init",
                init_params={"p0": 1.0},
                metadata_updates={"foo": "bar"},
            )
        )
        self.assertEqual(len(update_ids), 1)

        store, _ = self.connect(db_uri=uri)
        ds = list(store.find(id_pool=update_ids, ids_only=False))[0]
        self.assertEqual(getattr(ds, "foo"), "bar")
        self.assertEqual(store.get_record(update_ids[0])["type"], "some_new_type")
        store.close()
        os.remove(uri)

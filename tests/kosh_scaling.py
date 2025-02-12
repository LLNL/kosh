import os
import sys
import time
import random
import kosh
import argparse
import csv
import logging
from datetime import datetime
from sina.model import generate_record_from_json

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--store", "-s", help="store", type=str, default="test.sql")
parser.add_argument("--run-number", "-r", help="run number", type=int, default=0)
parser.add_argument("--datasets", "-d", help="number of datasets to create", type=int, default=2)
parser.add_argument("--ensembles", "-e", help="number of ensembles to create", type=int, default=2)
parser.add_argument("--lock-path", "-p", help="path to lock file", default=None)
parser.add_argument("--lock-strategy", "-l", help="Implement lock strategy",
                    choices=["None", "RFileLock", "OnlyRetry"])
parser.add_argument("--log-level", "-L", help="Log Level", type=int, default=20)
parser.add_argument("--timeout", "-t", help="Timeout per try", type=int, default=300)
parser.add_argument("--clear", "-c", help="Only clear store", action="store_true")
parser.add_argument("--retries", "-R", help="Number of retries", type=int, default=10)

args = parser.parse_args()

# Only clear store since running script in parallel will cause race conditions resulting in missing data.
if args.clear:
    store = kosh.connect(args.store, delete_all_contents=True)
    time.sleep(60)
    sys.exit(0)

start = datetime.now()

os.environ['LOCK_STRATEGIES_LOG_LEVEL'] = str(args.log_level)  # 20 is logging.INFO, 10 is logging.DEBUG
LOGGER = logging.getLogger()
LOGGER.setLevel(args.log_level)

if args.lock_strategy == "None":
    ls = None
elif args.lock_strategy == "RFileLock":
    ls = kosh.lock_strategies.RFileLock(num_tries=args.retries, lock_path=args.lock_path)
elif args.lock_strategy == "OnlyRetry":
    ls = kosh.lock_strategies.OnlyRetry(num_tries=args.retries)

store = kosh.connect(args.store, lock_strategy=ls)

#################
#     Store     #
#################

# DONE store.add_loader(loader, save=False)
# DONE store.delete_loader(loader, permanently=False)
# DONE store.remove_loader(loader)
# N/A  store.lock()
# N/A  store.unlock()
# DONE store.get_sina_store()
# DONE store.get_sina_records()
# N/A  store.close()
# N/A  store.delete_all_contents(force="")
# DONE store.save_loader(loader)
# DONE store.get_record(Id)
# DONE store.delete(Id)
# DONE store.create_ensemble(name="Unnamed Ensemble", id=None, metadata={}, schema=None, **kargs)
# DONE store.create(name="Unnamed Dataset", id=None, metadata={}, schema=None, sina_type=None, **kargs)
# DONE store.open(Id, loader=None, requestorId=None, *args, **kargs)
# DONE store.get(Id, feature, format=None, loader=None, transformers=[], requestorId=None, *args, **kargs)
# N/A  store.search(*atts, **keys)
# DONE store.find_ensembles(*atts, **keys)
# DONE store.find(*atts, **keys)
# DONE store.check_sync_conflicts(keys)
# DONE store.is_synchronous()
# DONE store.synchronous(mode=None)
# DONE store.sync(keys=None)
# DONE store.add_user(username, groups=[])
# DONE store.add_group(group)
# DONE store.add_user_to_group(username, groups)
# DONE store.export_dataset(datasets, file=None)
# DONE store.import_dataset(datasets, match_attributes=["name", ], merge_handler=None, merge_handler_kargs={},
#                           skip_sina_record_sections=[], ingest_funcs=None)
# DONE store.reassociate(target, source=None, absolute_path=True)
# DONE store.cleanup_files(dry_run=False, interactive=False, clean_fastsha=False, **dataset_search_keys)
# DONE store.check_integrity()
# DONE store.associate(store, reciprocal=False)
# DONE store.dissociate(store, reciprocal=False)
# DONE store.get_associated_store(uri)
# DONE store.get_associated_stores(uris=True)
# DONE store.mv(src, dst, stores=[], destination_stores=[], dataset_record_type="dataset",
#               dataset_matching_attributes=['name', ], version=False, merge_strategy="conservative", mk_dirs=False)
# DONE store.cp(src, dst, stores=[], destination_stores=[], dataset_record_type="dataset",
#               dataset_matching_attributes=['name', ], version=False, merge_strategy="conservative", mk_dirs=False)
# DONE store.tar(tar_file, tar_opts, src="", tar_type="tar", stores=[], dataset_record_type="dataset",
#                no_absolute_path=False, dataset_matching_attributes=["name", ], merge_strategy="conservative")
# DONE store.to_dataframe(data_columns=[], *atts, **keys)

###################
#     Dataset     #
###################

# DONE ds.cleanup_files(dry_run=False, interactive=False, clean_fastsha=False, **search_keys)
# DONE ds.check_integrity()
# DONE ds.open(Id=None, loader=None, *args, **kargs)
# DONE ds.list_features(Id=None, loader=None, use_cache=False, verbose=False, *args, **kargs)
# DONE ds.list_attributes()
# DONE ds.get_execution_graph(feature=None, Id=None, loader=None, transformers=[], *args, **kargs)
# DONE ds.get(feature=None, format=None, Id=None, loader=None, group=False, transformers=[], *args, **kargs)
# DONE ds.reassociate(target, source=None, absolute_path=True)
# DONE ds.validate()
# DONE ds.searchable_source_attributes()
# DONE ds.describe_feature(feature, Id=None, **kargs):
# DONE ds.dissociate(uri, absolute_path=True)
# DONE ds.associate(uri, mime_type, metadata={}, id_only=True, long_sha=False, absolute_path=True, loader_kwargs=None)
# N/A  ds.search(*atts, **keys)
# DONE ds.find(*atts, **keys)
# DONE ds.export(file=None, sina_record=False)
# DONE ds.get_associated_data(ids_only=False)
# DONE ds.is_member_of(ensemble)
# DONE ds.get_ensembles(ids_only=False)
# DONE ds.leave_ensemble( ensemble)
# DONE ds.join_ensemble(ensemble)
# DONE ds.clone(preserve_ensembles_memberships=False, id_only=False)
# DONE ds.is_ensemble_attribute(attribute, ensembles=None, ensemble_id=False)
# Done ds.add_curve(curve, curve_set=None, curve_name=None, independent=None, units=None, tags=None)
# DONE ds.remove_curve_or_curve_set(curve, curve_set=None)
# DONE ds.to_dataframe(data_columns=[], *atts, **keys)

####################
#     Ensemble     #
####################

# DONE ens.cleanup_files(dry_run=False, interactive=False, **search_keys)
# DONE ens.export(file=None)
# DONE ens.create(name="Unnamed Dataset", id=None, metadata={}, schema=None, sina_type=None, **kargs)
# DONE ens.add(dataset)
# DONE ens.remove(dataset)
# DONE ens.get_members(ids_only=False)
# DONE ens.find_datasets(*atts, **keys)
# N/A  ens.clone(*atts, **keys)
# DONE ens.list_attributes(dictionary=False, no_duplicate=False)


def create_csv(file_name):
    headers = ['Column1', 'Column2', 'Column3']

    data = list()

    for i in range(100):
        data.append({'Column1': random.randint(-100000, 100000),
                     'Column2': f'test_{random.randint(-100000, 100000)}',
                     'Column3': f'kosh_{random.randint(-100000, 100000)}'})

    with open(file_name, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)


print('##### Initial Datasets #####')
for i in range(args.datasets):
    i_start = datetime.now()
    print(f'Initial Datasets Start i: {i+1} of {args.datasets}')

    metadata = {f"dataset_att_{ia}": f"run_{args.run_number}_attributes_{ia}" for ia in range(args.datasets)}
    LOGGER.info(msg='kosh_gpu.py: ds = store.create()')
    ds = store.create(id=f"run_{args.run_number}_dataset_{i}",
                      metadata=metadata)

    LOGGER.info(msg='getattr(ds, "test", -1)')
    test = getattr(ds, 'test', -1)

    for j in range(args.datasets):
        j_start = datetime.now()
        print(f'\tInitial Datasets Start j: {j+1} of {args.datasets}')

        file_name = f"run_{args.run_number}_dataset_{i}_csv_{j}.csv"
        create_csv(file_name=file_name)
        LOGGER.info(msg='kosh_gpu.py: ds.associate()')
        ds.associate(file_name, mime_type="pandas/csv")

        print(f'\tInitial Datasets End j: {j+1} of {args.datasets}: {datetime.now() - j_start}')

    print(f'Initial Datasets End i: {i+1} of {args.datasets}: {datetime.now() - i_start}')


class FooLoader(kosh.KoshLoader):

    def extract(self):
        return []

    def list_features(self):
        return []


print('##### Ensembles and Datasets #####')
for i in range(args.ensembles):
    i_start = datetime.now()
    print(f'Ensembles and Datasets Start i: {i+1} of {args.ensembles}')

    my_store_path = f"run_{args.run_number}_store_{i}.sql"
    my_store = kosh.connect(my_store_path)
    LOGGER.info(msg='kosh_gpu.py: store.associate()')
    store.associate(my_store)

    metadata = {f"ensemble_att_{ia}": f"run_{args.run_number}_ensemble_{i}_attributes_{ia}"
                for ia in range(args.datasets)}
    ens_ID = f"run_{args.run_number}_ensemble_{i}"
    LOGGER.info(msg='kosh_gpu.py: ens = store.create_ensemble()')
    ens = store.create_ensemble(id=ens_ID, metadata=metadata)

    my_loader = type(f"run_{args.run_number}_loader_{i}", (FooLoader, ), {})
    LOGGER.info(msg='kosh_gpu.py: store.add_loader()')
    store.add_loader(my_loader)
    LOGGER.info(msg='kosh_gpu.py: store.delete_loader()')
    store.delete_loader(my_loader)

    my_loader_2 = type(f"run_{args.run_number}_loader_{i}_2", (FooLoader, ), {})
    globals()[f"run_{args.run_number}_loader_{i}_2"] = my_loader_2
    LOGGER.info(msg='kosh_gpu.py: store.save_loader()')
    store.save_loader(my_loader_2)
    LOGGER.info(msg='kosh_gpu.py: store.remove_loader()')
    store.remove_loader(my_loader_2)

    for icsv in range(args.datasets):
        file_name = f"run_{args.run_number}_ensemble_{i}_csv_{icsv}.csv"
        create_csv(file_name=file_name)
        LOGGER.info(msg='kosh_gpu.py: ens.associate()')
        ens.associate(file_name, mime_type="pandas/csv")

    for j in range(args.datasets):
        j_start = datetime.now()
        print(f'\tEnsembles and Datasets Start j: {j+1} of {args.datasets}')

        metadata = {f"dataset_att_{ia}": f"run_{args.run_number}_attributes_{ia}" for ia in range(args.datasets)}
        ds_ID = f"run_{args.run_number}_ensemble_{i}_dataset_{j}"
        LOGGER.info(msg='kosh_gpu.py: ds = ens.create()')
        ds = ens.create(id=ds_ID, metadata=metadata)

        for k in range(args.datasets):
            k_start = datetime.now()
            print(f'\t\tEnsembles and Datasets Start k: {k+1} of {args.datasets}')

            file_name = f"run_{args.run_number}_ensemble_{i}_dataset_{j}_csv_{k}.csv"
            create_csv(file_name=file_name)
            LOGGER.info(msg='kosh_gpu.py: temp_associate_id = ds.associate()')
            temp_associate_id = ds.associate(file_name, mime_type="pandas/csv",
                                             metadata={'file_test_level': 'dataset'})
            for col in ['Column1', 'Column2', 'Column3']:
                LOGGER.info(msg='kosh_gpu.py: describe_feature = ds.describe_feature()')
                describe_feature = ds.describe_feature(col, Id=temp_associate_id)
            new_file_name = file_name.replace(".csv", "_moved.csv")
            os.rename(file_name, new_file_name)  # move manually
            LOGGER.info(msg='kosh_gpu.py: ds.reassociate()')
            ds.reassociate(new_file_name, file_name)
            LOGGER.info(msg='kosh_gpu.py: ds.dissociate()')
            ds.dissociate(file_name)
            new_file_name_again = file_name.replace(".csv", "_moved_again.csv")
            os.rename(new_file_name, new_file_name_again)  # move manually
            LOGGER.info(msg='kosh_gpu.py: store.reassociate()')
            store.reassociate(new_file_name_again, new_file_name)
            LOGGER.info(msg='kosh_gpu.py: ds.dissociate()')
            ds.dissociate(new_file_name)

            print(f'\t\tEnsembles and Datasets End k: {k+1} of {args.datasets}: {datetime.now() - k_start}')

        LOGGER.info(msg='kosh_gpu.py: associated_files = list(ds.find())')
        associated_files = list(ds.find(data={'file_test_level': 'dataset'}))
        LOGGER.info(msg='kosh_gpu.py: all_data = ds.get()')
        all_data = ds.get()
        LOGGER.info(msg='kosh_gpu.py: all_data2 = ds.get_execution_graph()')
        all_data2 = ds.get_execution_graph()
        LOGGER.info(msg='kosh_gpu.py: csv_file = ds.open()')
        csv_file = ds.open(Id=associated_files[0].id)
        LOGGER.info(msg='kosh_gpu.py: all_associated_data = ds.get_associated_data()')
        all_associated_data = ds.get_associated_data()
        LOGGER.info(msg='kosh_gpu.py: ds.validate()')
        ds.validate()
        LOGGER.info(msg='kosh_gpu.py: associated_attributes = ds.searchable_source_attributes()')
        associated_attributes = ds.searchable_source_attributes()
        LOGGER.info(msg='kosh_gpu.py: conflicts = store.check_sync_conflicts()')
        conflicts = store.check_sync_conflicts(ds.id)
        LOGGER.info(msg='kosh_gpu.py: store.sync()')
        store.sync(ds.id)
        LOGGER.info(msg='kosh_gpu.py: ds_features = ds.list_features()')
        ds_features = ds.list_features()
        LOGGER.info(msg='kosh_gpu.py: ds_attributes = ds.list_attributes()')
        ds_attributes = ds.list_attributes()
        LOGGER.info(msg='kosh_gpu.py: ds_df = ds.to_dataframe()')
        ds_df = ds.to_dataframe()
        LOGGER.info(msg='kosh_gpu.py: current_dataset = store.open()')
        current_dataset = store.open(associated_files[0].id)
        LOGGER.info(msg='kosh_gpu.py: ds_ensembles = ds.get_ensembles()')
        ds_ensembles = ds.get_ensembles()
        LOGGER.info(msg='kosh_gpu.py: ds_is_member = ds.is_member_of()')
        ds_is_member = ds.is_member_of(ens)
        LOGGER.info(msg='kosh_gpu.py: store.export_dataset()')
        store.export_dataset(ds, file=f"{os.path.basename(associated_files[0].uri)}.json")
        LOGGER.info(msg='kosh_gpu.py: ds.is_ensemble_attribute()')
        ds.is_ensemble_attribute(f"run_{args.run_number}_attributes_0")
        LOGGER.info(msg='kosh_gpu.py: ds_uris_to_cleanup = ds.cleanup_files()')
        ds_uris_to_cleanup = ds.cleanup_files()
        LOGGER.info(msg='kosh_gpu.py: ds_uris_to_cleanup2 = ds.check_integrity()')
        ds_uris_to_cleanup2 = ds.check_integrity()
        LOGGER.info(msg='kosh_gpu.py: ds_export = ds.export()')
        ds_export = ds.export(f"{os.path.basename(associated_files[0].uri)}_2.json")

        print(f'\tEnsembles and Datasets End j: {j+1} of {args.datasets}: {datetime.now() - j_start}')

    LOGGER.info(msg='kosh_gpu.py: current_ensemble = store.open()')
    current_ensemble = store.open(ens_ID)
    LOGGER.info(msg='kosh_gpu.py: ens_uris_to_cleanup = current_ensemble.cleanup_files()')
    ens_uris_to_cleanup = current_ensemble.cleanup_files()
    LOGGER.info(msg='kosh_gpu.py: ens_export = current_ensemble.export()')
    ens_export = current_ensemble.export(f"{ens_ID}.json")
    LOGGER.info(msg='kosh_gpu.py: ens_get_members = current_ensemble.get_members()')
    ens_get_members = current_ensemble.get_members()
    LOGGER.info(msg='kosh_gpu.py: ens_datatsets = current_ensemble.find_datasets()')
    ens_datatsets = current_ensemble.find_datasets()
    LOGGER.info(msg='kosh_gpu.py: ens_attributes = current_ensemble.list_attributes()')
    ens_attributes = current_ensemble.list_attributes()

    print(f'Ensembles and Datasets End i: {i+1} of {args.ensembles}: {datetime.now() - i_start}')

LOGGER.info(msg='kosh_gpu.py: associated_store = store.get_associated_store()')
associated_store = store.get_associated_store(my_store_path)
LOGGER.info(msg='kosh_gpu.py: store.dissociate()')
store.dissociate(my_store)
LOGGER.info(msg='kosh_gpu.py: associated_stores = store.get_associated_stores()')
associated_stores = store.get_associated_stores()

for i in range(args.ensembles):
    group = f"run_{args.run_number}_ensemble_group_{i}"
    LOGGER.info(msg='kosh_gpu.py: store.add_group()')
    store.add_group(group)
    for j in range(args.datasets):
        user = f"run_{args.run_number}_group_{i}_user_{j}"
        LOGGER.info(msg='kosh_gpu.py: store.add_user()')
        store.add_user(user)
        LOGGER.info(msg='kosh_gpu.py: store.add_user_to_group()')
        store.add_user_to_group(user, [group])

tests_path = os.path.dirname(__file__)

print('##### Records #####')
for i in range(args.ensembles):
    i_start = datetime.now()
    print(f'Records Start i: {i+1} of {args.ensembles}')

    ens_ID = f"run_{args.run_number}_ensemble_{i}"

    for j in range(args.datasets):
        j_start = datetime.now()
        print(f'\tRecords Start j: {j+1} of {args.datasets}')

        ID = f"run_{args.run_number}_my_record_ensemble_{i}_dataset_{j}"
        simple_record_structure = {
            "id": ID,
            "type": "breakfast_sim_run",
            "data": {"egg_count": {"value": 10},
                     "omelette_count": {"value": 3},
                     "flavor": {"value": "tasty!"}}
        }
        simple_record = generate_record_from_json(simple_record_structure)
        simple_curve_set = simple_record.add_curve_set("sample_curves")
        simple_curve_set.add_independent("time", list(range((args.datasets))))
        for k in range(args.datasets):
            simple_curve_set.add_dependent(f"amount_{k}", list(range((args.datasets))))

        file_name = f"{ID}.json"
        simple_record.to_file(file_name)
        LOGGER.info(msg='kosh_gpu.py: ds = store.import_dataset()')
        ds = store.import_dataset(file_name, match_attributes=['id'])[0]
        LOGGER.info(msg='kosh_gpu.py: ds2 = ds.clone()')
        ds2 = ds.clone()
        LOGGER.info(msg='kosh_gpu.py: current_ensemble = store.open()')
        current_ensemble = store.open(ens_ID)
        LOGGER.info(msg='kosh_gpu.py: current_ensemble.add()')
        current_ensemble.add(ds)
        LOGGER.info(msg='kosh_gpu.py: current_ensemble.remove()')
        current_ensemble.remove(ds)
        LOGGER.info(msg='kosh_gpu.py: ds.join_ensemble()')
        ds.join_ensemble(current_ensemble)
        LOGGER.info(msg='kosh_gpu.py: ds.leave_ensemble()')
        ds.leave_ensemble(current_ensemble)

        for k in range(args.datasets):
            LOGGER.info(msg='kosh_gpu.py: ds.add_curve()')
            ds.add_curve(list(range((args.datasets))), curve_name=f"add_curve_{k}", curve_set=f"curve_set_{k}")

        LOGGER.info(msg='kosh_gpu.py: store.cp()')
        store.cp(file_name,
                 file_name.replace(".json", "_copy.json"))

        LOGGER.info(msg='kosh_gpu.py: store.mv()')
        store.mv(file_name.replace(".json", "_copy.json"),
                 os.path.join(ens_ID, file_name.replace(".json", "_copy_2.json")), mk_dirs=True)

        LOGGER.info(msg='kosh_gpu.py: current_record = store.get_record()')
        current_record = store.get_record(ID)

        for k in range(args.datasets):
            LOGGER.info(msg='kosh_gpu.py: feature = store.get()')
            feature = store.get(ds.id, f"curve_set_{k}/add_curve_{k}")

        for k in range(args.datasets):
            LOGGER.info(msg='kosh_gpu.py: ds.remove_curve_or_curve_set()')
            ds.remove_curve_or_curve_set(f"add_curve_{k}", curve_set=f"curve_set_{k}")

        print(f'\tRecords End j: {j+1} of {args.datasets}: {datetime.now() - j_start}')

    LOGGER.info(msg='kosh_gpu.py: store.tar()')
    store.tar(f"{ens_ID}.tar", "-c", src=os.path.abspath(ens_ID))
    LOGGER.info(msg='kosh_gpu.py: store.delete()')
    store.delete(ds.id)
    # second delete is used to test if lock strategies get stuck in loop
    # LOGGER.info(msg='kosh_gpu.py: store.delete()')
    # store.delete(ds.id)

    print(f'Records End i: {i+1} of {args.ensembles}: {datetime.now() - i_start}')

print("Final Store Level Methods")
# LOGGER.info(msg='kosh_gpu.py: uris_to_cleanup = store.cleanup_files()')
# uris_to_cleanup = store.cleanup_files()
# LOGGER.info(msg='kosh_gpu.py: uris_to_cleanup2 = store.check_integrity()')
# uris_to_cleanup2 = store.check_integrity()
LOGGER.info(msg='kosh_gpu.py: is_synchronous = store.is_synchronous()')
is_synchronous = store.is_synchronous()
# LOGGER.info(msg='kosh_gpu.py: store.synchronous()')
# store.synchronous()
LOGGER.info(msg='kosh_gpu.py: all_ensembles = store.find_ensembles()')
all_ensembles = store.find_ensembles()
LOGGER.info(msg='kosh_gpu.py: all_datasets = store.find()')
all_datasets = store.find()
# LOGGER.info(msg='kosh_gpu.py: df_store = store.to_dataframe()')
# df_store = store.to_dataframe()
LOGGER.info(msg='kosh_gpu.py: sina_store = store.get_sina_store()')
sina_store = store.get_sina_store()
LOGGER.info(msg='kosh_gpu.py: sina_records = store.get_sina_records()')
sina_records = store.get_sina_records()

print(f'Total Time to complete: {datetime.now()-start}')

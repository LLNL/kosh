---
name: kosh
description: Use Kosh to persist results and files, query them later, and
  build custom loaders, transformers, and operators for data extraction and
  slice-aware processing.
---

# Kosh Skill

Use this skill when an agent needs to store results, attach files, query them
later, or build lightweight data-processing hooks around files and datasets.

## Common Patterns
- Create or open a store with `kosh.connect(...)` or `kosh.utils.create_new_db(...)`.
- Create a dataset with `store.create(...)`, then store metadata and attach
  files with `dataset.associate(uri, mime_type, metadata=...)`.
- Create an ensemble with `store.create_ensemble(...)` when a group of datasets should be searched together.
- Query later with `store.find(...)` for datasets, `store.find_ensembles(...)`
  for ensembles, `dataset.find(...)` for associated sources,
  `ensemble.find_datasets(...)` for member datasets, plus
  `dataset.list_features()` and `dataset.describe_feature(...)`.
- Read results with `dataset.get(feature, transformers=[...])` or
  `dataset.get_execution_graph(feature, transformers=[...])`.

## Storing Results
- Use datasets to hold metadata about a run, a job, or a generated artifact.
- Associate output files directly to the dataset so they can be rediscovered by feature name or metadata.
- Use ensembles to collect related runs and add ensemble-level metadata for the group.
- Keep a given attribute in one place: put dataset-specific values on the dataset and group-level values on the ensemble.
- Use `ensemble_tags` for labels that belong to an ensemble membership, not the dataset itself. These tags are searchable with `ensemble.find_datasets(ensemble_tags=...)` and can be inspected with `dataset.list_ensemble_tags(...)`.
- Example: `ensemble_tags={"even_or_odd": "even", "data_type": "test data"}`.

## Workflow Tracking
- Use the workflow CLI (`kosh_workflow`) when you want to create or update datasets from a step plus parameter set.
- Use `kosh.parameter_store.StepRequest` and `kosh.parameter_store.apply_step(...)` from Python when scripting workflow updates.
- Treat `workflow_step` as the default step field unless a different field name is configured with `--step-field`.
- Use `--ensemble` to scope workflow operations to one ensemble, `--param key=value` or implicit `--key=value` filters to match init parameters, and `--upsert-init` when the init step should reuse an existing dataset instead of creating a duplicate.
- Use `--meta` for dataset metadata updates and `--associate` for files that belong on the dataset; use `--emeta`/`--ensemble-meta` and `--associate-ensemble` for ensemble-level updates.

## Parsing Output Files
- Add a custom loader when Kosh needs to parse a file format that is not
  already supported or when the file should expose custom features.
- Implement `list_features()` and `extract()` on a `KoshLoader`.
- A loader can expose multiple features and return data in different formats,
  so use it when the raw file needs parsing or format-specific access.
- Register the loader with `store.add_loader(MyLoader)` before querying
  associated files that depend on it.
- For loaders that support direct indexing, implement `__getitem__` so slices
  can be served efficiently.

## Feature Aliases
- Use `alias_feature` when different sources expose the same data under
  different names, such as uppercase CSV headers versus lowercase HDF5
  datasets.
- Map aliases both ways when needed, for example `{"col_a": "COL_A"}` on one
  dataset and `{"COL_A": "col_a"}` on the other.
- Aliases let `dataset["col_a"]` or `dataset["COL_A"]` resolve to the same
  underlying feature when the exact name is not present.

## Transformers and Operators
- Use `KoshTransformer` or `@kosh.numpy_transformer` /
  `@kosh.typed_transformer` to reshape, switch formats, or post-process one
  input stream.
- Use `KoshOperator` or `@kosh.numpy_operator` / `@kosh.typed_operator` to
  combine multiple inputs, including data loaded from different sources or
  formats.
- Keep transforms small and composable; use them to normalize or convert
  results after loading, not to replace the loader’s parsing job.

## Slicing and Propagation
- Kosh uses `__getitem__` for indexing.
- Implement `__getitem_propagate__(self, key, input_index)` in transformers or
  operators when a slice should be pushed back toward the loader.
- If propagation is not possible, return `None` and let Kosh apply the slice after the upstream step runs.

## Notebook References
- Store creation and metadata:
  `examples/Example_00_Open_Store_And_Add_Datasets.ipynb`,
  `Example_03_Working_with_Datasets.ipynb`
- Reading data and feature access: `examples/Example_02_Read_Data.ipynb`, `Example_column_based_text_files.ipynb`
- Ensembles and workflow tracking:
  `examples/Example_Ensembles.ipynb`, `Example_Simulation_Workflow.ipynb`,
  `Example_Workflow_Manager.ipynb`
- Custom loaders: `examples/Example_Custom_Loader.ipynb`, `Example_MNIST.ipynb`, `Example_Sidre.ipynb`
- Transformers and operators:
  `examples/Example_05a_Transformers.ipynb`,
  `Example_05b_Transformers-SKL.ipynb`, `Example_06_Operators.ipynb`
- Advanced slicing and propagation: `examples/Example_Advanced_Data_Slicing.ipynb`
- Data movement and interoperability:
  `examples/Example_07_Transferring_Datasets.ipynb`,
  `Example_Moving_Datasets.ipynb`, `Kosh_and_Sina_Interoperability.ipynb`

## Practical Notes
- Prefer attaching metadata to the dataset or ensemble instead of inventing a separate bookkeeping layer.
- Use existing file-backed loaders when possible; add a custom loader only
  when the built-ins do not expose the data cleanly.
- Keep notebook-style examples aligned with the existing `Example_*.ipynb` naming pattern.

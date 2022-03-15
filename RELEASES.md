# Release Notes

* [2.0](#2.0)
* [1.2](#1.2)
* [1.1](#1.1)
* [1.0](#1.0)
* [0.9](#0.9)
* [0.8](#0.8)

## 2.1 Release

### Description

This is a mainly maintenance release introducing a few new features.

### New in this release

* Decorators for transformers and operators
  * Regular `def foo(...):` can now be converted to transformers or operators via decorators. e.g for a numpy transformer: `@kosh.numpy_transformer`. See the [transformer](examples/Example_05a_Transformers.ipynb) and [operators](examples/Example_06_Operators.ipynb) notebooks for more details.
* Introducing a loader for text and column-based data, based on top `numpy.loadtxt`. See [this](examples/Example_column_based_text_files.ipynb) notebook.
* When cloning a dataset one can choose to preserve ensemble memberships (`preserve_ensembles_memberships=True`), simply copy over these attributes (`preserve_ensembles_memberships=False`) (**default**) or ignore information related to ensembles (`preserve_ensembles_memberships=-1`)
* Dataset objects now have a `is_ensemble_attribute()` function to know if an attribute belongs to an ensemble.
* Added `list_attributes()` function to ensemble objects.

### Improvements

* When printing a dataset, the attributes coming from ensembles the dataset belongs to are listed in a separate section.

### Bug fixes

* When cloning a dataset, an artificial `id` attribute was created with the original dataset `id` in it.
* Setting an attribute to an invalid value would cause the dataset to disappear from the store.

## 2.0 Release

### Description

This release aligns Kosh with Sina and makes it the only backend. Kosh and Sina (1.11) API's have been mostly aligned.
Sina curve and file section can now be recognized and taken advantage by Kosh.

### New in this release

* Sina alignment:
  * Sina is only supported backend, no more code to potentially support other backends
  * curves appear as associated
  * files with `mimetype` appear as virtual associated files
  * Kosh exported json files are sina-compatible and Kosh can ingest Sina's json files
  * Stores are now opened via: `kosh.connect(...)`
  * `search(...)` is now  `find(...)`
  * any non Kosh-reserved record type is considered a dataset
* `find` functions return generators (used to be lists)
* Support for ensembles
* Kosh stores can be associated with other Kosh stores.
* `kosh` command line:
  * can create stores
  * can add datasets
  * can use htar to tar up data
* datasets can be cloned
* While importing a dataset into a store, there are now options to handle conflicts.
* Loader for file saved by numpy (`.npy`)
* Store can fix changed/updated fast_sha

### Improvements

* Do not try to import external Python packages until needed -> some loader might appear as valid even though python packages are missing.
* versioning is now pip compatible
* `import_dataset(...)` can import list of datasets
* Added `verbose` argument to transformers and operators -> this will let the user know when retrieving data from cache
* set user name to "default" if can't get it from USER env var

### Bug fixes

* `matplotlib` import would crash if no DISPLAY environment variable
* hdf5 leading / fix
* No more error if a known mime type points to missing file (`list_features` will not show it)
* dissociate files from store after moving them


## 1.2 Release

### Description

This release is fully backward compatible but introduces new concepts.
Operators are introduced allowing the composition of features from one or many sources.
Feature selection without extraction is now possible via the new execution graphs introduced in this release.
Execution graphs are the recommended way to use Kosh going forward, as reflected in the updated notebooks.


### New in this release

* Operators: Compose multiple features (and their transformers).
* Execution graph concept (select and compose features before executing).
* New Conduit's Sidre Mesh Blueprint field loader.

## Deprecation Warning

* In future versions (not 1.2) the `search` functions will return a generator (they are currently returning a list). In this version a warning is issued when you use the `search` function.

### Improvements

* Multiple speed/caching optimizations.
* A cleanup function helps you clean your store from files that no longer exist.
* Cleaned up tables in documentation.
* Transformers get a `parent` attribute, allowing you to access its caller in the `transform` function.

### Bug fixes

* In some case where a feature was available from the loader but not listed, Kosh would let the user access it. With this bug fix, the loader's `list_features` must be fixed first in order to access the feature. This bug affected groups in HDF5 files. Groups were not listed but still accessible, HDF5 loader now lists `groups` as actual features.
* If a loader needed matplotlib and no X connection was available the import of `matplotlib.pyplot` would lead to an uncatchable error. We now check for a valid backend first (via environment's `DISPLAY` variable).

## 1.1 Release

### Description

This is a maintenance release, with a few small bugs fixes and optimizations.
A new object to help you drive existing scripts from searches in Kosh has been introduced.


### New in this release

* New script wrapper object allows you to drive existing command line based scripts using Kosh objects.
* New loader: `json` files.
* Transformers `cache` option now accepts `2` as a value allowing to clobber existing cache files.
* Datasets have new `searchable_source_attributes` to list attributes that can be used to filter sources associated with this dataset.
* Can search for attribute presence (no need to specify a value).

### Improvements

* Dataset export/import optimized, `import_dataset` now accepts dataset objects.
* Warning raised if trying to associate already associated source: metadata will not be automatically updated on source.

### Bug fixes

* `list_features` cache issue cleaned up.
* Respects user passed loader.

### In the weeds

* Logos are automatically generated to reflect version.
* Uses sbang to allow executable to work in environments with very long paths.

## 1.0 Release

* Transformers are now available. These allow for post processing when extracting features.
* Transformers allow for cache
* Kosh can now move, copy or tar files tied to (a) store(s) and update store(s) appropriately.
* Documentation cleanup, logo added.
* `list_features` operation is now cached for speed improvements
* ***lock** capability improved to allow for multiple users accessing the store at same time.

## 0.9 Release

Bug fix and optimization release

* possibility to "update" multiple attributes of a dataset at once, faster and less db access
* possibility to associate many files at once, faster, less db access
* creating a new store, returns an handle to that store
* switched to pytest rather than nosetests
* Added Ultra files loader
* loaders can be saved in store.
* when many uri were associated with a dataset and many metrics had the same name, it was not necessarily returning the desired metric even when asking for one by it's full path.

## 0.8 Release

* Command line tools (kosh program)
* Significant boost in search speed 
* Python 2 support
* Example Notebooks are self contained
* Version obtained via git
* Associated uri objects accept arguments to `open` function
* User id no longer required when opening the store (defaults to UNIX username)
* sina type for *datasets* does not need to be `dataset` any longer.
* Can open any sina db, not necessarily Kosh generated one.

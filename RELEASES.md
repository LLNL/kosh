# Release Notes

* [1.2](#1.2)
* [1.1](#1.1)
* [1.0](#1.0)
* [0.9](#0.9)
* [0.8](#0.8)

## 1.2 Release

### Description

This release is fully backward compatible but introducesnew conceptss.
Operators are introduced allowing to compose features from one or many sources.
Feature selection w/o extraction is now possible via the new execution graphs introduced in this release.
Execution graphs are the recommended way to use Kosh going forward as reflectd in the updated notebooks.


### New in this release

* Execution graph concept (select and compose features before executing)
* operators: Compose multiple features (and their transformers)
* Conduit's Sidre Meshblueprint field loader added.

### Improvements

* Multiple speed/caching optimizations
* A cleanup function helps you clean your store from dead files
* Cleaned up tables in documentation
* transformers get a `parent` attribute allowing you to access its caller in the `transform` function.

### Bug fixes

* Some edge case where a feature was available to the loader but not listed, Kosh would let the user access it. With this bug fix, the loader's `list_features` must be fixed. The affected groups in HDF5 files. Groups are now listed as actual features.
* If a loader needs matplotlib, if no X connection was available pydv import of mtplotlib would lead to a seg fault. We now check for a valid backend first (via environment's `DISPLAY` variable)

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

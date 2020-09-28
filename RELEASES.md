# Release Notes

* [0.9](#0.9)
* [0.8](#0.8)

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

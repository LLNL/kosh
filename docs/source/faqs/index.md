# Frequently Asked Questions

## Can I make Kosh faster?

Yes. By default Kosh runs in "safe" mode, synchronizing every changes to the store

You can turn this off and synchronize only when you're done working:

```python
import kosh
store = kosh.KoshStore(db_uri="some_kosh_store.sql", sync=False)
# Some Kosh work here
store.sync()
```

If you opened the store in sync mode you can switch back and forth:

```python
import kosh
store = kosh.KoshStore(db_uri="some_kosh_store.sql")

# some synchronous work
store.synchronous(False)
# Some async work (faster)
# now let's go back to synchrouns mode
store.synchronous(True)

# You can switch back and forth w/o passing the mode
mode = store.synchronous()
# Let's query
print("Synchrononus mode?", mode, store.is_synchronous())
```

If you are associating multiple files with a dataset pass them all at once

Rather than
```python
for i in range(200):
    ds.associate(str(i), metadata={"name":str(i)}, mime_type="type_{}".format(i))
```

Consider
```python
ds.associate([str(i) for i in range(200)], metadata = [ {"name":str(i)} for i in range(200)], mime_type=["type_{}".format(i) for i in range(200) ])
```

Similarly, each update to a dataset requires a database access, consider passing as many attributes as possible at creation time, and doing batch update rather than many single operation updates.

Rather than:
```python
ds = store.create()
ds.name = "My name"
ds.attr1 = 1
ds.attr2 = 'two'
[... some code ...]
# updating
ds.new_attr = 'new'
ds.attr1 = 'one'
ds.attr2 = 2
```

Consider
```python
ds.store.create(metadata={"name":"My name", "attr1":1, "attr2":"two"})
[... some code ...]
# batch update/creation
ds.update({"new_attr":"new", "attr1":"one", "attr2": 2})
```

## Can I move files after I associated them with datasets in Kosh?

Yes Kosh offers many option to manipulate the files directly and update your Kosh stores.

Look for `kosh cp`, `kosh mv`, `kosh rm`, `kosh tar`

In particular take a closer look at [this notebook](../jupyter/Example_06_Transfering_Datasets.ipynb)


## What loaders come with Kosh by default?

Currently Kosh comes with the following loaders

```eval_rst

.. list-table::
   :widths: auto

   * - Name
     - Description
     - mime_type
     - out type(s)
     - Required Python Modules
   * - HDF5Loader
     - Loads data from HDF5 format files
     - hdf5
     - numpy
     - h5py
   * - PGMLoader
     - Load pgm formatted images (P2 and P5)
     - pgm
     - numpy
     - None
   * - PILLoader
     - Load images that PIL can read
     - png, pil, tif, tiff, gif, image
     - numpy, raw binary
     - pillow
   * - UltraLoader
     - Loads ultra files
     - ultra
     - numpy
     - pydv
   * - JSONLoader
     - Loads in json files
     - json
     - any, dict, list, str
     - json
```

# What transformers come with Kosh by default?

Currently Kosh provides the following transformers

```eval_rst

.. list-table::
   :widths: auto

   * - Name
     - Description
     - input mime_type
     - output type
     - Required Python Modules
     - External Ref
   * - **Generic Numpy**
     - **Generic Numpy-related transformers**
     -
     -
     -
     -
   * - KoshSimpleNpCache
     - Does nothing to the data simply allows to cache
     - numpy
     - numpy
     -
     -
   * - Take
     - mimics numpy's take function
     - numpy
     - numpy
     -
     -
   * - Delta
     - computes delta between consecutive slice of an array over a specific axis
     - numpy
     - numpy
     -
     -
   * - Shuffle
     - shuffles input array over a specific axis
     - numpy
     - numpy
     -
     -
   * - **SKL**
     - **SKL-based transformers**
     -
     -
     -
     -
   * - Splitter
     - Splits the input into train/test and validation if specified
     - numpy
     - numpy
     - scikit-learn
     -
   * - StandardScaler
     - Returns a skl standard scaler
     - numpy
     - numpy
     - scikit-learn
     -
   * - DBSCAN
     - Returns a skl DBSCAN estimator or results from it
     - numpy
     - numpy, estimator
     - scikit-learn
     -
   * - KMeans
     - Returns a skl KMeans estimator or results from it
     - numpy
     - numpy, estimator
     - scikit-learn
     -
```


# Cache vs Association?

Is it better to cache results from transformers or should I associate the results with the store?

One thing to consider is that associating with the Kosh store usually requires a bit more work, for example making sure all the users of the Kosh store can access the data and the path is going to stay consistent through the whole life of the store.

In general you should re-associate with the store if:

* Many users are expected to use the result from the transformer
* The result will be used for many projects over a long period of time
* The operation takes a long time

You should probably use cache if:

* You're the only user
* The results are only going to be used for one project for a relatively small period of time
* The data is going to be loaded many times in that script or some subprocesses.


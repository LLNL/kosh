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
## Can I gather features with different names from different datasets with aliases?

Yes, using the `alias_feature` parameter. It can be set at the creation of the dataset or later on.

```python
features = dataset.list_features()
print(features)
# ['cycles', 'direction', 'elements', 'node', 'node/metrics_0', 'node/metrics_1', 'node/metrics_10', 'node/metrics_11', 'node/metrics_12', 'node/metrics_2', 'node/metrics_3', 'node/metrics_4', 'node/metrics_5', 'node/metrics_6', 'node/metrics_7', 'node/metrics_8', 'node/metrics_9', 'zone', 'zone/metrics_0', 'zone/metrics_1', 'zone/metrics_2', 'zone/metrics_3', 'zone/metrics_4']

alias_dict = {'param5': 'node/metrics_5',
              'P6': ['node/metrics_6'],
              'P0': 'metrics_0'}
dataset.alias_feature = alias_dict
# This can also be passed in at the creation of the dataset
# dataset = store.create(metadata={'alias_feature': alias_dict})

print(dataset['param5'][:])
# <HDF5 dataset "metrics_5": shape (2, 18), type "<f4">
print(dataset['P6'][:])
# <HDF5 dataset "metrics_6": shape (2, 18), type "<f4">
# print(dataset['P0'][:])  # Cannot uniquely pinpoint P0, could be one of ['node/metrics_0', 'zone/metrics_0']
```

## Can I move files after I associated them with datasets in Kosh?

Yes Kosh offers many option to manipulate the files directly and update your Kosh stores.

Look for `kosh cp`, `kosh mv`, `kosh rm`, `kosh tar`

In particular take a closer look at [this notebook](../jupyter/Example_07_Transferring_Datasets.ipynb)


## What loaders come with Kosh by default?

Currently Kosh comes with the following loaders

```eval_rst

.. list-table::
   :widths: auto
   :header-rows: 1 

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
   * - NpyLoader
     - Loads data saved by numpy in a npy file
     - npy
     - numpy
     - numpy
   * - NumpyTxtLoader
     - Built on top of numpy.loadtxt that can read text files.
     - numpy/txt
     - numpy
     - numpy
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
   * - SidreMeshBlueprintFieldLoader
     - Loads sidre Mesh Blueprint fields
     - sidre/path
     - dict
     - conduit
```

## What transformers come with Kosh by default?

Currently Kosh provides the following transformers

```eval_rst

.. list-table::
   :widths: auto
   :header-rows: 1 

   * - Name
     - Description
     - input mime_type
     - output type
     - Required Python Modules
     - External Ref
   * - **Generic Numpy**
     - *Generic Numpy-related transformers*
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
     - *SKL-based transformers*
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


## Cache vs Association?

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

## BLAS Error?

Are you running into this error?

BLAS : Program is Terminated. Because you tried to allocate too many memory regions.

"BLAS stands for Basic Linear Algebra Subprograms. BLAS provides standard interfaces for linear algebra, including BLAS1 (vector-vector operations), BLAS2 (matrix-vector operations), and BLAS3 (matrix-matrix operations).
As per the documentation, if your application is already multi-threaded, it will conflict with OpenBLAS multi-threading. Therefore, you must set OpenBLAS to use a single thread.
So, it seems that your application is conflicting with OpenBLAS multi-threading. You need to run the followings on the command line and it should fix the error:"


export OMP_NUM_THREADS=1


source: https://github.com/autogluon/autogluon/issues/1020

## I am using MPI should I do anything special?

In general we recommend making write operations on rank 0 only, especially when using sqlite as a backend.

## What about MPI and mariadb?

When opening a mariadb backend, in order to avoid sync error between ranks you should use:

```python
store = kosh.connect(mariadb, execution_options={"isolation_level": "READ COMMITTED"})
```

## I am using a mariadb backend and I want my attributes to be case sensitive

You will need to fix your dtabase collate:

```python
store = kosh.connect(mariadb)
store.get_sina_store()._dao_factory.session.execute("SET NAMES latin1 COLLATE latin1_general_ci")
```

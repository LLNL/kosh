# Users Guide

## Creating a store

You can create your own store to catalog your data by using

```python
import kosh
kosh_example_sql_file = "kosh_example.sql"
kosh.create_new_db(kosh_example_sql_file)
```

## Opening an existing store

Once you have a store you can connect to it

```python
import kosh
kosh_example_sql_file = "kosh_example.sql"
# connect to store
store = KoshStore(engine="sina", username=os.environ["USER"], db='sql', db_uri=kosh_example_sql_file)
```

## Adding datasets to the store

```python
ds = store.create()
```

## Adding attributes to the store

```python
ds.some_metadata = "A simple metadata"
```

## Associating data to a dataset

You can associate data to a dataset, you will need a "URI" to locate the associated data (this can be a file path or inernet address or database name, etc...) and a mimetype describing the data type. Mime-type are used to load the data

```python
ds.associate("myfile.txt", "text")
```

## Reading data

Once data and mimetype have been associated to a dataset you can load these data in your application

```python
features = ds.list_features()
print(features)
data = ds.get(features[0])
```

## Loaders

If multiple loaders are available you can specify the loader you want to use

```python
# Image loader
my_loader = kosh.loader.pil.PILLoader  # no need to instantiate
data = ds.get(features[0], loader=my_loader)
```

## Transformers

Once data is loaded from its source URI you further process it (subsampling, format change, augmentation, etc...) via *transformers*.

Transformers offer the possibility to cache their result for faster computation the next time around. The default cache directory in stored in `kosh.core.kosh_cache_dir` and points to: `os.path.join(os.environ["HOME"], ".cache", "kosh")`.

```python
# no transformation but stores cache as numpy (useful if loader takes a long time to convert to numpy)
my_transformer = kosh.transformers.npy.SimpleNpCache(cache=True, cache_dir="/some/path/to/cache")
data = ds.get(features[0], transformers=[my_transformer, ])
```




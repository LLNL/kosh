# Developers Guide

This document describe necessary implementations for various component of Kosh.



## Store objects

The store object is the entry point to your Kosh implementation.
That is how the user connects to your implementation.

The following function are expected to be implemented.

### connect

This method is meant to allow the end user to connect to the store, it should take all necessary parameters to connect, such as user name, credentials, store url, etc...

### search

This method search the kosh for dataset matching the user's criteria.

It returns a list of *datasets* objects unless `ids_only` was passed as true in which case it returns a list of the unique ids for the matching datasets.

### open

Given a unique id and an optional loader, *opens* the desired object. For example for a text file it would do the equivalent of

Default for opening (if applicable) should be in *read only* mode.

```
f = open("some_path")
``` 


### add_loader

Loaders are described bellow but essentially they are helpers that know how to manipulate Kosh object. This allows for handling various input types

### schema

Schemas are used by end users and programmatic tools to understand how Kosh can be searched.

## Dataset Objects

A *dataset* represents a collection of data related to each other in some ways. Datasets have attributes that help distinguish them from each others and are used by the store `search` function

Dataset object should implement the following functions.

### search

Search associated data with this datasets, for example filter down to files of a certain type only.

### open

Given the associated data unique id, opens it, essentially a shortcut to the store `open` function

### load

A shortcut to the store's `load` function

### add

Given a unique id or a Kosh-understood object, associates this object to the dataset.

## File Objects

File objects are to be associated to datsets, they should have a unique id, and the following two attributes:

`uri`: Describing how to get to the data.
`mime_type`: what kind of data is in this file.

Loader will rely on this information to be accessible in order to deal with file objects.

## Data Object

Data Object are returned by the loader, this allows for a pseudo constant representation accross formats
common attributes are:
`mime_type`: what kind of data

Potentially one can implement storing the DataObject directly in the store, or decide to access them via files.

## Loader Objects

Loader object allow for querying/ingestion of custom data representation and/or custom implementation of such query/ingestion.

A data loader must at a minimum implement the following functions

`list_features`: return a list of available features

`describe_feature`: return a dictionary with info about the feature, such as dimensions, etc...

`extract`: function to extract a given feature (or list of) from the object.
         necessary parameters will be available in loader under self._user_passed_parameters.
         'extract' is called from the loader's get function, AFTER the preprocess function and BEFORE the postprocess function.
         We recommend extract to return a pointer to the data rather than the data itself.
         feature, format and user arguments (args/kargs) are stored on the object and accessible via:
         self.format, self.feature, self._user_passed_parameters
         no arguments are expected

Optional functions that can be implemented are:

preprocess: usually a setup func to stage extract, no arguments are expected, instead access: self._user_passed_parameters
postprocess: a final function to further clean data returned from extract, no arguments are expected, instead access: self._user_passed_parameters

get: can be re-impemented but it is not recommended

## Transformer Objects

Transformer objects allow to further process the data after extraction from the original URI. Transformer can be chained and each step can be cached. The default cache directory in stored in `kosh.core.kosh_cache_dir` and points to: `os.path.join(os.environ["HOME"], ".cache", "kosh")`.

Similarly to loaders, a transformer is expected to define it's accepted input types and the possible output types for each of these.

A transformer is also expected to re-implement the `transform` function. This function takes two arguments `inputs` and `format`

`inputs` is what is returned by the `get` function on a dataset or by the previous *transformer* in the chain.
`format` is the output format that the user desires. When not passed Kosh will try to establish a graph of all possible input/outputs for each transformer and will chose the shortest path fro the extracted data to the final transformer output.

A transformer can also re-implement the `save` and `load` functions which are used to cache the result for faster subsequent loading

The `save` function signature is 
```python
def save(self, cache_file, *things_to_save):
    pass
```

`cache_file` is generated by Kosh and is a unique signature based on input parameters to both the transformer and the inputs/format to the `transform` function.

The `load` function only expects a cache_file name (unique signature generated by Kosh)

```python
def load(self, cache_file)
    return things_loaded
```
# Developpers Guide

This document describe necessary implementations for various component of Kosh.



## Store objects

The store object is the entry point to your Kosh implementation.
That is how the user connects to your implementation.

The following function are expected to be implemented

### connect

This method is meant to allow the end user to connect to the store, it should take all necessary parameters to connect, such as user name, credentials, store url, etc...

### search

this method search the kosh for dataset matching the user's criteria.

It returns a list of *datasets* objects unless `ids_only` was passed as true in which case it returns a list of the unique ids for the matching datasets

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

A *dataset* represents a collection of data related to each other in some ways. Datasets have attributes that help distinguish them from each others and are used by the store ~search` function

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
`mime-type`: what kind of data is in this file.

Loader will rely on this information to be accessible in order to deal with file objects.


## Loader Objects

Loader object allow for rare/custom representation of data and for custom implementation of data access.


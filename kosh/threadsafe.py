import os
from . import threadsafe_decorators
import warnings


warnings.warn(("This version of threadsafe will be depreciated in Kosh 3.2. "
               "Use lock_strategies instead: examples/Example_ThreadSafe.ipynb."))


NUM_TRIES = int(os.environ.get("KOSH_THREADSAFE_NUM_TRIES", 10))
PATIENCE = int(os.environ.get("KOSH_THREADSAFE_PATIENCE", 3))


@threadsafe_decorators.threadsafe_call(NUM_TRIES, PATIENCE)
def safe_open(store, *args, **kw_args):
    """
    A threadsafe version of Kosh's "open" method.

    Parameters:
    -----------
    store: Kosh store
        A kosh store that has the "open" method.

    Returns:
    --------
    object:
        The result of calling store.open(*args, **kw_args)
    """
    return store.open(*args, **kw_args)


@threadsafe_decorators.threadsafe_call(NUM_TRIES, PATIENCE)
def safe_get_execution_graph(dataset, **kw_args):
    """
    A threadsafe version of Kosh's "get_execution_graph" method.

    Parameters:
    -----------
    dataset: Kosh dataset
        A kosh dataset that has the "get_execution_graph" method.

    Returns:
    --------
    object:
        The result of calling dataset.get_execution_graph(**kw_args)
    """
    return dataset.get_execution_graph(**kw_args)


@threadsafe_decorators.threadsafe_call(NUM_TRIES, PATIENCE)
def safe_associate(dataset, **kw_args):
    """
    A threadsafe version of Kosh's "associate" method.

    Parameters:
    -----------
    dataset: Kosh dataset
        A kosh dataset that has the "associate" method.

    Returns:
    --------
    object:
        The result of calling dataset.associate(**kw_args)
    """
    return dataset.associate(**kw_args)


@threadsafe_decorators.threadsafe_call(NUM_TRIES, PATIENCE)
def safe_create(kosh_obj, **kw_args):
    """
    A threadsafe version of Kosh's "create" method.

    Parameters:
    -----------
    kosh_obj: object
        A kosh object that has the "create" method.

    Returns:
    --------
    object:
        The result of calling kosh_obj.create(**kw_args)
    """
    return kosh_obj.create(**kw_args)


@threadsafe_decorators.threadsafe_call(NUM_TRIES, PATIENCE)
def safe_create_ensemble(store, **kw_args):
    """
    A threadsafe version of Kosh's "create_ensemble" method.

    Parameters:
    -----------
    store: Kosh store
        A kosh store that has the "create_ensemble" method.

    Returns:
    --------
    object:
        The result of calling store.create_ensemble(**kw_args)
    """
    return store.create_ensemble(**kw_args)


@threadsafe_decorators.threadsafe_call(NUM_TRIES, PATIENCE)
def safe_delete(kosh_obj, del_object):
    """
    A threadsafe version of Kosh's "delete" method.

    Parameters:
    -----------
    kosh_obj: object
        A kosh object that has the "delete" method.
    del_object: object
        The kosh object to delete.

    Returns:
    --------
    object:
        The result of calling kosh_obj.delete(del_object)
    """
    return kosh_obj.delete(del_object)


@threadsafe_decorators.threadsafe_call(NUM_TRIES, PATIENCE)
def safe_add(kosh_obj, add_obj):
    """
    A threadsafe version of Kosh's "add" method.

    Parameters:
    -----------
    kosh_obj: kosh object
        A kosh object.
    add_obj: kosh object
        A kosh object to add to kosh_obj.

    Returns:
    --------
    object:
        The result of calling kosh_obj.add(add_obj)
    """
    return kosh_obj.add(add_obj)


@threadsafe_decorators.threadsafe_call(NUM_TRIES, PATIENCE)
def safe_clone(kosh_obj, **kw_args):
    """
    A threadsafe version of Kosh's "clone" method.

    Parameters:
    -----------
    kosh_obj: kosh object
        A kosh object.

    Returns:
    --------
    object:
        The result of calling kosh_obj.clone(**kw_args)
    """
    return kosh_obj.clone(**kw_args)


@threadsafe_decorators.patient_find(NUM_TRIES, PATIENCE)
@threadsafe_decorators.threadsafe_call(NUM_TRIES, PATIENCE)
def safe_find_datasets(ensemble, **kw_args):
    """
    A threadsafe version of Kosh's "find_datasets" method.

    IMPORTANT: Kosh's "find" methods return generators that don't
    perform database communications until the generator is executed.
    This means that we need to execute the generator inside of
    our threadsafe wrapper. This converts the return value to
    an iterator to conserve the return value interface. Note that
    this will have a small impact on performance, but it shouldn't
    be significant.

    Parameters:
    -----------
    ensemble: kosh ensemble
        A kosh ensemble.

    Returns:
    --------
    object:
        The result of calling ensemble.find_datasets(**kw_args)
    """
    return iter(list(ensemble.find_datasets(**kw_args)))


@threadsafe_decorators.threadsafe_call(NUM_TRIES, PATIENCE)
def safe_kosh_get(kosh_obj, *args, **kw_args):
    """
    A threadsafe version of Kosh's "get" method.

    Parameters:
    -----------
    kosh_obj: kosh object
        A kosh object.

    Returns:
    --------
    object:
        The result of calling kosh_obj.get(**kw_args)
    """
    return kosh_obj.get(*args, **kw_args)


@threadsafe_decorators.patient_find(NUM_TRIES, PATIENCE)
@threadsafe_decorators.threadsafe_call(NUM_TRIES, PATIENCE)
def safe_find_ensembles(store, **kw_args):
    """
    A threadsafe version of Kosh's "find_ensembles" method.

    IMPORTANT: Kosh's "find" methods return generators that don't
    perform database communications until the generator is executed.
    This means that we need to execute the generator inside of
    our threadsafe wrapper. This converts the return value to
    an iterator to conserve the return value interface. Note that
    this will have a small impact on performance, but it shouldn't
    be significant.

    Parameters:
    -----------
    store: kosh store
        A kosh store.

    Returns:
    --------
    object:
        The result of calling store.find_ensembles(**kw_args)
    """
    return iter(list(store.find_ensembles(**kw_args)))


@threadsafe_decorators.patient_find(NUM_TRIES, PATIENCE)
@threadsafe_decorators.threadsafe_call(NUM_TRIES, PATIENCE)
def safe_find(kosh_obj, **kw_args):
    """
    A threadsafe version of Kosh's "find" method.

    IMPORTANT: Kosh's "find" methods return generators that don't
    perform database communications until the generator is executed.
    This means that we need to execute the generator inside of
    our threadsafe wrapper. This converts the return value to
    an iterator to conserve the return value interface. Note that
    this will have a small impact on performance, but it shouldn't
    be significant.

    Parameters:
    -----------
    kosh_obj: kosh object
        A kosh object.

    Returns:
    --------
    object:
        The result of calling kosh_obj.find(**kw_args)
    """
    return iter(list(kosh_obj.find(**kw_args)))

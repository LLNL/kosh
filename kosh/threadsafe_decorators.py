from filelock import FileLock
import time
import functools
import os
import sys
import warnings


warnings.warn(("This version of threadsafe will be depreciated in Kosh 3.2. "
               "Use lock_strategies instead: examples/Example_ThreadSafe.ipynb."))


def threadsafe_call(num_tries, patience):
    """
    A custom thread lock decorator made specifically for Kosh calls.

    Parameters:
    -----------
    num_tries: int
        The number of allowable attempts.
    patience: float
        The number of seconds to wait between tries.

    Returns:
    --------
    A threadsafe version of the decorated function.
    """
    def decorator(func):

        @functools.wraps(func)
        def threadsafe(*args, **kw_args):
            exceptions = []
            for i in range(num_tries):
                try:
                    lock_path = os.path.join(os.path.expanduser("~"), ".kosh.lock")
                    lock = FileLock(lock_path, timeout=3600)

                    with lock:
                        ret = func(*args, **kw_args)
                        return ret

                except ValueError as e:
                    exceptions.append(e)
                    if i < (num_tries - 1):
                        msg = f"\nWARNING: function {func} threw exception within "
                        msg += f"threadsafe decorator. Waiting {patience} seconds "
                        msg += "before next attempt.\n"
                        sys.stderr.write(msg)
                        time.sleep(patience)

            for e in exceptions:
                sys.stderr.write(f"\n{e}")
            raise ValueError(exceptions[0])

        return threadsafe
    return decorator


def patient_find(num_tries, patience):
    """
    Kosh "find" methods can sometimes return empty results when they
    shouldn't. It's really unclear why/how this happens. This decorator
    allows us to perform the search multiple times if the initial results
    are empty.

    Parameters:
    -----------
    num_tries: int
        The number of allowable attempts.
    patience: float
        The number of seconds to wait between tries.

    Returns:
    --------
    A version of the "find" function that will re-attempt
    the find if the initial results are empty.
    """
    def decorator(func):

        @functools.wraps(func)
        def patient_call(*args, **kw_args):
            for i in range(num_tries):
                find_res = list(func(*args, **kw_args))

                if len(find_res) > 0:
                    return iter(find_res)
                else:
                    time.sleep(patience)

            msg = f"\nWARNING: patient_find failed {num_tries} attempts. Returning empty results.\n"
            sys.stderr.write(msg)

            return iter(find_res)

        return patient_call
    return decorator

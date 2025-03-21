from filelock import FileLock, Timeout
from collections.abc import Generator
import time
import functools
import os
import logging

LOGGER = logging.getLogger()
LOGGER.setLevel(int(os.environ.get('LOCK_STRATEGIES_LOG_LEVEL', 30)))   # 30 is logging.WARNING
pid_map = {}


def lock_function(func):
    """
    Decorator to wrap functions with a LockStrategy.

    The lock is acquired at the start of the method and released after the
    method execution or after the entire generator is consumed.

    This is only intended to be used on class methods

    :param func: The method to be wrapped with the lock.
    :type func: Callable
    :return: The wrapped method with applied locking strategy.
    :rtype: Callable
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        lock_strategy = kwargs.get('lock_strategy', None)
        if lock_strategy is None:
            lock_strategy = NoLocking()
        return lock_strategy.threadsafe_call(func)(*args, **kwargs)
    return wrapper


def lock_method(func):
    """
    Decorator to wrap class methods with a LockStrategy.

    This decorator ensures that the method is executed with a locking
    mechanism provided by the LockStrategy. For methods that return
    generators, the lock will remain active throughout the iteration
    over the generator to ensure thread safety.

    The lock is acquired at the start of the method and released after the
    method execution or after the entire generator is consumed.

    :param func: The method to be wrapped with the lock.
    :type func: Callable
    :return: The wrapped method with applied locking strategy.
    :rtype: Callable
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kargs):
        @functools.wraps(func)
        def kosh_method(*args, **kargs):
            return func(self, *args, **kargs)
        with self.lock_strategy:
            result = self.lock_strategy.threadsafe_call(kosh_method)(*args, **kargs)
        return result
    return wrapper


class LockStrategy:
    """
    Base class for implementing different locking strategies.

    This class provides a common interface for various locking mechanisms, ensuring
    that derived classes can define their own locking behavior. The `LockStrategy`
    class implements the context manager protocol, allowing it to be used in `with`
    statements to acquire and release locks automatically.

    Derived classes should implement the `lock` and `unlock` methods to provide
    specific locking behavior, such as thread-based locks, file-based locks, or
    distributed locks.

    Example usage:
        >>> lock_strategy = SomeLockStrategy()
        >>> with lock_strategy:
        >>>     # Critical section of code
        >>>     pass

    Methods:
        lock(): Acquire the lock.
        unlock(): Release the lock.
        __enter__(): Enter the context manager, calling lock().
        __exit__(exc_type, exc_value, traceback): Exit the context manager, calling unlock().

    :param lock_strategy: A LockStrategy object to incorporate locking mechanisms.
    :type lock_strategy: LockStrategy
    """

    def lock(self):
        """Acquire the lock."""
        pass

    def unlock(self):
        """Release the lock."""
        pass

    def __enter__(self):
        self.lock()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.unlock()

    def threadsafe_call(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kargs):
            return func(*args, **kargs)
        return wrapper

    def wrap_generator(self, generator_func):
        @functools.wraps(generator_func)
        def wrapper(*args, **kargs):
            generator = generator_func(*args, **kargs)
            return self.lock_generator(generator)
        return wrapper

    def lock_generator(self, generator):
        while True:
            with self:
                try:
                    next_item = next(generator)
                except StopIteration:
                    break
            yield next_item


class NoLocking(LockStrategy):
    """
    A no-op LockStrategy
    """
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class RFileLock(LockStrategy):
    """
    A Re-entrant Filelock.
    """
    def __init__(self, num_tries=10, patience=1, lock_path=None, timeout=180):
        self.num_tries = num_tries
        self.patience = patience
        if lock_path is None:
            lock_path = os.path.join(os.path.expanduser("~"), ".kosh.lock")
        self.timeout = timeout
        self._lock = FileLock(lock_path, timeout=timeout)
        self.pid = os.getpid()
        # Keeps track of number of nested functions
        if self.pid not in pid_map:
            pid_map[self.pid] = 0
        self.functions = []

    def lock(self):
        import random
        LOGGER.info(msg=f"    {self.pid = } entering lock with {self.timeout = } secs & count {pid_map[self.pid]}...")
        exceptions = []
        for i in range(self.num_tries):
            try:
                if pid_map[self.pid] == 0:
                    self._lock.acquire()
                # Entering nested function
                pid_map[self.pid] += 1
                return

            except Timeout as e:
                exceptions.append(e)
                if i < (self.num_tries - 1):
                    msg = f"    Lock {self.timeout = } timed out. Waiting {self.patience} seconds before next attempt."
                    LOGGER.warning(msg=msg)
                    time.sleep(self.patience + random.random())  # random in case parallel calls retry at same time
        for e in exceptions:
            LOGGER.exception(msg=repr(e))
        raise exceptions[0]

    def unlock(self):
        # Exiting nested function
        pid_map[self.pid] -= 1
        # Back to original function
        if pid_map[self.pid] <= 0:
            self._lock.release()
            self.functions = []

    def threadsafe_call(self, func):
        import random

        @functools.wraps(func)
        def wrapper(*args, **kargs):
            exceptions = []
            num_tries = self.num_tries
            self.functions.append(str(func))
            while num_tries > 0:
                try:
                    LOGGER.info(msg=f'Entering function: {func} with {*args,} & { {k: v for k, v in kargs.items()} }')
                    result = func(*args, **kargs)
                    if isinstance(result, Generator):
                        result = self.lock_generator(result)

                    LOGGER.info(msg=f'Exiting function: {func}')
                    return result

                except Exception as e:
                    if any(func_error == "RETRIES COMPLETE" for func_error in self.functions):
                        raise
                    import traceback
                    error_stack = traceback.extract_stack()
                    tb = e.__traceback__
                    num_tries -= 1
                    exceptions.append(e)
                    msg = f"\nError in parent function {error_stack[0]}.\n"
                    msg += f"Exception {e} in child function {func.__name__}. "
                    msg += f'Details: {func} with {*args,} & { {k: v for k, v in kargs.items()} } '
                    msg += f"Retrying in {self.patience} seconds. {num_tries} retries remaining..."
                    trace = "\n\t\t".join([str(x) for x in traceback.extract_tb(tb)])
                    msg += f"\nError Stack:\n\t\t{trace}"
                    LOGGER.warning(msg=msg)
                    print(msg)  # For users without a logger
                    time.sleep(self.patience + random.random())  # random in case parallel calls retry at same time
            self.functions.append("RETRIES COMPLETE")
            exception_reprs = '\n'.join([repr(e) for e in exceptions])
            msg = f"Function {func.__name__} failed after {self.num_tries} attempts. "
            msg += f"Exception Traceback(s): \n{exception_reprs}"
            LOGGER.exception(msg=msg)
            raise exceptions[0]
        return wrapper


class OnlyRetry(RFileLock):
    """
    Only retry, no locking.
    """
    def __init__(self, num_tries=10, patience=1):
        self.num_tries = num_tries
        self.patience = patience
        self.pid = os.getpid()
        # Keeps track of number of nested functions
        if self.pid not in pid_map:
            pid_map[self.pid] = 0
        self.functions = []

    def __enter__(self):
        # Entering nested function
        pid_map[self.pid] += 1
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        # Exiting nested function
        pid_map[self.pid] -= 1
        # Back to original function
        if pid_map[self.pid] <= 0:
            self.functions = []
        pass

    def lock(self):
        pass

    def unlock(self):
        pass

# this defines Kosh transformers object to augment/filter data from loaders
from abc import ABCMeta, abstractmethod
import hashlib
import os
from ..exec_graphs import KoshExecutionGraph
from kosh import kosh_cache_dir


class KoshTransformer(KoshExecutionGraph):
    # Defines which input types it can handle
    # and what output it sends back
    __metaclass__ = ABCMeta
    types = {"numpy": ["numpy", ]}

    def __init__(self,
                 cache_dir=kosh_cache_dir,
                 cache=False, verbose=False, *args, **kargs):
        """init function will receive the previous step's signature and the cache directory
        and output signature is also generated from the input args (w/o the cache_dir)
        :param cache_dir: directory to save cachd files
        :type cache_dir: str
        :param cache: do we use cache? 0: no, 1:yes, 2:yes but clobber if exists
        :type cache: int
        :param verbose: Turn on verbosity, by default this will turn on printing a message
                        when results are loaded from cache. Message is sent as lone argument
                        to `self._print` function.
                        value is stored in self._verbose
        :type verbose: bool
        """
        self.signature = hashlib.sha256(repr(self.__class__).encode())
        self.signature = self.update_signature(*args, **kargs)
        self.cache_dir = cache_dir
        self._verbose = verbose
        if cache:
            try:
                os.makedirs(self.cache_dir)
            except Exception:
                pass
        self.cache = cache

    def transform_(self, input, format, signature=None):
        """Given input from previous loader or transformer and desired format
        computes the unique signature and tries to extract from cache, calls transformer's
        `transform` function if no cache available.
        :param input: set of input passed from loader or previous transformer
        :type input: object
        :param format: desired output format
        :type format: str
        :return: The result from transform function
        :rtype: object
        """

        if self.cache:
            if signature is None:
                use_signature = self.update_signature(input, format).hexdigest()
            else:
                use_signature = signature

            cache_file = os.path.join(self.cache_dir, use_signature)
            if self.cache == 2 and os.path.exists(cache_file):
                # User wants to clobber cache
                os.remove(cache_file)

            try:
                result = self.load(use_signature)
                if self._verbose:
                    self._print("Loaded results from cache file {} using signature: {}".format(
                        cache_file, use_signature))
            except Exception:
                if signature is None:
                    signature = self.update_signature(input, format).hexdigest()
                result = self.transform(input, format)
                if self.cache > 0:  # Ok user wants to cache results
                    if not os.path.exists(self.cache_dir):
                        os.makedirs(self.cache_dir)
                    self.save(signature, result)
        else:
            result = self.transform(input, format)
        return result

    def _print(self, message):
        print(message)

    @abstractmethod
    def transform(self, input_, format):
        """The transform function
        :param input_: result returned by loader or previous transformer
        """
        raise NotImplementedError("the transform function is not implemented")

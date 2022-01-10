from ..exec_graphs import KoshExecutionGraph
from abc import abstractmethod
import hashlib
from kosh import kosh_cache_dir
import os


class KoshOperator(KoshExecutionGraph):
    # TODO something about operator accepting multiple types in (list?)
    # For now assuming it is type of first input received
    types = {}

    def __init__(self,
                 *args, **kargs):
        """init function
        output signature is generated from the input args (w/o the cache_dir)
        :param cache_dir: directory to save cached files must be passed as key/value
        :type cache_dir: str
        :param cache: do we use cache? 0: no, 1:yes, 2:yes but clobber if exists must be passed as key/value
        :type cache: int
        :param verbose: Turn on verbosity, by default this will turn on printing a message
                        when results are loaded from cache. Message is sent as lone argument
                        to `self._print` function.
                        value is stored in self._verbose
        :type verbose: bool
        """
        self.cache_dir = kargs.pop("cache_dir", kosh_cache_dir)
        self.signature = hashlib.sha256(repr(self.__class__).encode())
        self.signature = self.update_signature(*args, **kargs)
        self.use_cache = kargs.pop("use_cache", False)
        self._verbose = kargs.pop("verbose", False)
        cache = kargs.pop("cache", False)
        if cache:
            try:
                os.makedirs(self.cache_dir)
            except Exception:
                pass
        self.cache = cache
        super(KoshOperator, self).__init__(*args, **kargs)

    def operate_(self, *inputs, **kargs):
        """Given input(s) from previous loader, transformer or operators and desired format
        computes the unique signature and tries to extract from cache, calls operator's
        `operate` function if no cache available.
        :param *inputs: set of input passed from loader or previous transformer
        :type *inputs: object
        :param format: desired output format, must be passed as keyword
                       will be attached to object for retrieval in the function
        :type format: str
        :return: The result from operate function on inputs
        :rtype: object
        """

        format = kargs["format"]
        if self.cache:
            signature = kargs.get("signature", None)

            if signature is None:
                use_signature = self.update_signature(inputs, format).hexdigest()
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
                    signature = self.update_signature(inputs, format).hexdigest()
                result = self.operate(*inputs, format=format)
                if self.cache > 0:  # Ok user wants to cache results
                    if not os.path.exists(self.cache_dir):
                        os.makedirs(self.cache_dir)
                    self.save(signature, result)
        else:
            result = self.operate(*inputs, format=format)

        return result

    def _print(self, message):
        print(message)

    @abstractmethod
    def operate(self, *inputs, **kargs):
        """The operating function on the inputs
        :param inputs: result returned by loader or previous transformer
        :type inputs: tuple of features/execution graphs
        """
        raise NotImplementedError("the transform function is not implemented")

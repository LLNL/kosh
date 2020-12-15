# this defines Kosh transformers object to augment/filter data from loaders
from abc import ABCMeta, abstractmethod
import hashlib
import os
import pickle

kosh_cache_dir = os.path.join(os.environ["HOME"], ".cache", "kosh")  # noqa


class KoshTransformer(object):
    # Defines which input types it can handle
    # and what output it sends back
    __metaclass__ = ABCMeta
    types = {"numpy": ["numpy", ]}

    def __init__(self,
                 cache_dir=kosh_cache_dir,
                 cache=False, *args, **kargs):
        """init function will receive the previous step's signature and the cache directory
        and output signature is also generated from the input args (w/o the cache_dir)
        :param cache_dir: directory to save cachd files
        :type cache_dir: str
        :param cache: do we use cache? 0: no, 1:yes, 2:yes but clobber if exists
        :type cache: int
        """
        self.signature = hashlib.sha256(repr(self.__class__).encode())
        self.signature = self.update_signature(*args, **kargs)
        self.cache_dir = cache_dir
        if cache:
            try:
                os.makedirs(self.cache_dir)
            except Exception:
                pass
        self.cache = cache

    def update_signature(self, *args, **kargs):
        """Updated the signature based to a set of args and kargs
        :param *args: as many arguments as you want
        :type *args: list
        :param **kargs: key=value style argmunets
        :type **kargs: dict
        :return: updated signature
        :rtype: str
        """
        signature = self.signature.copy()
        for arg in args:
            signature.update(repr(arg).encode())
        for kw in kargs:
            signature.update(repr(kw).encode())
            signature.update(repr(kargs[kw]).encode())
        return signature

    def show_cache_file(self, input, format):
        """Given a set of input and format returns the unique signature used for cache file
        :param input: set of input passed from loader or previous transformer
        :type input: object
        :param format: desired output format
        :type format: str
        :return: The unique signature
        :rtype: str
        """
        signature = self.update_signature(input, format).hexdigest()
        return os.path.join(self.cache_dir, signature)

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

        if signature is None:
            use_signature = self.update_signature(input, format).hexdigest()
        else:
            use_signature = signature

        cache_file = os.path.join(self.cache_dir, use_signature)
        if self.cache == 2 and os.path.exists(cache_file):
            # User wants to clobber cahce
            os.remove(cache_file)

        try:
            result = self.load(use_signature)
        except Exception:
            if signature is None:
                signature = self.update_signature(input, format).hexdigest()
            result = self.transform(input, format)
            if self.cache > 0:  # Ok user wants to cache results
                if not os.path.exists(self.cache_dir):
                    os.makedirs(self.cache_dir)
                self.save(signature, result)
        return result

    def save(self, cache_file, *content):
        """Pickle some data to a cache file
        :param cache_file: name of cache file, will be joined with self.cache_dir
        :type cache_file: str
        :param content: content to save to cache
        :type content: object
        """
        with open(os.path.join(self.cache_dir, cache_file), "wb") as f:
            for sv in content:
                pickle.dump(sv, f)

    def load(self, cache_file):
        """loads content from cache
        :param cache_file: name of cache file, will be joined with self.cache_dir
        :type cache_file: str
        :return: unpickled data
        :rtpye: object
        """
        with open(os.path.join(self.cache_dir, cache_file), "rb") as f:
            cont = True
            data = []
            while cont:
                try:
                    data.append(pickle.load(f))
                except Exception:
                    cont = False
        if len(data) == 1:
            return data[0]
        else:
            return data

    @abstractmethod
    def transform(self, input_, format):
        """The transform function
        :param input_: result returned by loader or previous transformer
        """
        raise NotImplementedError("the transform function is not implemented")

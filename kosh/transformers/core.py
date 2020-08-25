# this defines Kosh transformers object to augment/filter data from loaders
from abc import ABCMeta, abstractmethod
import hashlib
import networkx as nx
import os
import pickle


kosh_cache_dir = os.path.join(os.environ["HOME"], ".cache", "kosh")  # noqa


def populate(G, node, output_formats, next_nodes, final_format=None):
    """Populates networkx
    :param G: networkx Graph to populate
    :type G: nx.Graph
    :param node: transformer to be chained needs to have dict "types"
    :type node: object with types attributes as a dictionary
    :param output_formats: output_format of the first node
    :type output_formats: list
    :param next_nodes: next set of transformers to add to graph
    :type next_nodes: object with types attriubte as a dictionary
    :param final_format: desired end format
    :type final_format: str
    :return: Nothing but the graph passed is updated
    """
    for format in output_formats:
        if format in next_nodes[0].types:
            this_node = (format, next_nodes[0])
            G.add_edge(node, this_node)
            if len(next_nodes) > 1:
                populate(
                    G, this_node, next_nodes[0].types[format], next_nodes[1:], final_format)
            else:
                if final_format in next_nodes[0].types[format] or final_format is None:
                    G.add_edge(this_node, (final_format, None))


def get_path(input_type, loader, transformers, output_format):
    """given a loader and its transformer return path to desired format
    e.g which output format should each transformer pick to be chained to the follwoing one
    in order to obtain the desired outcome for format
    :param input_type: input type of first node
    :type input_type: str
    :param loader: original loader
    :type loader: KoshLoader
    :param transformers: set of transformers to be added after loader exits
    :type transformers: list of KoshTransformer
    :param output_format: desired output format
    :type output_format: str
    :return: shortest path from desired input_type to desired format
    """
    if input_type not in loader.types:
        raise RuntimeError(
            "loader cannot load mime_type {}".format(input_type))
    G = nx.Graph()
    start_node = (input_type, loader)
    G.add_node(start_node)
    if len(transformers) == 0:
        # No transformer
        if output_format in loader.types[input_type] or output_format is None:
            G.add_edge(start_node, (output_format, None))
    else:
        populate(
            G,
            start_node,
            loader.types[input_type],
            transformers,
            output_format)
    pth = nx.shortest_path(G, (input_type, loader), (output_format, None))
    return pth


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
        :param cache: do we use cache?
        :type cache: bool
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
        try:
            result = self.load(use_signature)
        except Exception:
            if signature is None:
                signature = self.update_signature(input, format).hexdigest()
            result = self.transform(input, format)
            if self.cache:  # Ok user wants to cache results
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

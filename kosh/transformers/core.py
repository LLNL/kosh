# this defines Kosh transformers object to augment/filter data from loaders
from abc import ABCMeta, abstractmethod
import hashlib
import networkx as nx
import os
import pickle
import random

kosh_cache_dir = os.path.join(os.environ["HOME"], ".cache", "kosh")  # noqa


def populate(G, node, output_formats, next_nodes, final_format=None, depth=1, lbls_dict={}):
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
    #output_formats += ["graph", ]
    for format in output_formats:
        if format in list(next_nodes[0].types):
            this_node = (format, next_nodes[0])
            lbls_dict[this_node] = "{}: {}".format(depth, format)
            weight = 1.
            if this_node[0] == node[0]:
                weight /= 2.  # Gives more weight for i/o of same format
            if this_node[0] == final_format:
                weight /= 3.  # Gives more weight if format matches output_format
            G.add_edge(node, this_node, weight=weight)
            if len(next_nodes) > 1:
                populate(
                    G, this_node, next_nodes[0].types[format], next_nodes[1:], final_format, depth=depth+1, lbls_dict=lbls_dict)
            else:
                for final_fmt in next_nodes[0].types[format]:
                    weight = 1.
                    if this_node[0] == final_fmt:
                        weight /= 2.
                    if final_fmt == final_format:
                        weight /= 3.
                    final_node = (final_fmt, None) 
                    G.add_edge(this_node, final_node, weight=weight)
                    lbls_dict[final_node] = "end : {}".format(final_fmt)


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
    G = nx.DiGraph()
    start_node = (input_type, loader) # so each graph is unique
    G.add_node(start_node)
    lbls_dict = {start_node: "start: {}".format(input_type)}
    if len(transformers) == 0:
        # No transformer
        for out_format in loader.types[input_type]:
            node = (out_format, None)
            G.add_edge(start_node, node)
            lbls_dict[node] = "end: {}".format(out_format)
    else:
        populate(
            G,
            start_node,
            loader.types[input_type],
            transformers,
            output_format,
            lbls_dict=lbls_dict)

    G.labels_dict = lbls_dict

    if output_format is None:
        if len(transformers) == 0:
            output_format = loader.types[input_type][0]
            pth = nx.shortest_path(G, start_node, (output_format, None), weight="weight")
        else:
            pth = None
            for last_in_type in transformers[-1].types:
                if pth is not None:
                    break
                for out_format in transformers[-1].types[last_in_type]:
                    try:
                        pth = nx.shortest_path(G, start_node, (out_format, None), weight="weight")
                        break
                    except Exception:
                        pass
    else:
        pth = nx.shortest_path(G, start_node, (output_format, None), weight="weight")
    
    # Sets parents
    for i, node in enumerate(pth[1:-1]):
        node[1].parent = pth[i][1]
    return G, pth


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

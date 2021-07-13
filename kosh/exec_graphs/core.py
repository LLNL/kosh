import networkx as nx
from kosh import kosh_cache_dir
import kosh
import random
import pickle
import os
import collections


def possible_ends(graph, start_nodes, end_nodes):
    """Finds all network ends that can be reached by all start nodes
    :param graph: The full graph
    :type graph: networkx.OrderedDiGraph
    :param start_nodes: Node to start paths from
    :type start_nodes: list of nodes
    :param end_nodes: Node to end paths from
    :type end_nodes: list of nodes
    :returns: list of possible end nodes
    :rtype: list
    """
    ok_ends = []  # Matching end nodes for each start
    for start in start_nodes:
        ok_ends_this_start = []
        for end in end_nodes:
            try:
                nx.shortest_path(graph, start, end)
                ok_ends_this_start.append(end)
            except Exception:  # Can't get to this
                pass
        ok_ends.append(ok_ends_this_start)

    # Ok for each start we know know the ok end nodes
    # Which ones are common to every start?
    out = ok_ends[0]
    for ends in ok_ends[1:]:
        # now let's remove all end nodes that are not in the other paths
        for possible_end in list(out):
            if possible_end not in out:
                out.pop(possible_end)
    # out now contains the possible end nodes
    return out


def populate(G, node, output_formats, next_nodes):
    """Populates networkx
    :param G: networkx Graph to populate
    :type G: nx.Graph
    :param node: transformer to be chained, needs to have dict "types" attribute
    :type node: object with types attributes as a dictionary
    :param output_formats: output_format of the first node
    :type output_formats: list
    :param next_nodes: next set of transformers to add to graph
    :type next_nodes: object with 'types' attribute as a dictionary
    :returns: Nothing but the graph passed is updated
    """
    for format in output_formats:
        if format in list(next_nodes[0].types):
            this_node = (format, next_nodes[0], G.seed)
            G.add_edge(node, this_node)
            if len(next_nodes) > 1:
                populate(
                    G, this_node, next_nodes[0].types[format], next_nodes[1:])
            else:
                for final_fmt in next_nodes[0].types[format]:
                    final_node = (final_fmt, None, G.seed)
                    G.add_edge(this_node, final_node)


def find_network_ends(G, start=True, end=True):
    """Given a networkx.OrderedDiGraph finds start or end nodes or both.
    :param G: Network of interest
    :type G: networkx.OrderedDiGraph
    :param start: Are we searching for start nodes?
    :type start: bool
    :param end: Are we searching for end nodes?
    :type end: bool
    :returns: start and end node lists or just start/end node list
    :rtype: (tuple of) list
    :raises ValueError: if neither start or end is True
    """
    starters = []
    ends = []
    if not start and not end:
        raise ValueError("You need to set at least one of start/end to True")
    for node in G.nodes():
        if start and len(list(G.predecessors(node))) == 0:
            starters.append(node)
        if end and len(list(G.successors(node))) == 0:
            # Py2 seems to be returning these in random order
            # let's try to order this in consistent fashion
            # with predecessor's types order
            preds = list(G.predecessors(node))
            inserted = False
            for index, enode in enumerate(list(ends)):
                pred = preds[-1]
                pred_end = list(G.predecessors(enode))[-1]
                if pred[:1] == pred_end[:1]:
                    end_types = pred[1].types[pred[0]]
                    enode_index = end_types.index(enode[0])
                    node_index = end_types.index(node[0])
                    if node_index < enode_index:
                        ends.insert(index, node)
                        inserted = True
                        break
            if not inserted:
                ends.append(node)
    if start and not end:
        return starters
    elif end and not start:
        return ends
    else:
        return starters, ends


def apply_weight(G, output_format=None, weight_same=2., weight_output=3.):
    """Given a graph, lower the weight to edges that end in required format
    :param output_format: Desired output format, used to lower weight if edge ends in that format
    :type output_format: str
    :param weight_same: Weight to use if both end of an edge are the same format
    :type weight_same: float
    :param weight_output: Weight to use if end of an edge is the desired output format
    :type weight_output: float
    :return: Notne but the input graph is modified
    :rtype: None
    """
    for (n1, n2) in G.edges():
        weight = 1.
        # Does this edge connect identical formats?
        if n1[0] == n2[0]:
            weight /= weight_same

        # Does this edge ends with desired output format?
        if n2[0] == output_format and output_format is not None:
            weight /= weight_output
        G[n1][n2]["weight"] = weight


def get_seed(G, node, end_seed=None):
    """Assigns a new random seed to a node unless it is an end_seed, in which case we assign the Graph's seed
    :param G: Parent graph
    :type G: networkx.Graph
    :param node: Node of interest (is it and end node?)
    :type node: a graph node
    :param end_seed: The seed to assign if the node is an end seed. If None is passed then used parent Graph's seed
    :type seed: int (or None)
    :return: new seed for the node
    :rtype: int
    """
    if end_seed is None:
        end_seed = G.seed
    if len(list(G.successors(node))) == 0:
        seed = end_seed
    else:
        seed = random.random()
    return seed


class KoshExecutionGraph(object):
    types = {}

    def __len__(self):
        return len(self._graph)

    def __init__(self, *inputs, **kw):
        graphs = []
        # Get a new seed
        self.seed = random.random()
        # Set the variable to receive results per index
        self.index_results = {}
        # Create a new merged graph
        new_graph = nx.OrderedDiGraph()
        new_graph.seed = random.random()
        for i, G in enumerate(inputs):
            if isinstance(G, KoshExecutionGraph):
                G = G.execution_graph()
            elif not hasattr(G, "seed"):
                G.seed = random.random()
            # remember all graph we sent in
            graphs.append(G)
            new_graph.update(G)

        for mime in self.types:
            if not isinstance(mime, (list, tuple)):
                mime_list = [mime, ]
            else:
                mime_list = mime
            # Ok all inputs can be extracted to this thing input type
            for i, G in enumerate(graphs):
                if i >= len(mime_list):
                    output_format = mime_list[-1]
                else:
                    output_format = mime_list[i]
                for node in G.nodes():
                    if node == (output_format, None, G.seed):
                        # Connect last node to us
                        new_node = (node[0], self, self.seed)
                        pred = G.predecessors(node)
                        for n in pred:
                            new_graph.add_edge(n, new_node)
                        new_graph.remove_node(node)
                        # connect new node to export types
                        for export_type in self.types[mime]:
                            if not isinstance(export_type, (list, tuple)):
                                export_type = [export_type, ]
                            for export in export_type:
                                new_graph.add_edge(
                                    new_node, (export, None))
        # At this point we need to make sure there is a way out of this
        if isinstance(self, (kosh.operators.KoshOperator, kosh.transformers.KoshTransformer, kosh.loaders.KoshLoader)):
            start_nodes, end_nodes = find_network_ends(new_graph, start=True, end=True)
            out_types = set()
            for input_type in self.types:
                for output_type in self.types[input_type]:
                    out_types.add(output_type)
            # now let's go from each start node and see if we can go the an end node of an output type
            got_thru = False
            for output_type in out_types:
                for end_node in end_nodes:
                    if end_node[0] == output_type:
                        # ok can we go to this?
                        all_go = True
                        for start_node in start_nodes:
                            try:
                                _ = nx.shortest_path(new_graph, start_node, end_node)
                            except Exception:
                                all_go = False
                                break
                        if all_go:
                            got_thru = True
                            break
                if got_thru:
                    break

            if not got_thru:
                raise RuntimeError("Could not find an output format")

        self._graph = new_graph
        self.start_nodes, self.end_nodes = find_network_ends(new_graph)
        self.possible_end_nodes = possible_ends(new_graph, self.start_nodes, self.end_nodes)
        self.paths = collections.OrderedDict()

    def execution_graph(self, seed=None, verbose=False,
                        png_template="LOADER_GRAPH_{}"):
        """makes a new graph with unique seed
        Helps networkx differentiate between identical loaders/transformers/operators
        :param seed: seed to use for new graph
        :type seed: int
        :param verbose: verbose generation, also generates a png with the grap representation
                        Mostly used for debug purposes
        :type verbose: bool
        :param png_template: template to use to generate graph png in verbose mode
                             "_IN"/"_OUT" will be appended and seed will be fed
        :type png_template: str
        :return a new graph with new seed
        :rtype: networkx.OrderedDiGraph
        """
        G = nx.OrderedDiGraph()
        if seed is None:
            seed = random.random()
        G.seed = seed
        if verbose:
            try:
                if "DISPLAY" not in os.environ or os.environ["DISPLAY"] == "":
                    import matplotlib
                    matplotlib.use("agg", force=True)
                import matplotlib.pyplot as plt
                nx.draw(self._graph)
                plt.show()
                png_name = png_template + "_IN.png"
                plt.savefig(png_name.format(seed))
                plt.clf()
            except ImportError:
                raise RuntimeError("Could not import matplotlib, will not plot anything")
        used_nodes = collections.OrderedDict()
        for (n1, n2) in self._graph.edges():
            if n1 in used_nodes:
                # we already generated a new random number for that node
                N1 = used_nodes[n1]
            else:
                # Never seen that node
                seed = get_seed(self._graph, n1, G.seed)
                N1 = n1[0], n1[1], seed
                used_nodes[n1] = N1
            if n2 in used_nodes:
                # we already generated a new random number for that node
                N2 = used_nodes[n2]
            else:
                # Never seen that node
                seed = get_seed(self._graph, n2, G.seed)
                N2 = n2[0], n2[1], seed
                used_nodes[n2] = N2
            G.add_edge(N1, N2)
        if verbose:
            nx.draw(G)
            plt.show()
            png_name = png_template + "_OUT.png"
            plt.savefig(png_name.format(seed))
            plt.clf()
        return G

    def __getitem__(self, key):
        """Very bare bone get item function
        It is highly recommended to re-implement this.
        Calls traverse then __getitem__ on the result of traverse
        :param key: key to access
        :type key: object (usually int, slice or str)
        """
        return self.traverse(__getitem_key__=key)

    def traverse(self, format=None, *args, **kargs):
        """Traverse the execution graph and returns data
        :param format: desired output format
        :type format: str
        """
        G = self._graph
        start_nodes, _ = self.start_nodes, self.end_nodes

        possible_end_nodes = self.possible_end_nodes

        if len(possible_end_nodes) == 0:
            raise RuntimeError("This graph cannot be traversed to a single end node from each start. Aborting")
        # We first need to determine the output_format
        if format is None:  # User lets us pick
            format = possible_end_nodes[0][0]

        if format not in [end[0] for end in possible_end_nodes]:
            raise ValueError("Cannot output in format {}".format(format))

        # Which node is our exit node?
        for end_node in possible_end_nodes:
            if end_node[0] == format:
                break

        if format not in self.paths:
            # Ok now let's apply the weights
            apply_weight(G, output_format=format)

            # And get the shortest path(s)
            # For each entry path
            pths = []
            for start_node in start_nodes:
                pths.append(nx.shortest_path(G, start_node, end_node))
            self.paths[format] = pths
        else:
            pths = self.paths[format]
        # Ok let's generate the new network with only the paths
        out = nx.OrderedDiGraph()
        out.seed = G.seed
        for pth in pths:
            for i, node in enumerate(pth[:-1]):
                if pth[i+1][1] is not None:
                    pth[i+1][1].parent = node[1]
                out.add_edge(node, pth[i + 1])

        # We can now travel back the pth to obtain
        # the data.
        return self._operate(out, end_node, format, **kargs)

    __call__ = traverse

    def _operate(self, graph, node, output_format, **kargs):
        """Actual bells and whistles to get the data
        :param graph: The graph to follow in order to get the data
        :type graph: KoshExecutionGraph
        :param node: node to process (get inputs and call extract/transform/operate func)
        :type node: node in the network
        :param out_format: The desired output_format
        :type output_format: str
        :param kargs: key arguments that will be passed to loaders (start of each path)
        :type kargs: dict
        :returns: Data
        """
        getitem_key = kargs.pop("__getitem_key__", slice(None, None, None))
        cache_file_only = kargs.pop("cache_file_only", False)
        use_cache = kargs.pop("use_cache", False)
        cache_dir = kargs.pop("cache_dir", kosh_cache_dir)
        starters, end = find_network_ends(graph, start=True, end=True)
        previous = list(graph.predecessors(node))
        if len(previous) == 0:
            # Ok we are at the start e.g a loader
            node[1].cache_file_only = cache_file_only
            node[1].use_cache = use_cache
            node[1].cache_dir = cache_dir
            node[1]._user_passed_parameters = (None, kargs)
            if not isinstance(getitem_key, slice) or getitem_key != slice(None, None, None):
                if "__getitem__" in node[1].__class__.__dict__:
                    out = node[1][getitem_key]
                else:
                    out = node[1].extract_(format=output_format)[getitem_key]
            else:
                out = node[1].extract_(format=output_format)
            return out
        else:
            inputs = ()
            for input_index, prev in enumerate(previous):
                kargs2 = kargs.copy()
                do_res = True
                if hasattr(node[1], "__getitem_propagate__") and getitem_key != slice(None, None, None):
                    new_keys = node[1].__getitem_propagate__(getitem_key, input_index=input_index)
                    kargs2["__getitem_key__"] = new_keys
                    if new_keys is None:
                        res = getattr(node[1], "index_results", {}).get(input_index, None)
                        do_res = False
                elif type(self) == kosh.exec_graphs.core.KoshExecutionGraph or node == end[0]:
                    new_keys = False
                    kargs2["__getitem_key__"] = getitem_key
                else:
                    new_keys = None
                    kargs2["__getitem_key__"] = slice(None, None, None)
                if do_res:
                    res = prev[1]._operate(graph, prev, node[0], **kargs2)
                    if new_keys is None and getitem_key != slice(None, None, None):
                        res = res[getitem_key]
                inputs += (res,)
            if node == end[0]:
                # this is the last one, do not call operate on it
                # we are done
                return inputs[0]
            if hasattr(self, "operate_"):
                out = self.operate_(*inputs, format=node[0])
                return out
            elif hasattr(self, "transform_"):
                out = self.transform_(*inputs, format=node[0])
                return out
            elif isinstance(self, kosh.exec_graphs.core.KoshExecutionGraph):
                return inputs
            else:
                raise RuntimeError(
                    "Did not know which function to send inputs to. Aborting")

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

    def save(self, cache_file, *content):
        """Pickles some data to a cache file
        :param cache_file: name of cache file, will be joined with self.cache_dir
        :type cache_file: str
        :param content: content to save to cache
        :type content: object
        """
        with open(os.path.join(self.cache_dir, cache_file), "wb") as f:
            for sv in content:
                pickle.dump(sv, f)

    def load(self, cache_file):
        """Loads content from cache
        :param cache_file: name of cache fileA will be joined with self.cache_dir
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

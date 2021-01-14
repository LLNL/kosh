import networkx as nx
import itertools
import kosh
import random


def populate(G, node, output_formats, next_nodes):
    """Populates networkx
    :param G: networkx Graph to populate
    :type G: nx.Graph
    :param node: transformer to be chained needs to have dict "types"
    :type node: object with types attributes as a dictionary
    :param output_formats: output_format of the first node
    :type output_formats: list
    :param next_nodes: next set of transformers to add to graph
    :type next_nodes: object with types attriubte as a dictionary
    :return: Nothing but the graph passed is updated
    """
    #output_formats += ["graph", ]
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
    starters = []
    ends = []
    if not start and not end:
        raise ValueError("You need to set at least one of start/end to True")
    for node in G.nodes():
        if start and len(list(G.predecessors(node))) == 0:
            starters.append(node)
        if end and len(list(G.successors(node))) == 0:
            ends.append(node)
    if start and not end:
        return starters
    elif end and not start:
        return ends
    else:
        return starters, ends


def apply_weight(G, output_format=None, weight_same=2., weight_output=3.):
    """Given a graph, lower the weight to edges that end in required format"""
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
    if end_seed is None:
        end_seed = G.seed
    #if len(list(G.predecessors(node))) == 0:
    #    seed = end_seed
    if len(list(G.successors(node))) == 0:
        seed = end_seed
    else:
        seed = random.random()
    #seed = random.random()
    return seed

class KoshIOGraph(object):
    types = {}

    def __len__(self):
        return len(self._graph)

    def __init__(self, *inputs, **kw):
        graphs = []
        self.seed = random.random()
        for i, G in enumerate(inputs):
            if isinstance(G, KoshIOGraph):
                G = G.io_graph()
            elif not hasattr(G,"seed"):
                G.seed = random.random()
            graphs.append(G)

        compatible = {}
        for mime in self.types:
            if not isinstance(mime, (list, tuple)):
                mime_list = [mime, ]
            else:
                mime_list = mime
            compatible[mime] = True
        for ig, G in enumerate(graphs):
            for mime in compatible:
                if not isinstance(mime, (list, tuple)):
                    mime_list = [mime, ]
                else:
                    mime_list = mime
                if i >= len(mime_list):
                    output_format = mime_list[-1]
                else:
                    output_format = mime_list[i]
                starters, ends = find_network_ends(G)
                for starter in starters:
                    try:
                        nx.shortest_path(G, starter, (output_format, None, G.seed))
                    except Exception as err:
                        print("ERR:",err)
                        compatible[mime] = False
        new_graph = nx.DiGraph()
        new_graph.seed = random.random()

        # Now we need to connect all input graphs via compatible inputs
        is_compatible = True
        for mime in compatible:
            is_compatible *= compatible[mime]
        if not is_compatible:
            raise ValueError(
                "Could not match your input graph to any known mime type")
        for G in graphs:
            new_graph.update(G)

        for mime in self.types:
            if not isinstance(mime, (list, tuple)):
                mime_list = [mime, ]
            else:
                mime_list = mime
            if compatible[mime]:
                # Ok all inputs can be extracted to this thing input type
                for i, G in enumerate(graphs):
                    if i >= len(mime_list):
                        output_format = mime_list[-1]
                    else:
                        output_format = mime_list[i]
                    for node in G.nodes():
                        if node == (output_format, None, G.seed):
                            new_node = (node[0], self, self.seed)
                            pred = G.predecessors(node)
                            for n in pred:
                                new_graph.add_edge(n, new_node)
                            new_graph.remove_node(node)
                            for export_type in self.types[mime]:
                                if not isinstance(export_type, (list, tuple)):
                                    export_type = [export_type, ]
                                for export in export_type:
                                    new_graph.add_edge(new_node, (export, None))
        self._graph = new_graph
        print("INIT WITH: {} nodes: {}".format(len(new_graph.nodes()), new_graph.nodes()))


    def io_graph(self, seed=None, verbose=False, png_template="LOADER_GRAPH_{}"):
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
        """
        G = nx.DiGraph()
        if seed is None:
            seed = random.random()
        G.seed = seed
        if verbose:
            import matplotlib.pyplot as plt
            nx.draw(self._graph)
            plt.show()
            png_name = png_template+"_IN.png"
            plt.savefig(png_name.format(seed))
            plt.clf()
        used_nodes = {}
        for (n1, n2) in self._graph.edges():
            if n1 in used_nodes:
                # we already generated a new random number for that node
                N1 = used_nodes[n1]
            else:
                # Never seen that node
                seed = get_seed(self._graph, n1, G.seed)
                N1 = n1[0], n1[1], seed
                used_nodes[n1] = N1
            if verbose:
                print("N1:", N1)
            if n2 in used_nodes:
                # we already generated a new random number for that node
                N2 = used_nodes[n2]
            else:
                # Never seen that node
                seed = get_seed(self._graph, n2, G.seed)
                N2 = n2[0], n2[1], seed
                used_nodes[n2] = N2
            if verbose:
                print("N2:", N2)
            G.add_edge(N1, N2)
            if verbose:
                print("\t%%%%%%%%")
        if verbose:
            nx.draw(G)
            plt.show()
            png_name = png_template+"_OUT.png"
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
        return self.traverse()[key]

    def traverse(self, format=None):
        G = self.io_graph()
        start_nodes, end_nodes = find_network_ends(G, start=True, end=True)
        print("Nodes:", len(self._graph.nodes()), len(G.nodes()), G.nodes())
        print("{} start nodes: {}".format(len(start_nodes), start_nodes))
        # We first need to determine the output_format
        if format is None:
            format = end_nodes[0][0]

        # Which node is our exit node?
        for end_node in end_nodes:
            if end_node[0] == format:
                break

        # Ok now let's apply the weights
        apply_weight(G, output_format=format)

        # And get the shortest path(s)
        # For each entry path
        pths = []
        for start_node in start_nodes:
            pths.append(nx.shortest_path(G, start_node, end_node))
        # Ok let's generate the new netwrok with only the paths
        out = nx.DiGraph()
        out.seed = G.seed
        for pth in pths:
            for i, node in enumerate(pth[:-1]):
                out.add_edge(node, pth[i+1])

        # We can now travel back the pth to obtain
        # the data.
        return self._operate(out, pths, format)

    def _operate(self, graph, paths, output_format):
        import sys
        sys.stdout.flush()
        starters, end = find_network_ends(graph, start=True, end =True)
        end = end[0]
        previous = list(graph.predecessors(end))
        if len(previous) == 0:
            # Ok we are at the start e.g a loader
            return end[1].extract()
        else:
            inputs = []
            for prev in previous:
                G = nx.DiGraph()
                pths = []
                for i, pth in enumerate(paths):
                    if prev in pth:
                        pths.append(pth[:-1])
                        G.add_node(pth[0])
                        for i, node in enumerate(pth[:-2]): # -2 because I remove the end node
                            G.add_edge(node, pth[i+1])
                res = prev[1]._operate(G, pths, end[0])
                inputs.append(res)
            if hasattr(self, "operate"):
                return self.operate(*inputs, format=end[0])
            elif hasattr(self,"transform_"):
                return self.transform_(*inputs, format=end[0])
            elif isinstance(self, kosh.io_graphs.core.KoshIOGraph):
                if len(pths) == 1:
                    inputs = inputs[0]
                return inputs
            else:
                raise RuntimeError("Did not know which function to send inputs to. Aborting")



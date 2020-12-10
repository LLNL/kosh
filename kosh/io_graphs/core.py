import networkx as nx
import itertools
import kosh
import random


def find_starters(G):
    starters = []
    for node in G.nodes():
        if len(list(G.predecessors(node))) == 0:
            starters.append(node)
    return starters

class KoshIOGraph(object):
    types = {}
    def __init__(self, *inputs, **kw):
        graphs = []
        self.seed = random.random()
        for i, G in enumerate(inputs):
            if isinstance(G, KoshIOGraph):
                G = G.io_graph()
            graphs.append(G)

        compatible = {}
        for mime in self.types:
            if not isinstance(mime, (list, tuple)):
                mime_list = [mime,]
            else:
                mime_list = mime
            compatible[mime] = True
        for ig, G in enumerate(graphs):
            kosh.utils.draw_io_graph(G, png_name="ADD_{}.png".format(ig))
            for mime in compatible:
                if not isinstance(mime, (list, tuple)):
                    mime_list = [mime,]
                else:
                    mime_list = mime
                if i >= len(mime_list):
                    output_format = mime_list[-1]
                else:
                    output_format = mime_list[i]
                starters = find_starters(G)
                for starter in starters:
                    try:
                        nx.shortest_path(G, starter, (output_format, None))
                    except Exception:
                        compatible[mime] = False
        new_graph = nx.DiGraph()
        connect_nodes = []
        # Now we need to connect all input graphs via compatible inputs
        is_compatible = True
        for mime in compatible:
            is_compatible *= compatible[mime]
        if not is_compatible:
            raise ValueError("Could not match your input graph to any known mime type")
        for G in graphs:
            new_graph.update(G)

        for mime in self.types:
            if not isinstance(mime, (list, tuple)):
                mime_list = [mime,]
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
                    if node == (output_format, None):
                        new_node = (node[0], self, self.seed)
                        new_graph.add_edges_from(
                            itertools.product(
                                new_graph.predecessors(node),
                                new_graph.successors(node)
                            )
                        )
                        new_graph.remove_node(node)
                        for export_type in self.types[mime]:
                            if not isinstance(export_type, (list, tuple)):
                                export_type = [export_type,]
                            for export in export_type:
                                new_graph.add_edge(new_node, (export, None)) 
        self._graph = new_graph

    def io_graph(self):
        """makes a new graph with unique seed"""
        G = nx.DiGraph()
        G.seed = random.random()
        for (n1, n2) in self._graph.edges():
            if len(n1) == 3:
                N1 = n1[0], n1[1], G.seed
            else:
                N1 = n1
            if len(n2) == 3:
                N2 = n2[0], n2[1], G.seed
            else:
                N2 = n2
            G.add_edge(N1, N2)
        return G


    def traverse():
        pass

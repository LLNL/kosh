from subprocess import Popen, PIPE
import os
import shlex
import sys
import kosh
import hashlib
import numpy
import networkx as nx
from .wrapper import KoshScriptWrapper  # noqa
from kosh.exec_graphs import find_network_ends


try:
    default_nx_layout = nx.planar_layout
except AttributeError:  # planar is available from nx version 2.5
    default_nx_layout = nx.circular_layout


def merge_datasets_handler(target_dataset, imported_dataset, **kargs):
    """When importing a dataset, checks if the imported dataset has
    attributes that match the one in the datset already in this store.
    If attributes values conflict then we use 'handling_method to resolve the conflict

    The store_dataset is not updated here, we return a list of attributes/values pairs resolving the conflict

    :param target_dataset: The dataset that will received the merge
    :type target_dataset: kosh.KoshDataset
    :param imported_dataset: The dataset we are trying to merge into target_dataset or its attributes/values dictionary
    :type imported_dataset: kosh.KoshDataset or dict
    :param handling_method: How do we handle conflicts?
                            None, "conservative": Error exit
                            "preserve": Keep value from target_dataset
                            "overwrite": Use value from imported dataset
    :returns: Dictionary of attribute/value that the target_dataset should have
    :rtype: dict
    """
    handling_method = kargs.pop("handling_method", None)

    target_dict = target_dataset.list_attributes(dictionary=True)

    if not isinstance(imported_dataset, dict):
        imported_dataset = imported_dataset.list_attributes(dictionary=True)

    for attribute, value in imported_dataset.items():
        if attribute in target_dict:
            if target_dict[attribute] != value:
                if handling_method in [None, "conservative"]:
                    msg = "Trying to import dataset with attribute '{}'".format(attribute)
                    msg += " value : {}. ".format(value)
                    msg += "But value for this attribute in target is '{}'".format(target_dict[attribute])
                    raise ValueError(msg)
                elif handling_method == "overwrite":
                    # Do we want a warning here?
                    # handling says use new value
                    target_dict[attribute] = value
                elif handling_method == "preserve":
                    # Do we want a warning here?
                    # We preserve so not changing the target value
                    pass
                else:
                    raise ValueError("Unknown 'handling_method': {}".format(handling_method))
        else:
            # New attribute let's add it
            target_dict[attribute] = value
    return target_dict


def gen_labels(G):
    """Generates labels to draw on networkx plots of a graph
    :param G: Network to generate labels from
    :type G: networkx.OrderedDiGraph
    :returns: labels for this graph
    :rtype: dict
    """
    labels = {}
    cont = True
    nodes = list(G.nodes())
    N = len(nodes)
    while cont:
        for node in nodes:
            if G.nodes[node].get("depth", None) is not None:
                continue
            pre = list(G.predecessors(node))
            suc = list(G.successors(node))
            if len(pre) == 0:
                G.nodes[node]["depth"] = 0
            else:
                for pnode in pre:
                    if G.nodes[pnode].get("depth", None) is not None:
                        G.nodes[node]["depth"] = G.nodes[pnode]["depth"] + 1
            if len(suc) == 0:
                G.nodes[node]["depth"] = -1
        total = 0
        for node in nodes:
            if G.nodes[node].get("depth", None) is not None:
                total += 1
        if total == N:
            cont = False
    for node in nodes:
        depth = G.nodes[node]["depth"]
        if depth == 0:
            depth = "start"
        elif depth == -1:
            depth = "end"
        try:
            name = node[1].__name__
        except BaseException:
            name = str(node[1].__class__).split(".")[-1].split("'")[0]
            if isinstance(node[1], kosh.loaders.core.KoshLoader):
                name = "{}({})".format(name, node[1].feature)
        labels[node] = "{}/{}/{}".format(depth, name, node[0])
    return labels


def draw_execution_graph(G,
                         output_format=None,
                         png_name="kosh_execution_graph.png",
                         clear=True,
                         layout=default_nx_layout):
    """Draws the graph and if provided an output format, draws the shortest path to it
    :param G: networkx graph or KoshExecutionGraph
    :type G: networkx.Graph
    :param output_format: draw shortest path to this format
    :type output_format: str or None
    :param png_name: name of png file to output the graph to
    :type png_name: str
    :param clear: clear matplotlib figure after saving
    :type clear: bool
    :param layout: A dictionary with nodes as keys and positions as values.
                   If not specified a {} layout positioning will be computed.
                   See networkx.drawing.layout for functions that compute node positions.
    :type layout: dict or function
    :returns: None but draws the matplotlib plt is updated and possibly saved
    :rtype: None
    """.format(default_nx_layout.__name__)
    if not isinstance(layout, dict):
        layout = layout(G)

    if isinstance(G, kosh.exec_graphs.KoshExecutionGraph):
        G = G.execution_graph()
    lbls_dict = gen_labels(G)
    nx.draw(
        G,
        pos=layout,
        with_labels=True,
        labels=lbls_dict,
        alpha=.5,
        node_size=150,
        edge_color='black',
        style="dashed")
    labels = nx.get_edge_attributes(G, 'weight')
    for k in labels:
        labels[k] = "{:.3g}".format(labels[k])
    nx.draw_networkx_edge_labels(G, layout, edge_labels=labels)
    if output_format is not None:
        starters = find_network_ends(G, start=True, end=False)
        for start in starters:
            pth = nx.shortest_path(G, start, (output_format, None, G.seed), weight="weight")
            # build edges
            edges = []
            for i in range(len(pth) - 1):
                edges.append((pth[i], pth[i + 1]))
            nx.draw(G, pos=layout, with_labels=True, labels=lbls_dict, nodelist=pth, edgelist=edges, edge_color='red')
    try:
        if "DISPLAY" not in os.environ or os.environ["DISPLAY"] == "":
            import matplotlib
            matplotlib.use("agg", force=True)
        import matplotlib.pyplot as plt
        plt.show()
        plt.savefig(png_name)
        if clear:
            plt.clf()
    except ImportError:
        raise RuntimeError("Could not import matplotlib, will not plot anything")


def compute_fast_sha(uri, n_samples=10):
    """Compute a fast 'almost' unique identifier for a given uri
    Assumes the uri is a path to a file, otherwise simply return hexdigest of md5 on the uri string

    If uri path is valid the 'fast' sha is used by creating an hashlib from
    * file size
    * file first 2kb
    * file last 2kb
    * 2k samples read from `n_samples` evenly spaced in the file

    Warning if size is unchanged and data is changed somewhere else than those samples the sha will be identical
    :param uri: URI to compute fast_sha on
    :type uri: str
    :param n_samples: Number of samples to extract from uri (in addition to beg and end of file)
    :type n_sampe: int
    :return sha: hexdigested sha
    :rtype: str
    """
    if not os.path.exists(uri):
        sha = hashlib.sha256(uri.encode())
        return sha.hexdigest()
    with open(uri, "rb") as f:
        stats = os.fstat(f.fileno())
        size = stats.st_size
        sha = hashlib.sha256("{}".format(size).encode())
        # Create list of start read
        positions = [int(max(x, 0))
                     for x in numpy.linspace(0, size - 2048, n_samples + 2)]
        prev = -1
        for pos in positions:
            # Small file will have multiple times the same bit to read
            if pos != prev:
                # Go there
                f.seek(pos)
                # read some small chunk
                st = f.read(2048)
                prev = pos
            sha.update(st)
    return sha.hexdigest()


def compute_long_sha(uri, buff_size=65536):
    """ Computes sha for a given uri
    :param uri: URI to compute fast_sha on
    :type uri: str
    :param buff_size: How much data to read at once
    :type buff_size: int
    :return sha: hexdigested sha
    :rtype: str
    """
    sha = hashlib.sha256()

    with open(uri, "rb") as f:
        while True:
            st = f.read(buff_size)
            if not st:
                break
                sha.update(st)
    return sha.hexdigest()


def create_new_db(name, engine='sina', db='sql',
                  token="", keyspace=None, cluster=None, **kargs):
    """create_new_db creates a new Kosh database, adds a single user

    :param name: name of database
    :type name: str
    :param engine: engine to use, defaults to 'sina'
    :type engine: str, optional
    :param db: type of database for engine, defaults to 'sql', can be 'cass'
    :type db: str, optional
    :param token: for cassandra connection, token to use, defaults to "" means try to retrieve from user home dir
    :type token: str, optional
    :param keyspace: for cassandra keyspace to use, defaults to None means [user]_k
    :type keyspace: str, optional
    :param cluster: list of Casandra clusters to use
    :type cluster: list of str
    :param kargs: Any additional key/value pairs you need to pass to store creation
    :type kargs: dict
    :return store: An handle to the Kosh store created
    :rtype: KoshStoreClass
    """
    user = os.environ["USER"]
    if db == 'sql' and name[-4:].lower() != ".sql":
        name += ".sql"
    if engine == "sina":
        cmd = "{}/init_sina.py --user={} --sina={} --sina_db={}".format(
            sys.prefix + "/bin",
            user,
            db,
            name)
    elif engine == 'cassandra':
        if keyspace is None:
            keyspace = user + "_k"
        cmd = "{}/init_cassandra.py --user={} --token={}" \
            "--keyspace={} --tables_root={} --cluster={}".format(
                sys.prefix + "/bin",
                user,
                token,
                keyspace,
                db,
                cluster)
    p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
    o, e = p.communicate()
    if engine == "sina":
        return kosh.KoshStore(engine="sina", db_uri=name, **kargs)

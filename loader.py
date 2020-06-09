import pickle
import networkx as nx
import numpy as np

name2num = {
    'input': 0,
    'output': 1,
    'conv1x1-bn-relu': 2,
    'conv3x3-bn-relu': 3,
    'maxpool3x3': 4,
}
num2name = {v: k for k, v in name2num.items()}


def load_from_pickle(path: str, n_samples: int,  log_validation=True):
    X = []
    Y = []
    dump = pickle.load(open(path, 'rb'))
    for i in range(n_samples):
        A = dump['model_graph_specs'][i]['adjacency']
        nodes_names = dump['model_graph_specs'][i]['node_labels']
        G = build_graph(A, nodes_names)
        y = dump['validation_err'][i]
        X.append(G)
        Y.append(np.log(y)) if log_validation else Y.append(y)
    return X, Y


def build_graph(A, nodes_names, create_directed_graph=True) -> nx.Graph:
    """A: adjacency matrix"""
    if create_directed_graph:
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    else:
        G = nx.from_numpy_array(A, create_using=nx.Graph)

    for i, n in enumerate(nodes_names):
        G.nodes[i]['op_name'] = name2num[n]

    return G

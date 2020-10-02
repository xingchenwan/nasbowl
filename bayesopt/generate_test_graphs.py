# Robin Ru | 5 March 2020
# Modifed from nas_benchmarks

import os

import ConfigSpace
import numpy as np
from copy import deepcopy
import collections
import random
import networkx as nx
import copy
import networkx.algorithms.isomorphism as iso
import logging
from kernels import GraphKernels, WeisfilerLehman
from .gp import GraphGP

# === For NASBench-101 ====
MAX_EDGES = 9
VERTICES = 7
OPS = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']

# === For NASBench-201 ===
MAX_EDGES_201 = None
VERTICES_201 = None
OPS_201 = ['nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3', 'skip_connect', 'none']


def prune(original_matrix, ops):
    """Prune the extraneous parts of the graph.

    General procedure:
      1) Remove parts of graph not connected to input.
      2) Remove parts of graph not connected to output.
      3) Reorder the vertices so that they are consecutive after steps 1 and 2.

    These 3 steps can be combined by deleting the rows and columns of the
    vertices that are not reachable from both the input and output (in reverse).
    """
    num_vertices = np.shape(original_matrix)[0]
    new_matrix = copy.deepcopy(original_matrix)
    new_ops = copy.deepcopy(ops)
    # DFS forward from input
    visited_from_input = {0}
    frontier = [0]
    while frontier:
        top = frontier.pop()
        for v in range(top + 1, num_vertices):
            if original_matrix[top, v] and v not in visited_from_input:
                visited_from_input.add(v)
                frontier.append(v)

    # DFS backward from output
    visited_from_output = {num_vertices - 1}
    frontier = [num_vertices - 1]
    while frontier:
        top = frontier.pop()
        for v in range(0, top):
            if original_matrix[v, top] and v not in visited_from_output:
                visited_from_output.add(v)
                frontier.append(v)

    # Any vertex that isn't connected to both input and output is extraneous to
    # the computation graph.
    extraneous = set(range(num_vertices)).difference(
        visited_from_input.intersection(visited_from_output))

    # If the non-extraneous graph is less than 2 vertices, the input is not
    # connected to the output and the spec is invalid.
    if len(extraneous) > num_vertices - 2:
        new_matrix = None
        new_ops = None
        valid_spec = False
        return

    new_matrix = np.delete(new_matrix, list(extraneous), axis=0)
    new_matrix = np.delete(new_matrix, list(extraneous), axis=1)
    for index in sorted(extraneous, reverse=True):
        del new_ops[index]

    return new_matrix, new_ops


def get_nas101_configuration_space():
    # NAS-CIFAR10 A
    nas101_cs = ConfigSpace.ConfigurationSpace()

    nas101_cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_0", OPS))
    nas101_cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_1", OPS))
    nas101_cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_2", OPS))
    nas101_cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_3", OPS))
    nas101_cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_4", OPS))
    for i in range(VERTICES * (VERTICES - 1) // 2):
        nas101_cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("edge_%d" % i, [0, 1]))
    return nas101_cs


# === For NASBench-201 ====
def create_nasbench201_graph(op_node_labelling, edge_attr=False):
    assert len(op_node_labelling) == 6
    # the graph has 8 nodes (6 operation nodes + input + output)
    G = nx.DiGraph()
    if edge_attr:
        edge_list = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
        G.add_edges_from(edge_list)
        edge_attribute = {}
        remove_edge_list = []
        for i, edge in enumerate(edge_list):
            edge_attribute[edge] = {'op_name': op_node_labelling[i]}
            if op_node_labelling[i] == 'none':
                remove_edge_list.append(edge)
        nx.set_edge_attributes(G, edge_attribute)
        G.remove_edges_from(remove_edge_list)

        # after removal, some op nodes have no input nodes and some have no output nodes
        # --> remove these redundant nodes
        nodes_to_be_further_removed = []
        for n_id in G.nodes():
            in_edges = G.in_edges(n_id)
            out_edges = G.out_edges(n_id)
            if n_id != 0 and len(in_edges) == 0:
                nodes_to_be_further_removed.append(n_id)
            elif n_id != 3 and len(out_edges) == 0:
                nodes_to_be_further_removed.append(n_id)

        G.remove_nodes_from(nodes_to_be_further_removed)
        # Assign dummy variables as node attributes:
        for i in G.nodes:
            G.nodes[i]['op_name'] = "1"
        G.graph_type = 'edge_attr'
    else:
        edge_list = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 6), (3, 6), (4, 7), (5, 7), (6, 7)]
        G.add_edges_from(edge_list)

        # assign node attributes and collate the information for nodes to be removed
        # (i.e. nodes with 'skip_connect' or 'none' label)
        node_labelling = ['input'] + op_node_labelling + ['output']
        nodes_to_remove_list = []
        remove_nodes_list = []
        edges_to_add_list = []
        for i, n in enumerate(node_labelling):
            G.nodes[i]['op_name'] = n
            if n == 'none' or n == 'skip_connect':
                input_nodes = [edge[0] for edge in G.in_edges(i)]
                output_nodes = [edge[1] for edge in G.out_edges(i)]
                nodes_to_remove_info = {'id': i, 'input_nodes': input_nodes, 'output_nodes': output_nodes}
                nodes_to_remove_list.append(nodes_to_remove_info)
                remove_nodes_list.append(i)

                if n == 'skip_connect':
                    for n_i in input_nodes:
                        edges_to_add = [(n_i, n_o) for n_o in output_nodes]
                        edges_to_add_list += edges_to_add

        # reconnect edges for removed nodes with 'skip_connect'
        G.add_edges_from(edges_to_add_list)

        # remove nodes with 'skip_connect' or 'none' label
        G.remove_nodes_from(remove_nodes_list)

        # after removal, some op nodes have no input nodes and some have no output nodes
        # --> remove these redundant nodes
        nodes_to_be_further_removed = []
        for n_id in G.nodes():
            in_edges = G.in_edges(n_id)
            out_edges = G.out_edges(n_id)
            if n_id != 0 and len(in_edges) == 0:
                nodes_to_be_further_removed.append(n_id)
            elif n_id != 7 and len(out_edges) == 0:
                nodes_to_be_further_removed.append(n_id)

        G.remove_nodes_from(nodes_to_be_further_removed)
        G.graph_type = 'node_attr'

    # create the arch string for querying nasbench dataset
    arch_query_string = f'|{op_node_labelling[0]}~0|+' \
                        f'|{op_node_labelling[1]}~0|{op_node_labelling[2]}~1|+' \
                        f'|{op_node_labelling[3]}~0|{op_node_labelling[4]}~1|{op_node_labelling[5]}~2|'

    G.name = arch_query_string
    return G


def get_nas201_configuration_space():
    # for unpruned graph
    cs = ConfigSpace.ConfigurationSpace()
    ops_choices = ['nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3', 'skip_connect', 'none']
    for i in range(6):
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("edge_%d" % i, ops_choices))
    return cs


# Regularised evolution to generate new graphs
def mutate_arch(parent_arch, benchmark, return_unpruned_arch=True):
    if benchmark == 'nasbench101':
        # get parent_arch node label and adjacency matrix
        child_arch = deepcopy(parent_arch)
        node_labeling = list(nx.get_node_attributes(child_arch, 'op_name').values())
        adjacency_matrix = np.array(nx.adjacency_matrix(child_arch).todense())

        parent_node_labeling = deepcopy(node_labeling)
        parent_adjacency_matrix = deepcopy(adjacency_matrix)

        dim_op_labeling = (len(node_labeling) - 2)
        dim_adjacency_matrix = adjacency_matrix.shape[0] * (adjacency_matrix.shape[0] - 1) // 2

        mutation_failed = True

        while mutation_failed:
            # pick random parameter
            dim = np.random.randint(dim_op_labeling + dim_adjacency_matrix)

            if dim < dim_op_labeling:
                choices = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
                node_number = int(dim + 1)
                parent_choice = node_labeling[node_number]

                # drop current values from potential choices
                choices.remove(parent_choice)

                # flip parameter
                choice_idx = np.random.randint(len(choices))
                node_labeling[node_number] = choices[choice_idx]

            else:
                choices = [0, 1]
                # find the corresponding row and colum in adjacency matrix
                idx = np.triu_indices(adjacency_matrix.shape[0], k=1)
                edge_i = int(dim - dim_op_labeling)
                row = idx[0][edge_i]
                col = idx[1][edge_i]
                parent_choice = adjacency_matrix[row, col]

                # drop current values from potential choices
                choices.remove(parent_choice)

                # flip parameter
                choice_idx = np.random.randint(len(choices))
                adjacency_matrix[row, col] = choices[choice_idx]

            try:
                pruned_adjacency_matrix, pruned_node_labeling = prune(adjacency_matrix, node_labeling)
                mutation_failed = False
            except:
                continue
            # if pruned_adjacency_matrix.all() != parent_adjacency_matrix.all() or pruned_node_labeling != parent_node_labeling:
            #

        child_arch = nx.from_numpy_array(pruned_adjacency_matrix, create_using=nx.DiGraph)

        for i, n in enumerate(pruned_node_labeling):
            child_arch.nodes[i]['op_name'] = n
        if return_unpruned_arch:
            child_arch_unpruned = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
            for i, n in enumerate(node_labeling):
                child_arch_unpruned.nodes[i]['op_name'] = n

    elif benchmark == 'nasbench201':
        arch_str_list = parent_arch.name.split('|')
        op_label_rebuild = []
        for str_i in arch_str_list:
            if '~' in str_i:
                op_label_rebuild.append(str_i[:-2])

        # pick random parameter
        dim = np.random.randint(len(op_label_rebuild))
        parent_choice = op_label_rebuild[dim]

        # drop current values from potential choices
        ops_choices = OPS_201[:]
        ops_choices.remove(parent_choice)

        # flip parameter
        choice_idx = np.random.randint(len(ops_choices))
        op_label_rebuild[dim] = ops_choices[choice_idx]
        child_arch = create_nasbench201_graph(op_label_rebuild, edge_attr=parent_arch.graph_type == 'edge_attr')

        if return_unpruned_arch:
            child_arch_unpruned = child_arch

    if return_unpruned_arch:
        return child_arch, child_arch_unpruned

    return child_arch, None


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def guided_mutate_arch101(parent_arch, feature_names, features_grad, return_unpruned_arch=True,
                          unpruned_parent_arch=None,
                          change_grad_sign=True,
                          like_bananas=True  ,
                          verbose=False):
    """

    Parameters
    ----------
    parent_arch: nx.DiGraph. The parent architecture in nx directed graph
    benchmark: the data-set of the nasbench dataset. nasbench101
    feature_names
    features_grad
    return_unpruned_arch
    unpruned_parent_arch: if supplied, we will mutate the unpruned architectures instead of the pruned
        architectures. however, the gradient of the features will still be computed wrt the pruned
        architecture.
    change_grad_sign
    like_bananas
    verbose

    Returns
    -------
    Tuple (pruned_architecture: nx.DiGraph, unpruned_architecture: nx.DiGraph or None).
    Note: if unpruned_parent_arch is not suplied or the return_unpruned_arch flag is False, the second
    return value is None.
    """
    # combine feature_names, features_grad into one dic
    # the gradient w.r.t all features seen so far both test arch and observed arch
    all_features = {}
    # for feature in feature_names.items():
    #     idx, feature_name = feature
    #     if change_grad_sign:
    #         all_features[feature_name] = - features_grad[idx]
    #     else:
    #         all_features[feature_name] = features_grad[idx]
    for idx, feature_name in enumerate(feature_names):
        if change_grad_sign:
            features_grad_idx = - features_grad[idx]
        else:
            features_grad_idx = features_grad[idx]

        if feature_name.count('~') <=1:
            all_features[feature_name] = features_grad_idx

        else:
            feature_name_list = feature_name.split('~')
            for idx2 in range(1, len(feature_name_list)):
                all_features[f'{feature_name_list[0]}~{feature_name_list[idx2]}'] = features_grad_idx

    # find the grad for all the op choices
    op_list = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
    op_grad_list = []
    for op in op_list:
        if op in feature_names:
            op_grad_value = all_features[op]
        else:
            op_grad_value = 0
        op_grad_list.append(op_grad_value)
    # op_grad_list = [all_features[op] for op in op_list if op in feature_names]

    # define all possible edges in the parent graph and encode them as "in_node_op_name~out_node_op_name"
    edges_pool = [(ni, no) for ni in range(parent_arch.number_of_nodes() - 1) for no in
                  range(ni + 1, parent_arch.number_of_nodes())]
    edge_encode_dic = {}
    for j1, e in enumerate(edges_pool):
        in_node, out_node = e
        edge_encode = parent_arch.nodes[in_node]['op_name'] + '~' + parent_arch.nodes[out_node]['op_name']
        edge_encode_dic[e] = edge_encode

    # encode each node with its op_name
    node_encode_dic = {}
    for j2, node in enumerate(parent_arch.nodes):
        node_encode = parent_arch.nodes[node]['op_name']
        node_encode_dic[node] = node_encode

    # the collate the overall probs of being chosen for all edges and nodes
    edge_prob_dic = {}
    for edge_item in edge_encode_dic.items():
        edge_key = edge_item[0]
        edge_encode_str = edge_item[1]
        edge_grads = [all_features[key] for key in list(all_features.keys()) if edge_encode_str in key]
        # Change to edge_grads = [] so that the mutation probability of edge is not affected.
        # edge_grads = []
        edge_grads_sum = np.sum(edge_grads)
        edge_prob_dic[edge_key] = edge_grads_sum

    node_prob_dic = {}
    for node_item in node_encode_dic.items():
        node_key = node_item[0]
        node_encode_str = node_item[1]
        if node_encode_str == 'output' or node_encode_str == 'input':
            continue
        else:
            node_grads = [all_features[node_encode_str]]
            node_grads_sum = np.sum(node_grads)
            node_prob_dic[node_key] = node_grads_sum

    # define the prob for randomly choosing 1 node or edge to mutate
    node_edge_list = list(node_prob_dic.keys()) + list(edge_prob_dic.keys())
    node_edge_prob_list = list(node_prob_dic.values()) + list(edge_prob_dic.values())

    node_edge_prob_array = np.array(node_edge_prob_list)
    normalised_node_edge_probs = np.exp(node_edge_prob_array) / np.sum(np.exp(node_edge_prob_array))
    # normalised_node_edge_probs = sigmoid(node_edge_prob_array) / np.sum(sigmoid(node_edge_prob_array))

    # start mutate
    if unpruned_parent_arch is not None:
        child_arch = deepcopy(unpruned_parent_arch)
    else:
        child_arch = deepcopy(parent_arch)
    if like_bananas:
        # loop through all node and possible edges in the parent arch to decide whether to mutate any of them
        for i, item in enumerate(node_edge_list):
            prob_of_mutation = normalised_node_edge_probs[i]

            if random.random() < prob_of_mutation:
                if type(item) == tuple:
                    if verbose: print(f'mutate edge {item}')
                    # an edge is chosen
                    if item in child_arch.edges:
                        child_arch.remove_edges_from([item])
                    else:
                        child_arch.add_edges_from([item])
                else:
                    if verbose: print(f'mutate node {item}')
                    # a node is chosen
                    op_list_copy = op_list.copy()
                    op_grad_list_copy = op_grad_list.copy()
                    # remove the current node operation from all op choices
                    parent_node_op = node_encode_dic[item]
                    index_to_remove_from_opt_list = op_list_copy.index(parent_node_op)
                    op_list_copy.pop(index_to_remove_from_opt_list)
                    op_grad_list_copy.pop(index_to_remove_from_opt_list)

                    # choose among the remaining op choices
                    op_grad_array = np.array(op_grad_list_copy)
                    # normalised_op_prob = sigmoid(op_grad_array) / np.sum(sigmoid(op_grad_array))
                    normalised_op_prob = np.exp(op_grad_array) / np.sum(np.exp(op_grad_array))
                    try:
                        choosen_op = random.choices(population=op_list_copy, weights=normalised_op_prob, k=1)
                    except:
                        print('hold')
                    child_arch.nodes[item]['op_name'] = choosen_op[0]

    else:
        # pick one of all node and possible edges in the parent arch to mutate
        choosen_item = random.choices(population=node_edge_list, weights=normalised_node_edge_probs,
                                      k=1)  # choosen_item is a list of k item
        if type(choosen_item[0]) == tuple:
            if verbose: print(f'mutate edge {choosen_item[0]}')
            # an edge is chosen
            if choosen_item[0] in child_arch.edges:
                child_arch.remove_edges_from(choosen_item)
            else:
                child_arch.add_edges_from(choosen_item)
        else:
            # print(f'mutate node {choosen_item[0]}')
            # a node is chosen
            op_list_copy = op_list.copy()
            op_grad_list_copy = op_grad_list.copy()
            # remove the current node operation from all op choices
            parent_node_op = node_encode_dic[choosen_item[0]]
            index_to_remove_from_opt_list = op_list_copy.index(parent_node_op)
            op_list_copy.pop(index_to_remove_from_opt_list)
            op_grad_list_copy.pop(index_to_remove_from_opt_list)

            # choose among the remaining op choices
            op_grad_array = np.array(op_grad_list_copy)
            # normalised_op_prob = sigmoid(op_grad_array) / np.sum(sigmoid(op_grad_array))
            normalised_op_prob = np.exp(op_grad_array) / np.sum(np.exp(op_grad_array))
            choosen_op = random.choices(population=op_list_copy, weights=normalised_op_prob, k=1)
            child_arch.nodes[choosen_item[0]]['op_name'] = choosen_op[0]

    # pruning
    node_labeling = list(nx.get_node_attributes(child_arch, 'op_name').values())
    adjacency_matrix = np.array(nx.adjacency_matrix(child_arch).todense())
    try:
        pruned_adjacency_matrix, pruned_node_labeling = prune(adjacency_matrix, node_labeling)
        child_arch = nx.from_numpy_array(pruned_adjacency_matrix, create_using=nx.DiGraph)
        for i, n in enumerate(pruned_node_labeling):
            child_arch.nodes[i]['op_name'] = n
    except:
        # Mutation failed, return None for both the pruned and unpruned architectures.
        return None, None

    if return_unpruned_arch:
        child_arch_unpruned = nx.from_numpy_array(adjacency_matrix, node_labeling, create_using=nx.DiGraph)
        for i, n in enumerate(node_labeling):
            child_arch_unpruned.nodes[i]['op_name'] = n
        return child_arch, child_arch_unpruned
    return child_arch, None


def guided_mutate_arch201(parent_arch, feature_names, features_grad, return_unpruned_arch=True,
                          unpruned_parent_arch=None,
                          change_grad_sign=True, like_bananas=True,
                          verbose=False, debug_feature_names=None):
    """

    Parameters
    ----------
    parent_arch: nx.DiGraph. The parent architecture in nx directed graph
    benchmark: the data-set of the nasbench dataset. nasbench201
    feature_names
    features_grad
    return_unpruned_arch
    unpruned_parent_arch: if supplied, we will mutate the unpruned architectures instead of the pruned
        architectures. however, the gradient of the features will still be computed wrt the pruned
        architecture.
    change_grad_sign
    like_bananas
    verbose

    Returns
    -------
    Tuple (pruned_architecture: nx.DiGraph, unpruned_architecture: nx.DiGraph or None).
    Note: if unpruned_parent_arch is not suplied or the return_unpruned_arch flag is False, the second
    return value is None.
    """
    # combine feature_names, features_grad into one dic
    # the gradient w.r.t all features seen so far both test arch and observed arch
    all_features = {}
    # for feature in feature_names.items():
    #     idx, feature_name = feature
    #     if change_grad_sign:
    #         all_features[feature_name] = - features_grad[idx]
    #     else:
    #         all_features[feature_name] = features_grad[idx]

    for idx, feature_name in enumerate(feature_names):
        if change_grad_sign:
            features_grad_idx = - features_grad[idx]
        else:
            features_grad_idx = features_grad[idx]

        if feature_name.count('~') <=1:
            all_features[feature_name] = features_grad_idx

        else:
            feature_name_list = feature_name.split('~')
            for idx2 in range(1, len(feature_name_list)):
                all_features[f'{feature_name_list[0]}~{feature_name_list[idx2]}'] = features_grad_idx

    # find the grad for all the op choices
    # OPS_201 = ['nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3', 'skip_connect', 'none']
    all_ops_grad_dic = {'nor_conv_3x3': [], 'nor_conv_1x1': [], 'avg_pool_3x3': [], 'skip_connect': [], 'none': []}

    # build unpruned nasbench201 graphs
    G = nx.DiGraph()
    edge_list = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 6), (3, 6), (4, 7), (5, 7), (6, 7)]
    G.add_edges_from(edge_list)
    arch_str_list = parent_arch.name.split('|')
    op_node_labelling = []
    for str_i in arch_str_list:
        if '~' in str_i:
            op_node_labelling.append(str_i[:-2])
    node_labelling = ['input'] + op_node_labelling + ['output']

    # compute grad for each node in unpruned graph (i.e. prob of mutation)
    node_prob_dic = {}
    for i, n in enumerate(node_labelling):
        G.nodes[i]['op_name'] = n

        if n == 'input' or n == 'output':
            continue
        elif n == 'none' or n == 'skip_connect':
            input_nodes = [edge[0] for edge in G.in_edges(i)]
            output_nodes = [edge[1] for edge in G.out_edges(i)]

            # collect related feature names
            for n_i in input_nodes:
                related_feature_names = [node_labelling[n_i] + '~' + node_labelling[n_o] for n_o in output_nodes]

            # collect the grads for the related feature TODO check this line!
            # feature_grad * z_feature_exist_in_graph = feature_relevant
            # related_feature_grads = []
            # for feature_name in related_feature_names:
            #     feature_grad_list = [all_features[key] for key in list(all_features.keys()) if feature_name in key]
            #     related_feature_grads += feature_grad_list
            # feature_grad_sum = np.sum(related_feature_grads)

            related_feature_grads = 0
            if n == 'skip_connect':
                for edge_feature_name in related_feature_names:
                    try:
                        edge_feature_grad = all_features[edge_feature_name]
                        related_feature_grads += edge_feature_grad
                    except:
                        # if the feature hasn't appeared in the training set yet
                        related_feature_grads += 0

            feature_grad_sum = related_feature_grads
        else:
            try:
                feature_grad_sum = all_features[n]
            except:
                feature_grad_sum = 0
                # print('hold')
        node_prob_dic[i] = feature_grad_sum
        all_ops_grad_dic[n].append(feature_grad_sum)

    # define the prob for all potential op choices:
    all_ops_prob_dic = {}
    for item in all_ops_grad_dic.items():
        op_key = item[0]
        op_grad_list = item[1]
        if len(op_grad_list) == 0:
            all_ops_prob_dic[op_key] = 0.0
        else:
            all_ops_prob_dic[op_key] = np.mean(op_grad_list)

    # define the prob for randomly choosing 1 node or edge to mutate
    unpruned_node_list = list(node_prob_dic.keys())
    unpruned_node_prob_list = list(node_prob_dic.values())

    unpruned_node_prob_array = np.array(unpruned_node_prob_list)
    # normalised_node_probs = sigmoid(unpruned_node_prob_array) / np.sum(sigmoid(unpruned_node_prob_array))
    normalised_node_probs = np.exp(unpruned_node_prob_array) / np.sum(np.exp(unpruned_node_prob_array))

    # start mutate
    child_arch = deepcopy(G)
    if like_bananas:
        # loop through all node and possible edges in the parent arch to decide whether to mutate any of them
        for i, item in enumerate(unpruned_node_list):
            prob_of_mutation = normalised_node_probs[i]

            if random.random() < prob_of_mutation:
                if verbose: print(f'mutate node {item}')
                # a node is chosen
                all_ops_prob_dic_copy = all_ops_prob_dic.copy()
                # remove the current node operation from all op choices
                all_ops_prob_dic_copy.pop(G.nodes[item]['op_name'])

                # choose among the remaining op choices
                op_list_copy = list(all_ops_prob_dic_copy.keys())
                op_grad_array = np.array(list(all_ops_prob_dic_copy.values()))
                # normalised_op_prob = sigmoid(op_grad_array) / np.sum(sigmoid(op_grad_array))
                normalised_op_prob = np.exp(op_grad_array) / np.sum(np.exp(op_grad_array))
                choosen_op = random.choices(population=op_list_copy, weights=normalised_op_prob, k=1)
                child_arch.nodes[item]['op_name'] = choosen_op[0]
                node_labelling[item] = choosen_op[0]


    else:
        # pick one of all node and possible edges in the parent arch to mutate
        choosen_item = random.choices(population=unpruned_node_list, weights=normalised_node_probs,
                                      k=1)  # choosen_item is a list of k item

        if verbose: print(f'mutate node {choosen_item[0]}')
        # a node is chosen
        all_ops_prob_dic_copy = all_ops_prob_dic.copy()
        # remove the current node operation from all op choices
        all_ops_prob_dic_copy.pop(G.nodes[choosen_item[0]]['op_name'])

        # choose among the remaining op choices
        op_list_copy = list(all_ops_prob_dic_copy.keys())
        op_grad_array = np.array(list(all_ops_prob_dic_copy.values()))
        # normalised_op_prob = sigmoid(op_grad_array) / np.sum(sigmoid(op_grad_array))
        normalised_op_prob = np.exp(op_grad_array) / np.sum(np.exp(op_grad_array))
        choosen_op = random.choices(population=op_list_copy, weights=normalised_op_prob, k=1)
        child_arch.node[choosen_item[0]]['op_name'] = choosen_op[0]
        node_labelling[choosen_item[0]] = choosen_op[0]

    # pruning
    try:
        child_arch_pruned = create_nasbench201_graph(node_labelling[1:-1], edge_attr=False)
    except:
        # Mutation failed, return None for both the pruned and unpruned architectures.
        return None, None

    if return_unpruned_arch:
        child_arch_unpruned = child_arch
        # create the arch string for querying nasbench dataset
        arch_query_string = f'|{node_labelling[1]}~0|+' \
                            f'|{node_labelling[2]}~0|{node_labelling[3]}~1|+' \
                            f'|{node_labelling[4]}~0|{node_labelling[5]}~1|{node_labelling[6]}~2|'

        child_arch_unpruned.name = arch_query_string

        return child_arch_pruned, child_arch_unpruned
    return child_arch_pruned, None


def regularized_evolution(acquisition_func,
                          observed_archs,
                          observed_archs_unpruned=None,
                          benchmark='nasbench101',
                          pool_size=200, cycles=40, n_mutation=10, batch_size=1,
                          mutate_unpruned_arch=True):
    """Algorithm for regularized evolution (i.e. aging evolution).

    Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image
    Classifier Architecture Search".

    """
    # Generate some random archs into the evaluation pool
    if mutate_unpruned_arch and observed_archs_unpruned is None:
        raise ValueError("When mutate_unpruned_arch option is toggled on, you need to supplied the list of unpruned "
                         "observed architectures.")
    if observed_archs_unpruned is not None:
        assert len(observed_archs_unpruned) == len(observed_archs), " unequal length between the pruned/unpruned " \
                                                                    "architecture lists"

    n_random_archs = pool_size - len(observed_archs)
    if mutate_unpruned_arch:
        (random_archs, _, random_archs_unpruned) = random_sampling(pool_size=n_random_archs, benchmark=benchmark,
                                                                   return_unpruned_archs=True)
        population_unpruned = observed_archs_unpruned + random_archs_unpruned
    else:
        (random_archs, _, _) = random_sampling(pool_size=n_random_archs, benchmark=benchmark, )
        population_unpruned = None
    population = observed_archs + random_archs

    # Fill the population with the observed archs (a list of labelled graphs) and validation error
    population_performance = []
    for i, archs in enumerate(population):
        arch_acq = acquisition_func.eval(archs, asscalar=True)
        population_performance.append(arch_acq)

    # Carry out evolution in cycles. Each cycle produces a bat model and removes another.
    k_cycle = 0

    while k_cycle < cycles:
        # Sample randomly chosen models from the current population based on the acquisition function values
        pseudo_prob = np.array(population_performance) / (np.sum(population_performance))
        if mutate_unpruned_arch:
            samples = random.choices(population_unpruned, weights=pseudo_prob, k=30)
            sample_indices = [population_unpruned.index(s) for s in samples]
        else:
            samples = random.choices(population, weights=pseudo_prob, k=30)
            sample_indices = [population.index(s) for s in samples]
        sample_performance = [population_performance[idx] for idx in sample_indices]

        # The parents is the best n_mutation model in the sample. skip 2-node archs
        top_n_mutation_archs_indices = np.argsort(sample_performance)[-n_mutation:]  # argsort>ascending
        parents_archs = [samples[idx] for idx in top_n_mutation_archs_indices if len(samples[idx].nodes) > 3]

        # Create the child model and store it.
        for parent in parents_archs:
            child, child_unpruned = mutate_arch(parent, benchmark)
            # skip invalid architectures whose number of edges exceed the max limit of 9
            if np.sum(nx.to_numpy_array(child)) > MAX_EDGES:
                continue
            if iso.is_isomorphic(child, parent):
                continue

            skip = False
            for prev_edit in population:
                if iso.is_isomorphic(child, prev_edit, ):
                    skip = True
                    break
            if skip: continue
            child_arch_acq = acquisition_func.eval(child, asscalar=True)
            population.append(child)
            if mutate_unpruned_arch:
                population_unpruned.append(child_unpruned)
            population_performance.append(child_arch_acq)

        # Remove the worst performing model and move to next evolution cycle
        worst_n_mutation_archs_indices = np.argsort(population_performance)[:n_mutation]
        for bad_idx in sorted(worst_n_mutation_archs_indices, reverse=True):
            population.pop(bad_idx)
            population_performance.pop(bad_idx)
            if mutate_unpruned_arch:
                population_unpruned.pop(bad_idx)
            # print(f'len pop = {len(population)}')
        k_cycle += 1

    # choose batch_size archs with highest acquisition function values to be evaluated next
    best_archs_indices = np.argsort(population_performance)[-batch_size:]
    recommended_pool = [population[best_idx] for best_idx in best_archs_indices]
    if mutate_unpruned_arch:
        recommended_pool_unpruned = [population_unpruned[best_idx] for best_idx in best_archs_indices]
        return (recommended_pool, recommended_pool_unpruned), (population, population_unpruned, population_performance)
    return (recommended_pool, None), (population, None, population_performance)


class Model(object):
    """A class representing a model.
    """

    def __init__(self):
        self.arch = None
        self.unpruned_arch = None
        self.error = None

    def __str__(self):
        """Prints a readable version of this bitstring."""
        return '{0:b}'.format(self.arch)


# def regularized_evolution2(observed_archs,
#                            observed_errors, pool_size=100, cycles=5, sample_size=10,
#                            observed_archs_unpruned=None,
#                            benchmark='nasbench101'):
#     """Algorithm for regularized evolution (i.e. aging evolution).
#
#     Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image
#     Classifier Architecture Search".
#
#     """
#
#     population = collections.deque()
#     evaluation_pool = []  # A list of graphs whose acquisition function values to be evaluated
#     evaluation_pool_unpruned = []  # Same as above, but unpruned architectures.
#     # Fill the population with the observed archs (a list of labelled graphs) and validation error
#     for i, archs in enumerate(observed_archs):
#         model = Model()
#         model.arch = archs
#         model.unpruned_arch = observed_archs_unpruned[i]
#         model.error = observed_errors[i]
#         population.append(model)
#     # Carry out evolution in cycles. Each cycle produces a model and removes another.
#     k_cycle = 0
#     # cycles = min(cycles, len(population))
#     if cycles is None:
#         cycles = len(population)
#
#     while k_cycle < cycles:
#         sample = random.choices(population, k=sample_size)
#
#         # The parent is the best model in the sample.
#         parent = max(sample, key=lambda i: i.error)
#
#         # skip 2-node archs
#         if len(parent.arch.nodes) <= 3:
#             continue
#
#         # Create the child model and store it.
#         child = Model()
#         child.arch, child.unpruned_arch = mutate_arch(parent.unpruned_arch if observed_archs_unpruned is not None
#                                                       else parent.arch, benchmark=benchmark,
#                                                       return_unpruned_arch=observed_archs_unpruned is not None)
#         if np.sum(np.array(nx.adjacency_matrix(child.arch).todense())) > MAX_EDGES:
#             continue
#
#         child.error = 0  # assign a fake and low error to child to prevent it being a parent
#         population.append(child)
#         # population_performance.append(0)
#         # Remove the oldest model and move to next evolution cycle
#         population.popleft()
#         # population_performance = population_performance [1:] #Same as popleft
#         k_cycle += 1
#
#     # Add the updated population to the evaluation pool
#     for individual in population:
#         evaluation_pool.append(individual.arch)
#         evaluation_pool_unpruned.append(individual.unpruned_arch)
#
#     # Add some random archs into the evaluation pool
#     nrandom_archs = max(pool_size - len(evaluation_pool), 0)
#     if nrandom_archs:
#         random_evaluation_pool, _, random_evaluation_pool_unpruned = random_sampling(pool_size=nrandom_archs,
#                                                                                      benchmark=benchmark,
#                                                                                      return_unpruned_archs=True)
#         evaluation_pool += random_evaluation_pool
#         evaluation_pool_unpruned += random_evaluation_pool_unpruned
#     if observed_archs_unpruned is None:
#         return evaluation_pool, None
#     return evaluation_pool, evaluation_pool_unpruned


def mutation(observed_archs, observed_errors,
             n_best=10,
             n_mutate=None,
             pool_size=250,
             allow_isomorphism=False,
             patience=50,
             benchmark='nasbench101', observed_archs_unpruned=None):
    """
    BANANAS-style mutation.
    The main difference with the previously implemented evolutionary algorithms is that it allows unlimited number of
    edits based on the parent architecture. For previous implementations, we only allow 1-edit distance mutation.
    (although *in expectation*, there is only one edit.)
    """
    if n_mutate is None:
        n_mutate = int(0.5 * pool_size)
    assert pool_size >= n_mutate, " pool_size must be larger or equal to n_mutate"

    def _banana_mutate(arch, mutation_rate=1.0, benchmark='nasbench101'):
        """Bananas Style mutation of the cell"""
        if benchmark == 'nasbench201':
            parent_arch_list = arch.name.split("|")
            op_label_rebuild = [str_i[:-2] for str_i in parent_arch_list if "~" in str_i]
            mutation_prob = mutation_rate / len(OPS_201)
            child = None
            while True:
                try:
                    for idx, parent_choice in enumerate(op_label_rebuild):
                        ops_choices = OPS_201[:]
                        ops_choices.remove(parent_choice)
                        # Flip parameter with probability of mutation prob
                        if random.random() < mutation_prob:
                            choice_idx = np.random.randint(len(ops_choices))
                            op_label_rebuild[idx] = ops_choices[choice_idx]
                    child = create_nasbench201_graph(op_label_rebuild, edge_attr=arch.graph_type == 'edge_attr')
                    break
                except:
                    continue
            child_unpruned = child
            return child, child_unpruned
        elif benchmark == 'nasbench101':
            while True:
                new_ops = list(nx.get_node_attributes(arch, 'op_name').values())
                new_matrix = np.array(nx.adjacency_matrix(arch).todense())
                vertice = min(VERTICES, new_matrix.shape[0])

                if vertice > 2:
                    edge_mutation_prob = mutation_rate / vertice
                    for src in range(0, vertice - 1):
                        for dst in range(src + 1, vertice):
                            if random.random() < edge_mutation_prob:
                                new_matrix[src, dst] = 1 - new_matrix[src, dst]

                    op_mutation_prob = mutation_rate / (vertice - 2)
                    for ind in range(1, vertice - 1):
                        if random.random() < op_mutation_prob:
                            available = [o for o in OPS if o != new_ops[ind]]
                            new_ops[ind] = random.choice(available)
                try:
                    pruned_adjacency_matrix, pruned_node_labeling = prune(new_matrix, new_ops)
                    child_arch = nx.from_numpy_array(pruned_adjacency_matrix, create_using=nx.DiGraph)
                    child_arch_unpruned = nx.from_numpy_array(new_matrix, create_using=nx.DiGraph)
                    for i, n in enumerate(pruned_node_labeling):
                        child_arch.nodes[i]['op_name'] = n
                    for i, n in enumerate(new_ops):
                        child_arch_unpruned.nodes[i]['op_name'] = n
                    return child_arch, child_arch_unpruned
                except:
                    continue
        else:
            raise NotImplementedError("Search space " + str(benchmark) + " not implemented!")

    population = collections.deque()
    # Fill the population with the observed archs (a list of labelled graphs) and validation error
    for i, archs in enumerate(observed_archs):
        model = Model()
        model.arch = archs
        model.unpruned_arch = observed_archs_unpruned[i] if observed_archs_unpruned is not None else None
        model.error = observed_errors[i]
        population.append(model)

    best_archs = [arch for arch in sorted(population, key=lambda i: -i.error)][:n_best]
    # pools = [[] for _ in range(len(best_archs))]
    # unpruned_pools = [[] for _ in range(len(best_archs))]
    evaluation_pool, evaluation_pool_unpruned = [], []
    per_arch = n_mutate // n_best
    for arch in best_archs:
        n_child = 0
        patience_ = patience
        while n_child < per_arch and patience_ > 0:
            child = Model()
            child.arch, child.unpruned_arch = _banana_mutate(arch.unpruned_arch if observed_archs_unpruned is not None
                                                             else arch.arch, benchmark=benchmark, )
            # skip 2-node archs
            if benchmark == 'nasbench101':
                if len(child.arch.nodes) <= 3:  # Skip input-output 3-graphs in nasbench101 search space
                    patience_ -= 1
                    continue
                if np.sum(np.array(nx.adjacency_matrix(child.arch).todense())) > MAX_EDGES:
                    patience_ -= 1
                    continue
            elif benchmark == 'nasbench201':
                if len(child.arch.nodes) == 0:  # Skip empty graphs in nasbench201 search space
                    patience_ -= 1
                    continue

            skip = False
            if not allow_isomorphism:
                # if disallow isomorphism, we enforce that each time, we mutate n distinct graphs. For now we do not
                # check the isomorphism in all of the previous graphs though
                if benchmark == 'nasbench201':
                    name = child.arch.name
                    if name == arch.arch.name:
                        patience_ -= 1
                        continue
                    prev_pool_names = (prev_edit.name for prev_edit in evaluation_pool)
                    if name in prev_pool_names:
                        patience_ -= 1
                        continue
                elif benchmark == 'nasbench101':
                    if iso.is_isomorphic(child.arch, arch.arch):
                        # todo: for now we simply use the is_isomorphic function in networkx. However, it is possible
                        #  and desirable to simply do Weisfiler-Lehman Isomorphism test, since we are based on
                        #  Weisfeiler-Lehman graph kernel already and the isomorphism test is essentially free.
                        #  (Xingchen, 26 Apr)
                        patience_ -= 1
                        continue
                    # for prev_obs in observed_archs:
                    #     if iso.is_isomorphic(child.arch, prev_obs):
                    #         skip = True
                    #         break
                    for prev_edit in evaluation_pool:
                        if iso.is_isomorphic(child.arch, prev_edit, ):
                            skip = True
                            break
                    if skip:
                        patience_ -= 1
                        continue

            child.error = 0
            evaluation_pool.append(child.arch)
            evaluation_pool_unpruned.append(child.unpruned_arch)
            n_child += 1
    # evaluation_pool = list(chain(*pools))
    # evaluation_pool_unpruned = list(chain(*unpruned_pools))

    # Add some random archs into the evaluation pool, if either 1) patience is reached or 2)
    nrandom_archs = max(pool_size - len(evaluation_pool), 0)
    if nrandom_archs:
        random_evaluation_pool, _, random_evaluation_pool_unpruned = random_sampling(pool_size=nrandom_archs,
                                                                                     benchmark=benchmark,
                                                                                     return_unpruned_archs=True)
        evaluation_pool += random_evaluation_pool
        evaluation_pool_unpruned += random_evaluation_pool_unpruned
    if observed_archs_unpruned is None:
        return evaluation_pool, [None] * len(evaluation_pool)
    return evaluation_pool, evaluation_pool_unpruned


def grad_guided_mutation(observed_archs, observed_errors,
                         kern: GraphKernels,
                         gp: GraphGP,
                         n_best=10,
                         n_mutate=None, pool_size=250, allow_isomorphism=False, patience=50,
                         benchmark='nasbench101', observed_archs_unpruned=None):
    """
    Gradient-guided mutation algorithm (the probability of each mutation is based on the gradient information
    of the GP posterior (gradient reflects the relative importance of the different nodes/edges based on
    first and second order WL iterations)
    Parameters
    ----------
    observed_archs: the list of observed architectures in nx.DiGraph format
    observed_errors: the tensor of the observed validation error (or other relevant performance metric)
    kern: GraphKernels object: the graph kernel used
    gp: GraphGP: the graphGP object used
    n_best: number of top-n best architectures to mutate. If n_best=10, then the children architectures will be
        generated from the top-10 best architectures in the observed architectures.
    n_mutate: number of architectures out of the total pool_size to be generated from mutation. (E.g, if pool_size=200
        and n_mutate=100, then 100 architectures will be generated from mutation, and the other 100 from random
        sampling)
    pool_size: total number of architectures to return in the pool.
    allow_isomorphism: whether allow identical (or perceived-to-be-identical) architectures to be present in the same
        batch. Default False: a simple isomorphism test will be conducted, and identical architectures will be excluded.
        Note: This does not preclude the possibility of an architecture being identical to another in *ALL* evaluated
        architectures
    patience: the patience count which will be decremented every time a mutation fails. If patience reaches 0 and still
        no valid architecture is generated, an empty value will be returned.
    benchmark: the data-set of interest. Options: 'nasbench101' or 'nasbench201' for now
    observed_archs_unpruned: if supplied, the mutation will be done on the unpruned architectures directly. If not None,
        this list must be of the same length as the list of observed_archs.

    Returns
    -------
    Tuple (evaluated_pool, evaluated_pool_unpruned) of the length of the pool_size. If observed_archs_unpruned argument
    is not filled, then the second returned tuple element will be None.
    """
    if n_mutate is None:
        n_mutate = int(0.5 * pool_size)
    assert pool_size >= n_mutate, " pool_size must be larger or equal to n_mutate"
    assert isinstance(kern, WeisfilerLehman), " Grad_guided_mutation only supports Weisfeiler-Lehman kernel!"

    population = collections.deque()
    # Fill the population with the observed archs (a list of labelled graphs) and validation error
    for i, archs in enumerate(observed_archs):
        model = Model()
        model.arch = archs
        model.unpruned_arch = observed_archs_unpruned[i] if observed_archs_unpruned is not None else None
        model.error = observed_errors[i]
        population.append(model)

    best_archs = [arch for arch in sorted(population, key=lambda i: -i.error)][:n_best]
    evaluation_pool, evaluation_pool_unpruned = [], []
    per_arch = n_mutate // n_best

    for arch in best_archs:
        # TODO NEED TO COMPUTE FEATURE_NAMES and FEATURE GRADS of ALL THE BEST/PARENT ARCHS
        # grad, grad_var = GP.dmu_dphi([arch.arch], compute_grad_var=True)
        # feature_names = GP.kernels[0].feature_map(True)
        # grads = grad[0].numpy()
        # feature_grads = grads[np.where(grads != 0)]

        n_child = 0
        patience_ = patience
        while n_child < per_arch and patience_ > 0:
            child = Model()
            # child.arch, child.unpruned_arch = _banana_mutate(arch.unpruned_arch if observed_archs_unpruned is not None
            #                                                  else arch.arch, benchmark=benchmark, )

            # Compute the gradient of the GP posterior, at the best locations.
            feature_names = kern.feature_map(flatten=True)
            grads, grad_var, feature_incidence = gp.dmu_dphi([arch.arch], False, False)
            grads = grads[0][0].numpy()
            feature_grads = grads[np.where(grads != 0)]  # todo: fix this
            feature_incidence = feature_incidence.detach().numpy().flatten()[np.where(grads != 0)]
            features_in_this_arch = np.where(feature_incidence != 0)
            feature_grads_in_this_arch = feature_grads[features_in_this_arch]
            feature_names_in_this_arch = [feature_names[i] for i in list(features_in_this_arch[0])]

            if benchmark == 'nasbench101':

                if observed_archs_unpruned is not None:
                    child.arch, child.unpruned_arch = guided_mutate_arch101(arch.arch, feature_names_in_this_arch, feature_grads_in_this_arch,
                                                                            return_unpruned_arch=True,
                                                                            change_grad_sign=True,
                                                                            unpruned_parent_arch=arch.unpruned_arch)
                else:
                    child.arch, child.unpruned_arch = guided_mutate_arch101(arch.arch, feature_names_in_this_arch, feature_grads_in_this_arch,
                                                                            return_unpruned_arch=False,
                                                                            change_grad_sign=True, )

            elif benchmark == 'nasbench201':
                # for nasbench201 the pruned arch contains unpruned operation list in its name
                child.arch, child.unpruned_arch = guided_mutate_arch201(arch.arch, feature_names_in_this_arch, feature_grads_in_this_arch,
                                                                        return_unpruned_arch=False,
                                                                        change_grad_sign=True, debug_feature_names=[feature_names, feature_incidence])

            if child.arch is None:
                patience_ -= 1
                continue

            # guided_mutate_arch(arch.unpruned_arch)
            # skip 2-node archs
            if benchmark == 'nasbench101':
                if len(child.arch.nodes) <= 3:  # Skip input-output 3-graphs in nasbench101 search space
                    patience_ -= 1
                    continue
                if np.sum(np.array(nx.adjacency_matrix(child.arch).todense())) > MAX_EDGES:
                    patience_ -= 1
                    continue

            elif benchmark == 'nasbench201':
                if len(child.arch.nodes) == 0:  # Skip empty graphs in nasbench201 search space
                    patience_ -= 1
                    continue

            skip = False
            if not allow_isomorphism:
                # if disallow isomorphism, we enforce that each time, we mutate n distinct graphs. For now we do not
                # check the isomorphism in all of the previous graphs though
                if benchmark == 'nasbench201':
                    name = child.arch.name
                    if name == arch.arch.name:
                        patience_ -= 1
                        continue
                    prev_pool_names = (prev_edit.name for prev_edit in evaluation_pool)
                    if name in prev_pool_names:
                        patience_ -= 1
                        continue
                elif benchmark == 'nasbench101':
                    if iso.is_isomorphic(child.arch, arch.arch):
                        patience_ -= 1
                        continue

                    for prev_edit in evaluation_pool:
                        if iso.is_isomorphic(child.arch, prev_edit):
                            skip = True
                            break
                    if skip:
                        patience_ -= 1
                        continue

            child.error = 0
            evaluation_pool.append(child.arch)
            evaluation_pool_unpruned.append(child.unpruned_arch)
            n_child += 1

    # Add some random archs into the evaluation pool, if either 1) patience is reached or 2)
    nrandom_archs = max(pool_size - len(evaluation_pool), 0)
    if nrandom_archs:
        random_evaluation_pool, _, random_evaluation_pool_unpruned = random_sampling(pool_size=nrandom_archs,
                                                                                     benchmark=benchmark,
                                                                                     return_unpruned_archs=True)
        evaluation_pool += random_evaluation_pool
        evaluation_pool_unpruned += random_evaluation_pool_unpruned
    if observed_archs_unpruned is None:
        return evaluation_pool, [None] * len(evaluation_pool)
    return evaluation_pool, evaluation_pool_unpruned


# Random to generate new graphs
def random_sampling(pool_size=100, benchmark='nasbench101',
                    save_config=False,
                    edge_attr=False,
                    return_unpruned_archs=False):
    """
    Return_unpruned_archs: bool: If True, both the list of pruned architectures and unpruned architectures will be
        returned.

    """
    evaluation_pool = []
    pruned_labeling_list = []
    unpruned_evaluation_pool = []  # The unpruned architectures. These might contain invalid architectures
    nasbench201_op_label_list = []
    nasbench_config_list = []
    # attribute_name_list = []
    while len(evaluation_pool) < pool_size:
        if benchmark == 'nasbench101':
            # generate random architecture for nasbench101
            if edge_attr:
                logging.warning("NAS-Bench-101 dataset search space is node-attributed. edge_attr option is not "
                                "applicable.")

            nas101_cs = get_nas101_configuration_space()
            config = nas101_cs.sample_configuration()

            adjacency_matrix = np.zeros([VERTICES, VERTICES], dtype=np.int8)
            idx = np.triu_indices(adjacency_matrix.shape[0], k=1)
            for i in range(VERTICES * (VERTICES - 1) // 2):
                row = idx[0][i]
                col = idx[1][i]
                adjacency_matrix[row, col] = config["edge_%d" % i]

            labeling = [config["op_node_%d" % i] for i in range(5)]
            labeling = ['input'] + list(labeling) + ['output']

            try:
                pruned_adjacency_matrix, pruned_labeling = prune(adjacency_matrix, labeling)
            except:
                continue

            # skip only duplicating 2-node architecture
            if len(pruned_labeling) == 2 and pruned_labeling in pruned_labeling_list:
                continue

            # skip invalid architectures whose number of edges exceed the max limit of 9
            if np.sum(pruned_adjacency_matrix) > MAX_EDGES or np.sum(pruned_adjacency_matrix) == 0:
                continue

            rand_arch = nx.from_numpy_array(pruned_adjacency_matrix, create_using=nx.DiGraph)
            rand_arch.graph_type = 'node_attr'

            for i, n in enumerate(pruned_labeling):
                rand_arch.nodes[i]['op_name'] = n
            if return_unpruned_archs:
                unpruned_rand_arch = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
                unpruned_rand_arch.graph_type = 'node_attr'
                for i, n in enumerate(labeling):
                    unpruned_rand_arch.nodes[i]['op_name'] = n

            pruned_labeling_list.append(pruned_labeling)
            nasbench_config_list.append(config)

        elif benchmark == 'nasbench201':
            # generate random architecture for nasbench201

            nas201_cs = get_nas201_configuration_space()
            config = nas201_cs.sample_configuration()
            op_labeling = [config["edge_%d" % i] for i in range(len(config.keys()))]
            # skip only duplicating architecture
            if op_labeling in nasbench201_op_label_list:
                continue

            nasbench201_op_label_list.append(op_labeling)
            rand_arch = create_nasbench201_graph(op_labeling, edge_attr)
            nasbench_config_list.append(config)

            # IN Nasbench201, it is possible that invalid graphs consisting entirely from None and skip-line are
            # generated; remove these invalid architectures.

            # Also remove if the number of edges is zero. This is is possible, one example in NAS-Bench-201:
            # '|none~0|+|avg_pool_3x3~0|nor_conv_3x3~1|+|none~0|avg_pool_3x3~1|none~2|'
            if len(rand_arch) == 0 or rand_arch.number_of_edges() == 0:
                continue

            unpruned_rand_arch = rand_arch
        evaluation_pool.append(rand_arch)
        if return_unpruned_archs:
            unpruned_evaluation_pool.append(unpruned_rand_arch)
        # attribute_name_list.append(nx.get_node_attributes(rand_arch, 'op_name'))
    if save_config:
        if return_unpruned_archs:
            return evaluation_pool, nasbench_config_list, unpruned_evaluation_pool
        else:
            return evaluation_pool, nasbench_config_list, None
    else:
        if return_unpruned_archs:
            return evaluation_pool, None, unpruned_evaluation_pool
        else:
            return evaluation_pool, None, None


# Random graph model to generate new graphs for nasbench101 only
def build_graph(graph_model, graph_params, seed):
    Nodes, P, M, K = graph_params
    if graph_model == 'ER':
        return nx.random_graphs.erdos_renyi_graph(Nodes, P, seed)
    elif graph_model == 'BA':
        return nx.random_graphs.barabasi_albert_graph(Nodes, M, seed)
    elif graph_model == 'WS':
        return nx.random_graphs.connected_watts_strogatz_graph(Nodes, K, P, tries=200, seed=seed)
        # return nx.random_graphs.watts_strogatz_graph(Nodes, K, P, seed=seed)


def random_graph_generation(graph_model='ER', pool_size=100):
    # todo: this does not work for Nasbench201
    evaluation_pool = []
    ops_choices = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
    n_edges_list = []
    seed = 0
    while len(evaluation_pool) < pool_size:
        random.seed(seed)
        op_labeling = [random.choice(ops_choices) for i in range(5)]
        labeling = ['input'] + op_labeling + ['output']
        if graph_model == 'ER':
            graph_params = [7, 0.43, None, None]

        G = build_graph(graph_model, graph_params, seed=seed)
        full_adjacency_matrix = np.array(nx.adjacency_matrix(G).todense())
        adjacency_matrix = np.triu(full_adjacency_matrix, k=0)

        try:
            pruned_adjacency_matrix, pruned_labeling = prune(adjacency_matrix, labeling)
            n_edges = np.sum(pruned_adjacency_matrix)
            n_edges_list.append(n_edges)
            if n_edges > MAX_EDGES:
                assert False

            pruned_G = nx.from_numpy_array(pruned_adjacency_matrix, create_using=nx.DiGraph)
            for i, n in enumerate(pruned_labeling):
                pruned_G.nodes[i]['op_name'] = n

            evaluation_pool.append(pruned_G)
            seed += 1
        except:
            seed += 1
            print('invalid arch')
            continue

    return evaluation_pool


if __name__ == '__main__':
    import pickle


    class acquisition_func:
        def __init__(self):
            self.v = None

        def eval(self, G, asscalar=True):
            y = len(G.nodes) + len(G.edges)
            return y


    #
    # output_path = '../data/'
    # with open(os.path.join(output_path, 'valid_arch_samples_pruned'), 'rb') as outfile:
    #     res = pickle.load(outfile)
    #
    # observed_archs_list = []
    # observed_err_list = []
    # n_init = 30
    # k = 0
    # while k < n_init:
    #     model = res['model_graph_specs'][k]
    #     A = model['adjacency']
    #     nl = model['node_labels']
    #     val_err = res['validation_err'][k]
    #     G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    #     for i, n in enumerate(nl):
    #         G.node[i]['op_name'] = n
    #
    #     observed_archs_list.append(G)
    #     observed_err_list.append(val_err)
    #
    #     k += 1

    # Generate new archs using random sampling
    observed_archs_list, _, _ = random_sampling(pool_size=500, benchmark='nasbench201', edge_attr=True)
    print('hold')
    # Or using regularised evolution
    # af = acquisition_func()
    # best_n_arhcs = regularized_evolution(af, observed_archs_list, benchmark='nasbench201', pool_size=100, cycles=5,
    #                                      n_mutation=10, batch_size=1)
    # Or using random graph model
    # pool3 = random_graph_generation(graph_model='ER', pool_size=100)

    # x_testset = generate_new_test_locations()
    # acq_values = acqquisition_func(x_testset)
    # idx = argmax(acq_values)
    # x_next = x_testset[idx]

    #  y  = f(x1,x2) -->  now
    #  y1 = f1(x1,x2,x)   ER
    #  y2 = f2(x1,x2,k)   BA
    #  y3 = f3(x1,x2,k,x) WS

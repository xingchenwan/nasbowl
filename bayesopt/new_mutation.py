#  | 5 March 2020
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
import pickle
from bayesopt.generate_test_graphs import prune, random_sampling, Model, MAX_EDGES, create_nasbench201_graph


def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def guided_mutate_arch(parent_arch, feature_names, features_grad, return_unpruned_arch=True, change_grad_sign=True, like_banans=True):

    # combine feature_names, features_grad into one dic
    # the gradient w.r.t all features seen so far both test arch and observed arch
    all_features = {}
    for feature in feature_names.items():
        idx, feature_name = feature
        if change_grad_sign:
            all_features[feature_name] = - features_grad[idx]
        else:
            all_features[feature_name] = features_grad[idx]


    # find the grad for all the op choices
    op_list = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
    op_grad_list = [all_features[op] for op in op_list]

    # define all possible edges in the parent graph and encode them as "in_node_op_name~out_node_op_name"
    edges_pool = [(ni, no) for ni in range(parent_arch.number_of_nodes() - 1) for no in
                  range(ni + 1, parent_arch.number_of_nodes())]
    edge_encode_dic = {}
    for j1, e in enumerate(edges_pool):
        in_node, out_node = e
        edge_encode = parent_arch.node[out_node]['op_name']+ '~' + parent_arch.node[out_node]['op_name']
        if parent_arch.node[out_node]['op_name'] == 'input' or parent_arch.node[out_node]['op_name']:
            print(f'invalid edge: {edge_encode}')
        edge_encode_dic[e] = edge_encode

    # encode each node with its op_name
    node_encode_dic = {}
    for j2, node in enumerate(parent_arch.nodes):
        node_encode = parent_arch.node[node]['op_name']
        node_encode_dic[node] = node_encode

    # the collate the overall probs of being chosen for all edges and nodes
    edge_prob_dic = {}
    for edge_item in edge_encode_dic.items():
        edge_key = edge_item[0]
        edge_encode_str = edge_item[1]
        edge_grads = [all_features[key] for key in list(all_features.keys()) if edge_encode_str in key]
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
    normalised_node_edge_probs = sigmoid(node_edge_prob_array) / np.sum(sigmoid(node_edge_prob_array))

    # start mutate
    child_arch = deepcopy(parent_arch)
    if like_banans:
        # loop through all node and possible edges in the parent arch to decide whether to mutate any of them
        for i, item in enumerate(node_edge_list):
            prob_of_mutation = normalised_node_edge_probs[i]

            if random.random() < prob_of_mutation:
                if type(item) == tuple:
                    print(f'mutate edge {item}')
                    # an edge is chosen
                    if item in child_arch.edges:
                        child_arch.remove_edges_from([item])
                    else:
                        child_arch.add_edges_from([item])
                else:
                    print(f'mutate node {item}')
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
                    normalised_op_prob = sigmoid(op_grad_array) / np.sum(sigmoid(op_grad_array))
                    choosen_op = random.choices(population=op_list_copy, weights=normalised_op_prob, k=1)
                    child_arch.node[item]['op_name'] = choosen_op[0]

    else:
        # pick one of all node and possible edges in the parent arch to mutate
        choosen_item = random.choices(population=node_edge_list, weights=normalised_node_edge_probs,
                                      k=1)  # choosen_item is a list of k item
        if type(choosen_item[0]) == tuple:
            print(f'mutate edge {choosen_item[0]}')
            # an edge is chosen
            if choosen_item[0] in child_arch.edges:
                child_arch.remove_edges_from(choosen_item)
            else:
                child_arch.add_edges_from(choosen_item)
        else:
            print(f'mutate node {choosen_item[0]}')
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
            normalised_op_prob = sigmoid(op_grad_array)/ np.sum(sigmoid(op_grad_array))
            choosen_op = random.choices(population=op_list_copy, weights=normalised_op_prob, k=1)
            child_arch.node[choosen_item[0]]['op_name'] = choosen_op[0]

    if not return_unpruned_arch:
        # pruning
        node_labeling = list(nx.get_node_attributes(child_arch, 'op_name').values())
        adjacency_matrix = np.array(nx.adjacency_matrix(child_arch).todense())
        pruned_adjacency_matrix, pruned_node_labeling = prune(adjacency_matrix, node_labeling)
        child_arch = nx.from_numpy_array(pruned_adjacency_matrix, create_using=nx.DiGraph)
        for i, n in enumerate(pruned_node_labeling):
            child_arch.nodes[i]['op_name'] = n

    return child_arch, None

def guided_mutate_arch201(parent_arch, feature_names, features_grad, return_unpruned_arch=True, change_grad_sign=True, like_banans=False):

    # combine feature_names, features_grad into one dic
    # the gradient w.r.t all features seen so far both test arch and observed arch
    all_features = {}
    for feature in feature_names.items():
        idx, feature_name = feature
        if change_grad_sign:
            all_features[feature_name] = - features_grad[idx]
        else:
            all_features[feature_name] = features_grad[idx]

    # find the grad for all the op choices
    # OPS_201 = ['nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3', 'skip_connect', 'none']
    all_ops_grad_dic = {'nor_conv_3x3':[], 'nor_conv_1x1':[], 'avg_pool_3x3':[], 'skip_connect':[], 'none':[]}
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
                related_feature_names = [node_labelling[n_i]+ '~' + node_labelling[n_o] for n_o in output_nodes]

            # collect the grads for the related feature
            related_feature_grads = []
            for feature_name in related_feature_names:
                feature_grad_list = [all_features[key] for key in list(all_features.keys()) if feature_name in key]
                related_feature_grads += feature_grad_list
            feature_grad_sum = np.sum(related_feature_grads)

        else:
            feature_grad_sum = all_features[n]
        node_prob_dic[i] = feature_grad_sum
        all_ops_grad_dic[n].append(feature_grad_sum)

    # define the prob for all potential op choices:
    all_ops_prob_dic={}
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
    normalised_node_probs = sigmoid(unpruned_node_prob_array) / np.sum(sigmoid(unpruned_node_prob_array))

    # start mutate
    child_arch = deepcopy(G)
    if like_banans:
        # loop through all node and possible edges in the parent arch to decide whether to mutate any of them
        for i, item in enumerate(unpruned_node_list):
            prob_of_mutation = normalised_node_probs[i]

            if random.random() < prob_of_mutation:
                print(f'mutate node {item}')
                # a node is chosen
                all_ops_prob_dic_copy = all_ops_prob_dic.copy()
                # remove the current node operation from all op choices
                all_ops_prob_dic_copy.pop(G.nodes[item]['op_name'])

                # choose among the remaining op choices
                op_list_copy = list(all_ops_prob_dic_copy.keys())
                op_grad_array = np.array(list(all_ops_prob_dic_copy.values()))
                normalised_op_prob = sigmoid(op_grad_array) / np.sum(sigmoid(op_grad_array))
                choosen_op = random.choices(population=op_list_copy, weights=normalised_op_prob, k=1)
                child_arch.nodes[item]['op_name'] = choosen_op[0]
                node_labelling[item] = choosen_op[0]


    else:
        # pick one of all node and possible edges in the parent arch to mutate
        choosen_item = random.choices(population=unpruned_node_list, weights=normalised_node_probs,
                                      k=1)  # choosen_item is a list of k item

        print(f'mutate node {choosen_item[0]}')
        # a node is chosen
        all_ops_prob_dic_copy = all_ops_prob_dic.copy()
        # remove the current node operation from all op choices
        all_ops_prob_dic_copy.pop(G.nodes[choosen_item[0]]['op_name'])

        # choose among the remaining op choices
        op_list_copy = list(all_ops_prob_dic_copy.keys())
        op_grad_array = np.array(list(all_ops_prob_dic_copy.values()))
        normalised_op_prob = sigmoid(op_grad_array) / np.sum(sigmoid(op_grad_array))
        choosen_op = random.choices(population=op_list_copy, weights=normalised_op_prob, k=1)
        child_arch.node[choosen_item[0]]['op_name'] = choosen_op[0]
        node_labelling[choosen_item[0]] = choosen_op[0]

    unpruned_child_arch = child_arch
    # create the arch string for querying nasbench dataset
    arch_query_string = f'|{node_labelling[1]}~0|+' \
                        f'|{node_labelling[2]}~0|{node_labelling[3]}~1|+' \
                        f'|{node_labelling[4]}~0|{node_labelling[5]}~1|{node_labelling[6]}~2|'

    unpruned_child_arch.name = arch_query_string

    pruned_child_arch = create_nasbench201_graph(node_labelling[1:-1], edge_attr=False)

    return pruned_child_arch, unpruned_child_arch


def grad_guided_mutation(observed_archs, observed_errors, feature_names_list, feature_grads_list, n_best=10,
             n_mutate=None, pool_size=250, allow_isomorphism=False, patience=50,
             benchmark='nasbench101', observed_archs_unpruned=None):
    if n_mutate is None:
        n_mutate = int(0.5 * pool_size)
    assert pool_size >= n_mutate, " pool_size must be larger or equal to n_mutate"

    population = collections.deque()
    # Fill the population with the observed archs (a list of labelled graphs) and validation error
    for i, archs in enumerate(observed_archs):
        model = Model()
        model.arch = archs
        model.unpruned_arch = observed_archs_unpruned[i]
        model.error = observed_errors[i]
        population.append(model)

    best_archs = [arch for arch in sorted(population, key=lambda i: -i.error)][:n_best]
    evaluation_pool, evaluation_pool_unpruned = [], []
    per_arch = n_mutate // n_best
    # TODO NEED TO COMPUTE FEATURE_NAMES and FEATURE GRADS of ALL THE BEST/PARENT ARCHS
    for arch in best_archs:
        n_child = 0
        patience_ = patience
        while n_child < per_arch and patience_ > 0:
            child = Model()
            # child.arch, child.unpruned_arch = _banana_mutate(arch.unpruned_arch if observed_archs_unpruned is not None
            #                                                  else arch.arch, benchmark=benchmark, )
            feature_names = feature_names_list[observed_archs.index(arch.arch)]
            feature_grads = feature_grads_list[observed_archs.index(arch.arch)]

            if benchmark == 'nasbench101':
                child.arch, child.unpruned_arch = guided_mutate_arch(arch.arch, feature_names, feature_grads,
                                                                     return_unpruned_arch=False, change_grad_sign=True, like_banans=True)
            elif benchmark == 'nasbench201':
                child.arch, child.unpruned_arch = guided_mutate_arch201(arch.arch, feature_names, feature_grads,
                                                                     return_unpruned_arch=False, change_grad_sign=True,
                                                                     like_banans=True)


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
        random_evaluation_pool, _, random_evaluation_pool_unpruned = random_sampling(pool_size=nrandom_archs, benchmark=benchmark, return_unpruned_archs=True)
        evaluation_pool += random_evaluation_pool
        evaluation_pool_unpruned += random_evaluation_pool_unpruned
    if observed_archs_unpruned is None:
        return evaluation_pool, None
    return evaluation_pool, evaluation_pool_unpruned

if __name__ == '__main__':
    # feature name = model.kernels[0].feature_map(True)
    # features_grad = grads[np.where(grads!=0)] --> a flatten array

    with open('../data/nasbench201_cifar10valid_observed_archs', 'rb') as outfile:
    # with open('../data/observed_archs_res', 'rb') as outfile:
        observed_arch_res = pickle.load(outfile)

    observed_archs = observed_arch_res['archs']
    observed_errors = observed_arch_res['error']
    feature_names = observed_arch_res['feature_names']
    feature_grads = observed_arch_res['feature_grads']

    child_arch ,_ = grad_guided_mutation(observed_archs, observed_errors, feature_names, feature_grads, n_best=2,
             n_mutate=10, pool_size=20, allow_isomorphism=False, patience=50, benchmark='nasbench101', observed_archs_unpruned=[None, None, None])
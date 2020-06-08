import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pickle
import copy


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
    visited_from_input = set([0])
    frontier = [0]
    while frontier:
      top = frontier.pop()
      for v in range(top + 1, num_vertices):
        if original_matrix[top, v] and v not in visited_from_input:
          visited_from_input.add(v)
          frontier.append(v)

    # DFS backward from output
    visited_from_output = set([num_vertices - 1])
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
    try:
        for index in sorted(extraneous, reverse=True):
          del new_ops[index]
    except:
        print('hold')

    return new_matrix, new_ops

output_path = './data/'
with open(os.path.join(output_path, 'valid_arch_samples'),'rb') as outfile:
    res = pickle.load(outfile)

# prune architecture samples
graph_A_list = []
graph_node_label_list = []
graph_val_err_list = res['validation_err']
graph_test_err_list = res['test_err']
pruned_model_graph_specs_list = []
for i in range(len(graph_val_err_list)):
    A = res['model_graph_specs'][i]['adjacency']
    nl=res['model_graph_specs'][i]['node_labels']
    graph_A_list.append(A)
    graph_node_label_list.append(nl)

    pruned_adjacency_matrix, pruned_opts = prune(A, nl)
    pruned_model_specs = {'adjacency': pruned_adjacency_matrix, 'node_labels': pruned_opts}
    pruned_model_graph_specs_list.append(pruned_model_specs)

# resave pruned results
res_pruned = {}
res_pruned['validation_err'] = graph_val_err_list
res_pruned['test_err'] = graph_test_err_list
res_pruned['runtime'] = res['runtime']
res_pruned['model_graph_specs'] = pruned_model_graph_specs_list
with open(os.path.join(output_path, 'valid_arch_samples_pruned'),'wb') as outfile:
    pickle.dump(res_pruned, outfile)

# visualise an architecture sample
sample_idx = 2
# the original graph architecture
original_adjacency_matrix = graph_A_list[sample_idx]
original_opts = graph_node_label_list[sample_idx]
G=nx.from_numpy_array(original_adjacency_matrix, create_using=nx.DiGraph)
# for i, n in enumerate(graph_node_label_list[idx]):
#     G.node[i]['op_name'] = n
pos = nx.layout.circular_layout(G)
node_colors = ['red']+['lightblue']*(7-2) + ['red']
nx.draw(G, pos, node_size=1000, width=2, alpha=1.0, node_shape='o', node_color=node_colors)
nx.draw_networkx_labels(G, pos, labels={i: label for i, label in enumerate(original_opts) if i in pos}, font_size=8)
plt.title('original architecture graph')
plt.tight_layout()

# the pruned graph architecture
pruned_adjacency_matrix, pruned_opts = prune(original_adjacency_matrix, original_opts)
G_pruned=nx.from_numpy_array(pruned_adjacency_matrix, create_using=nx.DiGraph)
pos_pruned = nx.layout.circular_layout(G_pruned)
node_colors = ['red']+['lightblue']*(pruned_adjacency_matrix.shape[0]-2) + ['red']
plt.figure()
nx.draw(G_pruned, pos_pruned, node_size=1000, width=2, alpha=1.0, node_shape='o', node_color=node_colors)
nx.draw_networkx_labels(G_pruned, pos_pruned, labels={i: label for i, label in enumerate(pruned_opts) if i in pos}, font_size=8)
plt.title('pruned architecture graph')
print(f'arch_{sample_idx}: val_acc={graph_val_err_list[sample_idx]}')
plt.tight_layout()
plt.show()


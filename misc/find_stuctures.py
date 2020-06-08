import networkx as nx


def find_node(gr, att, val):
    """Applicable for the first-layer WL feature (i.e. the nodes themselves)"""
    return len([node for node in gr.nodes(data=True) if node[1][att] == val])


def find_2_structure(gr, att, encoding):
    """Applicable for the second-layer WL features (i.e. the nodes + their 1-neighbours)"""
    if "~" in encoding:
        # Temporary fix
        encoding = encoding.split("~")
        encoding = [(e, ) for e in encoding]
    root_node = encoding[0][0]
    leaf_node = [encoding[e][0] for e in range(1, len(encoding))]
    counter = {x: leaf_node.count(x) for x in set(leaf_node)}
    counts = []
    for node in gr.nodes(data=True):
        if node[1][att] == root_node:
            count = {x: 0 for x in set(leaf_node)}
            for neighbor in nx.neighbors(gr, node[0]):
                if gr.nodes[neighbor][att] in leaf_node:
                    count[gr.nodes[neighbor][att]] += 1
            counts.append(count)
    for c in counts:
        if c == counter: return True
    return False
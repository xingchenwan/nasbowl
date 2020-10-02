import logging

import networkx as nx
import numpy as np
import torch


def add_color(arch, color_map):
    """Add a node attribute called color, which color the nodes based on the op type"""
    for i, (node, data) in enumerate(arch.nodes(data=True)):
        try:
            arch.nodes[i]['color'] = color_map[data['op_name']]
        except KeyError:
            logging.warning('node operation ' + color_map[data['op_name']] + ' is not found in the color_map! Skipping')
    return arch


def encoding_to_nx(encoding: str, color_map: dict = None):
    """Convert a feature encoding (example 'input~maxpool3x3~conv3x3') to a networkx graph
    for WL features up to h=1.
    color_map: dict. When defined, supplement the encoding motifs with a color information that can be
    useful for later plotting.
    WARNING: this def is not tested for higher-order WL features, and thus the code might break."""
    g_nx = nx.DiGraph()
    nodes = encoding.split("~")
    for i, n in enumerate(nodes):
        if color_map is None:
            g_nx.add_node(
                i, op_name=n
            )
        else:
            try:
                g_nx.add_node(
                    i, op_name=n,
                    color=color_map[n]
                )
            except KeyError:
                logging.warning('node operation ' + n + ' is not found in the color_map! Skipping')
                g_nx.add_node(
                    i, op_name=n
                )
        if i > 0:
            g_nx.add_edge(0, i)
    return g_nx


def _preprocess(X, y=None):
    from .generate_test_graphs import prune
    tmp = []
    valid_indices = []
    for idx, c in enumerate(X):
        node_labeling = list(nx.get_node_attributes(c, 'op_name').values())
        try:
            res = prune(nx.to_numpy_array(c), node_labeling)
            if res is None:
                continue
            c_new, label_new = res
            c_nx = nx.from_numpy_array(c_new, create_using=nx.DiGraph)
            for i, n in enumerate(label_new):
                c_nx.nodes[i]['op_name'] = n
        except KeyError:
            print('Pruning error!')
            c_nx = c
        tmp.append(c_nx)
        valid_indices.append(idx)
    if y is not None: y = y[valid_indices]
    if y is None:
        return tmp
    return tmp, y


def normalize_y(y: torch.Tensor):
    y_mean = torch.mean(y) if isinstance(y, torch.Tensor) else np.mean(y)
    y_std = torch.std(y) if isinstance(y, torch.Tensor) else np.std(y)
    y = (y - y_mean) / y_std
    return y, y_mean, y_std


def unnormalize_y(y, y_mean, y_std, scale_std=False):
    """Similar to the undoing of the pre-processing step above, but on the output predictions"""
    if not scale_std:
        y = y * y_std + y_mean
    else:
        y *= y_std
    return y


def standardize_x(x: torch.Tensor, x_min: torch.Tensor = None, x_max: torch.Tensor = None):
    """Standardize the vectorial input into a d-dimensional hypercube [0, 1]^d, where d is the number of features.
    if x_min ond x_max are supplied, x2 will be standardised using these instead. This is used when standardising the
    validation/test inputs.
    """
    if (x_min is not None and x_max is None) or (x_min is None and x_max is not None):
        raise ValueError("Either *both* or *neither* of x_min, x_max need to be supplied!")
    if x_min is None:
        x_min = torch.min(x, 0)[0]
        x_max = torch.max(x, 0)[0]
    x = (x - x_min) / (x_max - x_min)
    return x, x_min, x_max


def compute_log_marginal_likelihood(K_i, logDetK, y, normalize=True,
                                    log_prior_dist=None):
    """Compute the zero mean Gaussian process log marginal likelihood given the inverse of Gram matrix K(x2,x2), its
    log determinant, and the training label vector y.
    Option:

    normalize: normalize the log marginal likelihood by the length of the label vector, as per the gpytorch
    routine.

    prior: A pytorch distribution object. If specified, the hyperparameter prior will be taken into consideration and
    we use Type-II MAP instead of Type-II MLE (compute log_posterior instead of log_evidence)
    """
    lml = -0.5 * y.t() @ K_i @ y + 0.5 * logDetK - y.shape[0] / 2. * torch.log(2 * torch.tensor(np.pi, ))
    if log_prior_dist is not None:
        lml -= log_prior_dist
    return lml / y.shape[0] if normalize else lml


def compute_pd_inverse(K, jitter=1e-5):
    """Compute the inverse of a postive-(semi)definite matrix K using Cholesky inversion."""
    n = K.shape[0]
    assert isinstance(jitter, float) or jitter.ndim == 0, 'only homoscedastic noise variance is allowed here!'
    is_successful = False
    fail_count = 0
    max_fail = 3
    while fail_count < max_fail and not is_successful:
        try:
            jitter_diag = jitter * torch.eye(n, device=K.device) * 10 ** fail_count
            K_ = K + jitter_diag
            Kc = torch.cholesky(K_)
            is_successful = True
        except RuntimeError:
            fail_count += 1
    if not is_successful:
        print(K)
        raise RuntimeError("Gram matrix not positive definite despite of jitter")
    logDetK = -2 * torch.sum(torch.log(torch.diag(Kc)))
    K_i = torch.cholesky_inverse(Kc)
    return K_i.float(), logDetK.float()


import matplotlib.pyplot as plt
import matplotlib


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x2:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x2:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def gradient(outputs, inputs, grad_outputs=None, retain_graph=True, create_graph=True):
    """
    Compute the gradient of `outputs` with respect to `inputs`
    gradient(x.sum(), x)
    gradient((x * y).sum(), [x, y])
    """
    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)
    grads = torch.autograd.grad(outputs, inputs, grad_outputs,
                                allow_unused=True,
                                retain_graph=retain_graph,
                                create_graph=create_graph)
    grads = [x if x is not None else torch.zeros_like(y) for x, y in zip(grads, inputs)]
    return torch.cat([x.contiguous().view(-1) for x in grads])


def jacobian(outputs, inputs, create_graph=False):
    """
    Compute the Jacobian of `outputs` with respect to `inputs`
    jacobian(x, x)
    jacobian(x * y, [x, y])
    jacobian([x * y, x.sqrt()], [x, y])
    """
    if torch.is_tensor(outputs):
        outputs = [outputs]
    else:
        outputs = list(outputs)

    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)

    jac = []
    for output in outputs:
        output_flat = output.view(-1)
        output_grad = torch.zeros_like(output_flat)
        for i in range(len(output_flat)):
            output_grad[i] = 1
            jac += [gradient(output_flat, inputs, output_grad, True, create_graph)]
            output_grad[i] = 0
    return torch.stack(jac)


def hessian(output, inputs, out=None, allow_unused=False, create_graph=False):
    """
        Compute the Hessian of `output` with respect to `inputs`
        hessian((x * y).sum(), [x, y])
    """
    assert output.ndimension() == 0

    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)

    n = sum(p.numel() for p in inputs)
    if out is None:
        out = output.new_zeros(n, n)

    ai = 0
    for i, inp in enumerate(inputs):
        [grad] = torch.autograd.grad(output, inp, create_graph=True, allow_unused=allow_unused)
        grad = torch.zeros_like(inp) if grad is None else grad
        grad = grad.contiguous().view(-1)

        for j in range(inp.numel()):
            if grad[j].requires_grad:
                row = gradient(grad[j], inputs[i:], retain_graph=True, create_graph=create_graph)[j:]
            else:
                row = grad[j].new_zeros(sum(x.numel() for x in inputs[i:]) - j)

            out[ai, ai:].add_(row.type_as(out))  # ai's row
            if ai + 1 < n:
                out[ai + 1:, ai].add_(row[1:].type_as(out))  # ai's column
            del row
            ai += 1
        del grad

    return out

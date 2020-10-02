# This file provides the conversion between of our API the DARTS interface
import logging

import torch
from torch.multiprocessing import Pool

from benchmarks.objectives import ObjectiveFunction
from darts.arch_trainer import DARTSTrainer
from darts.utils import *

INPUT_1 = 'c_k-2'
INPUT_2 = 'c_k-1'


def get_ops(search_space):
    if search_space == 'darts':
        return ['max_pool_3x3',
                'avg_pool_3x3',
                'skip_connect',
                'sep_conv_3x3',
                'sep_conv_5x5',
                'dil_conv_3x3',
                'dil_conv_5x5',
                ], 4
    elif search_space == 'nasnet':
        return [
                   'max_pool_3x3',
                   'max_pool_5x5',
                   'sep_conv_3x3',
                   'sep_conv_5x5',
                   'sep_conv_7x7',
                   'skip_connect',
               ], 7


class DARTSObjective(ObjectiveFunction):
    def __init__(self,
                 data_path: str,
                 save_path: str,
                 dataset: str = 'cifar10',
                 cutout=False,
                 log_scale=True,
                 negative=True,
                 query_policy='best',
                 seed=None,
                 n_gpu='all', epochs=50,
                 dummy_eval=True,
                 search_space='darts',
                 auxiliary=True):
        super(DARTSObjective, self).__init__()
        assert query_policy in ['best', 'last', 'last5']
        self.query_policy = query_policy
        self.cutout = cutout
        self.log_scale = log_scale
        self.negative = negative
        self.seed = seed
        self.dummy_eval = dummy_eval
        self.dataset = dataset
        self.data_path = data_path
        self.save_path = save_path
        self.n_gpu = n_gpu if isinstance(n_gpu, int) else torch.cuda.device_count()
        self.epochs = epochs
        self.search_space = search_space
        self.auxiliary = auxiliary

    def eval(self, X, *args):
        if self.dummy_eval:
            return self.dummy_eval_(X, *args)
        return self.eval_(X, *args)

    def eval_(self, X, *args):
        """
        Evaluate a number of DARTS architecture in parallel. X should be a list of Genotypes defined by DARTS API.
        """
        from math import ceil
        n_parallel = min(len(X), self.n_gpu)
        res = []
        diag_stats = []

        if n_parallel == 0:
            raise ValueError("No GPUs available!")
        elif n_parallel == 1:
            for i, genotype in enumerate(X):
                t = DARTSTrainer(self.data_path, self.save_path, genotype, self.dataset, cutout=self.cutout,
                                 auxiliary_tower=self.auxiliary,
                                 epochs=self.epochs, eval_policy=self.query_policy)
                print('Start training: ', i + 1, "/ ", len(X))
                try:
                    t.train()  # bottleneck
                    result = t.retrieve()
                    res.append(1. - result[0] / 100.)  # Turn into error
                    diag_stats.append(result[1])
                except Exception as e:
                    logging.error(
                        "An error occured in the current architecture. Assigning nan to the arch. The error is:")
                    logging.error(e)
                    res.append(np.nan)
                    diag_stats.append(None)

        else:
            gpu_ids = range(n_parallel)
            num_reps = ceil(len(X) / float(n_parallel))
            for i in range(num_reps):
                x = X[i * n_parallel: min((i + 1) * n_parallel,
                                          len(X))]  # select the number of parallel archs to evaluate
                selected_gpus = gpu_ids[:len(x)]
                other_arg = [self.data_path, self.save_path, self.dataset, self.cutout, self.epochs, self.query_policy]
                args = list(map(list, zip(x, selected_gpus, ), ))
                args = [a + other_arg for a in args]
                pool = Pool(processes=len(x))
                current_res = pool.starmap(parallel_eval, args)
                pool.close()
                pool.join()
                res.extend([i for i in current_res if i >= 0])  # Filter out the negative results due to errors
        res = np.array(res).flatten()
        if self.log_scale:
            res = np.log(res)
        if self.negative:
            res = -res
        return res, diag_stats

    def dummy_eval_(self, X: list, *args):
        # Evaluate a dummy variable for fast debugging
        # The dummy variable is a linear function of the number of 'dil_conv_5x5' units -- the objective
        # is to maximise the number of such operations.
        res = []
        for i, x in enumerate(X):
            count = 1.
            for i in x.nodes:
                if x.nodes[i]['op_name'] == 'sep_conv_3x3':
                    count -= 0.1
                # elif x.nodes[i]['op_name'] == 'sep_conv_3x3':
                #     count -= 0.1
                # elif x.nodes[i]['op_name'] == 'dil_conv_5x5':
                #     count -= 0.2
            res.append(max(count, 0.01))
        res = np.array(res).flatten()
        if self.log_scale:
            res = np.log(res)
        if self.negative:
            res = -res
        return res, None


def parallel_eval(*args):
    """The function to be used for parallelism"""
    genotype, gpuid, data_path, save_path, dataset, cutout, epochs, policy = args
    print(args)
    t = DARTSTrainer(data_path, save_path, genotype, dataset,
                     cutout=cutout, epochs=epochs, gpu_id=gpuid, eval_policy=policy)
    try:
        t.train()
    except:
        logging.error("Error occurred in training on of the archs. The child process is terminated")
        return -1
    return 1. - t.retrieve() / 100.


def random_sample_darts(n, search_space, same_arch=True, n_tower=4):
    """
    n: number of random samples to yield
    same_arch (bool): whether to use the same architecture for the normal cell and the reduction cell
    """
    if not same_arch:
        raise NotImplementedError

    """Generate a list of 2 tuples, consisting of the random DARTS Genotype and DiGraph"""
    OPS, n_tower = get_ops(search_space)

    def _sample():
        normal = []
        reduction = []
        for i in range(n_tower):
            ops = np.random.choice(range(len(OPS)), n_tower)

            # input nodes for conv
            nodes_in_normal = np.random.choice(range(i + 2), 2, replace=False)
            # input nodes for reduce
            if same_arch:
                nodes_in_reduce = nodes_in_normal
                normal.extend([(OPS[ops[0]], nodes_in_normal[0]), (OPS[ops[1]], nodes_in_normal[1])])
                reduction.extend([(OPS[ops[0]], nodes_in_reduce[0],), (OPS[ops[1]], nodes_in_reduce[1])])
            else:
                nodes_in_reduce = np.random.choice(range(i + 2), 2, replace=False)
                normal.extend([(OPS[ops[0]], nodes_in_normal[0]), (OPS[ops[1]], nodes_in_normal[1])])
                reduction.extend([(OPS[ops[2]], nodes_in_reduce[2],), (OPS[ops[3]], nodes_in_reduce[3])])

        darts_genotype = Genotype(normal=normal, normal_concat=range(2, 2 + n_tower),
                                  reduce=reduction, reduce_concat=range(2, 2 + n_tower))
        darts_digraph = darts2graph(darts_genotype)
        return darts_digraph[0], darts_genotype

    res = []
    for i in range(n):
        r = _sample()
        if is_valid_darts(r[1]):
            res.append(r)
    return res


def _mutate(arch: Genotype, search_space, edits, mutate_main_only=True) -> (nx.DiGraph, Genotype):
    """BANANAS style mutation on a DARTS-style Genotype"""
    OPS, n_towers = get_ops(search_space)

    def _mutate_cell(cell, edits):
        mutable_cell = [list(i) for i in cell]  # Convert tuple to list to allow edits
        for _ in range(edits):
            num = np.random.choice(2)
            op_to_mutate = np.random.choice(len(cell))
            if num == 1:  # Mutate the ops

                op_chosen = np.random.choice(OPS)
                mutable_cell[op_to_mutate][0] = op_chosen
            else:  # Mutate the wiring
                inputs = len(mutable_cell) // 2 + 2
                while True:
                    choice = np.random.choice(inputs)
                    if op_to_mutate % 2 == 0 and mutable_cell[op_to_mutate + 1][1] == choice:
                        continue
                    elif op_to_mutate % 2 == 1 and mutable_cell[op_to_mutate - 1][1] == choice:
                        continue
                    elif mutable_cell[op_to_mutate][1] == choice:
                        continue
                    else:
                        mutable_cell[op_to_mutate][1] = choice
                        break
        mutated_cell = [tuple(i) for i in mutable_cell]
        return mutated_cell

    if mutate_main_only:
        mutated = _mutate_cell(arch.normal, edits)
        mutated_genotype = Genotype(normal=mutated, normal_concat=arch.normal_concat,
                                    reduce=mutated, reduce_concat=arch.reduce_concat)
    else:
        choice = np.random.choice(2)
        if choice == 1:  # Mutate the main
            mutated_normal = _mutate_cell(arch.normal, edits)
            mutated_reduce = arch.reduce
        else:
            mutated_normal = arch.normal
            mutated_reduce = _mutate_cell(arch.reduce, edits)
        mutated_genotype = Genotype(normal=mutated_normal, normal_concat=arch.normal_concat,
                                    reduce=mutated_reduce, reduce_concat=arch.reduce_concat)
    mutated_digraph = darts2graph(mutated_genotype)
    return mutated_digraph, mutated_genotype


def mutation_darts(n, search_space, parents, edits, same_arch=True, n_rand=None, ):
    if not same_arch:
        raise NotImplementedError
    res = []
    if n_rand is None:
        n_rand = n
    while len(res) < n:
        # Randomly choose a parent
        parent_arch_idx = np.random.choice(len(parents))
        parent_arch = parents[parent_arch_idx]
        # Mutate the parent
        (child_arch_main, _), child_genotype = _mutate(parent_arch, search_space, edits, same_arch)
        while not is_valid_darts(child_genotype):
            (child_arch_main, _), child_genotype = _mutate(parent_arch, search_space, edits, same_arch)
        res.append((child_arch_main, child_genotype))
    if n_rand > 0:
        rand_archs = random_sample_darts(n_rand, search_space, same_arch=same_arch, )
        res.extend(rand_archs)
    return res

#
# if __name__ == '__main__':
#     from misc.draw_nx import draw_graph
#     from darts.utils import darts2graph
#
#     parent = random_sample_darts(1)[0][1]
#     mutated = mutation_darts(10, [parent], 1)
#     for m in mutated:
#         draw_graph(m[0])

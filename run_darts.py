from darts.darts_objectives import DARTSObjective, random_sample_darts, mutation_darts
import tabulate
import bayesopt, kernels
import pandas as pd
from kernels import *
import argparse
import time
import torch
import os
from misc.random_string import random_generator
import pickle
from datetime import datetime

parser = argparse.ArgumentParser(description='DARTS Search')
parser.add_argument('--n_init', type=int, default=10)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--eval_policy', type=str, default='last5', help='The policy to query an architecture')
parser.add_argument('--data_path', type=str, default='data/')
parser.add_argument('--max_archs', type=int, default=200)
parser.add_argument('--pool_size', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--pool_strategy', default='mutate')
parser.add_argument('--save_path', default='results/darts/')
parser.add_argument('--cutout', action='store_true')
parser.add_argument('--auxiliary', action='store_true')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('-a', '--acquisition', default='EI')
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('--maximum_noise', type=float, default=0.01)
parser.add_argument('-d', '--draw_arch', action='store_true')
parser.add_argument('--mutate_k_archs', type=int, default=10)
parser.add_argument('--n_mutation_edits', type=int, default=1)
parser.add_argument('--resume_from_path', type=str, default=None)
parser.add_argument('--search_space', default='nasnet', type=str, help='search space to use. DARTS or NASNet')
parser.add_argument('--save_topn', default=8, type=int)

# debugging
dummy_val = False


def main():
    args = parser.parse_args()
    options = vars(args)
    print(options)

    assert args.dataset in ['cifar10', 'cifar100', 'imagenet']
    assert args.eval_policy in ['best', 'last', 'last5']
    if args.dataset == 'imagenet':
        raise NotImplementedError
    assert args.search_space in ['darts', 'nasnet']
    if args.search_space == 'darts':
        n_towers = 4
    else:
        n_towers = 7

    args.data_path += '/' + args.dataset
    if args.resume_from_path is None:
        args.save_path = os.path.join(args.save_path, str(datetime.now().strftime("%Y%m%d_%H%M")))
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path + "/")
    else:
        args.save_path = args.resume_from_path
    option_file = open(args.save_path + "/command.txt", "w+")
    option_file.write(str(options))
    option_file.close()

    o = DARTSObjective(dataset=args.dataset, data_path=args.data_path, save_path=args.save_path + "/",
                       cutout=args.cutout, dummy_eval=dummy_val,  # at training stage, apply no cutout by default
                       auxiliary=args.auxiliary,
                       query_policy=args.eval_policy,
                       epochs=args.epochs)

    columns = ['Iteration', 'Best func val', 'Time', 'TrainTime']
    total_iters = (args.max_archs - args.n_init) // args.batch_size
    res = pd.DataFrame(np.nan, index=range(total_iters + 1), columns=columns)

    if args.resume_from_path is None:
        xs = random_sample_darts(args.n_init, args.search_space, same_arch=True, )
        x = [_[0] for _ in xs]
        x_genotypes = [_[1] for _ in xs]
        y, y_stats = o.eval(x_genotypes) if not dummy_val else o.eval(x)
        valid_idx = [k for k, v in enumerate(y) if v == v]
        y = [y[i] for i in range(len(y)) if i in valid_idx]
        x = [x[i] for i in range(len(x)) if i in valid_idx]
        x_genotypes = [x_genotypes[i] for i in range(len(x_genotypes)) if i in valid_idx]

        # Actual evaluation of the archs. This is the bottleneck step
        y = torch.tensor(y).view(-1).float()
        kern = []
        for k in args.kernels:
            # Graph kernels
            if k == 'wl':
                k = WeisfilerLehman(oa=True, h=2, )
            elif k == 'mlk':
                k = MultiscaleLaplacian(n=1)
            else:
                try:
                    k = getattr(kernels, k)
                    k = k()
                except AttributeError:
                    logging.warning('Kernel type ' + str(k) + ' is not understood. Skipped.')
                    continue
            kern.append(k)
        if kern is None:
            raise ValueError("None of the kernels entered is valid. Quitting.")

        # Initialise GPWL instance
        gp = bayesopt.GraphGP(x, y, kern, verbose=args.verbose)
        gp.fit(max_lik=args.maximum_noise, wl_subtree_candidates=(0, 1, 2, 3))
        data = {'x': x, 'x_genotypes': x_genotypes, 'y': y, 'iters': 0, 'gp': gp}
        res.iloc[0, :] = ["0", str(torch.exp(-torch.max(y)).numpy()), np.nan, np.nan]  # Record the archs at init.
        res.to_csv(args.save_path + "/stats.csv")
        pickle.dump(data, open(args.save_path + "/data.pickle", 'wb'))
    else:
        data = pickle.load(open(args.resume_from_path + "/data.pickle", 'rb'))
        x, x_genotypes, y, gp, y_stats = data['x'], data['x_genotypes'], data['y'], data['gp'], data['y_stats']
        gp.reset_XY(x, y)
        gp.fit(max_lik=args.maximum_noise, wl_subtree_candidates=(0, 1, 2, 3))
        try:
            res = pd.read_csv(args.save_path + "/stats.csv")
            res = res.iloc[:, 1:]
        except FileNotFoundError:
            logging.warning("stats.csv file not found. creating a new copy.")
            res = pd.DataFrame(np.nan, index=range(args.total_iters + 1), columns=columns)
        print('Resuming at iteration', data['iters'])

    starting_iter = 0 if args.resume_from_path is None else data['iters']

    # Optimisation Loop
    prev_best_idx = np.nan
    best_index = 0
    for i in range(starting_iter, total_iters):
        t1 = time.time()
        if args.pool_strategy == 'random':
            pools = random_sample_darts(args.pool_size, True, n_towers)
        elif args.pool_strategy == 'mutate':
            # Find the indices of the top-k best performing archs -- select these as parents for mutation
            best_arch_index = y.numpy().argsort()[:min(args.mutate_k_archs, y.shape[0])]
            parent_archs = [x_genotypes[i] for i in best_arch_index]
            pools = mutation_darts(args.pool_size, args.search_space, parent_archs,
                                   args.n_mutation_edits, True, n_rand=args.pool_size, )
        else:
            raise NotImplementedError
        genotype = [p[1] for p in pools]
        pool = [p[0] for p in pools]
        a = bayesopt.GraphExpectedImprovement(gp)
        next_x, eis, indices = a.propose_location(candidates=pool, top_n=args.batch_size)
        next_x_genotypes = [genotype[i] for i in indices]
        # Evaluate this
        t2 = time.time()
        gp_time = t2 - t1
        next_y, next_y_stats = o.eval(next_x_genotypes) if not dummy_val else o.eval(next_x)
        # Remove the invalid indices
        valid_idx = [k for k, v in enumerate(next_y) if v == v]
        next_y = [next_y[i] for i in range(len(next_y)) if i in valid_idx]
        next_x = [next_x[i] for i in range(len(next_y)) if i in valid_idx]
        next_y_stats = [next_y_stats[i] for i in range(len(next_y)) if i in valid_idx]
        next_x_genotypes = [next_x_genotypes[i] for i in range(len(next_x_genotypes)) if i in valid_idx]

        # Actual evaluation of the archs. This is the bottleneck step
        t3 = time.time()
        train_time = t3 - t2
        # Update GP surrogate
        x.extend(next_x)
        x_genotypes.extend(next_x_genotypes)
        y_stats.extend(next_y_stats)
        y = torch.cat((y, torch.tensor(next_y).view(-1).float()))
        gp.reset_XY(x, y)
        # print(x)
        gp.fit(max_lik=args.maximum_noise, wl_subtree_candidates=(0, 1, 2, 3))

        # Update eval metrics
        best_vals = torch.exp(-torch.max(y))
        best_index = int(torch.argmax(y))

        values = [str(i + 1), str(best_vals.numpy()), str(gp_time), str(train_time)]
        table = tabulate.tabulate([values], headers=columns, tablefmt='simple', floatfmt='8.4f')
        print(table)
        res.iloc[i + 1, :] = values
        save_path = args.save_path + random_generator() + "/"
        if prev_best_idx != best_index:
            print('New best arch found!')

        if args.draw_arch:
            from misc.draw_nx import draw_graph

            filename = save_path + "/arch_draws/"
            if not os.path.exists(filename):
                os.makedirs(filename)
            if prev_best_idx != best_index:
                draw_graph(x[best_index])
        prev_best_idx = best_index
        data = {'x': x, 'x_genotypes': x_genotypes, 'y': y, 'iters': i + 1, 'gp': gp, 'y_stats': y_stats}
        res.to_csv(args.save_path + "/stats.csv")
        pickle.dump(data, open(args.save_path + "/data.pickle", 'wb'))

    # Return the top-k architectures
    _, indices = y.topk(min(y.shape[0], args.save_topn))
    top_genotypes = [x_genotypes[int(i)] for i in indices]
    pickle.dump(top_genotypes, open(args.save_path + "/top_genotypes.pickle", 'wb'))


if __name__ == '__main__':
    main()

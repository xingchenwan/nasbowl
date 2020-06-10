# This script tests the graph BO on NASBench101 dataset

import argparse
import pickle
import time  # time module to randomise seed outside the fixed seed regimes

import pandas as pd
import tabulate
import os
import bayesopt
import kernels
from bayesopt.generate_test_graphs import *
from benchmarks import NAS101Cifar10, NAS201
from kernels import *
from misc.random_string import random_generator

parser = argparse.ArgumentParser(description='NAS-BOWL')
parser.add_argument('--dataset', default='nasbench101', help='The benchmark dataset to run the experiments. '
                                                             'options = ["nasbench101", "nasbench201"].')
parser.add_argument('--task', default=['cifar10-valid'],
                    nargs="+", help='the benchmark task *for nasbench201 only*.')
parser.add_argument('--task_weights', default=None, help='the weights assigned to the tasks *for n201 only*')

parser.add_argument("--use_12_epochs_result", action='store_true',
                    help='Whether to use the statistics at the end of the 12th epoch, instead of using the final '
                         'statistics *for nasbench201 only*')
parser.add_argument('--n_repeat', type=int, default=20, help='number of repeats of experiments')
parser.add_argument("--data_path", default='data/')
parser.add_argument('--n_init', type=int, default=30, help='number of initialising points')
parser.add_argument("--max_iters", type=int, default=17, help='number of maximum iterations')
parser.add_argument('--pool_size', type=int, default=100, help='number of candidates generated at each iteration')
parser.add_argument('--mutate_size', type=int, help='number of mutation candidates. Only applicable for mutate'
                                                    'or grad_mutate. By default, half of the pool_size is generated'
                                                    'from mutation.')
parser.add_argument('--pool_strategy', default='mutate', help='the pool generation strategy. Options: random,'
                                                              ' regularised evolutionary, random graph generation')
parser.add_argument('--save_path', default='results/', help='path to save log file')
parser.add_argument('-s', '--strategy', default='gbo', help='optimisation strategy: option: gbo (graph bo), '
                                                            'random (random search)')
parser.add_argument('-a', "--acquisition", default='EI', help='the acquisition function for the BO algorithm. option: '
                                                              'UCB, EI, AEI')
parser.add_argument('-k', '--kernels', default=['wl'],
                    nargs="+",
                    help='graph kernel to use. This can take multiple input arguments, and '
                         'the weights between the kernels will be automatically determined'
                         ' during optimisation (weights will be deemed as additional '
                         'hyper-parameters.')
parser.add_argument('--batch_size', type=int, default=5, help='Number of samples to evaluate')
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--cuda', action='store_true', help='Whether to use GPU acceleration')
parser.add_argument('--fixed_query_seed', type=int, default=None,
                    help='Whether to use deterministic objective function as NAS-Bench-101 has 3 different seeds for '
                         'validation and test accuracies. Options in [None, 0, 1, 2]. If None the query will be '
                         'random.')
parser.add_argument('--load_from_cache', action='store_true', help='Whether to load the pickle of the dataset. ')
parser.add_argument('--use_node_weight', action='store_true', help='Whether to use different node weights. Experimental'
                                                                   'feature')
parser.add_argument('--mutate_unpruned_archs', action='store_true',
                    help='Whether to mutate on the unpruned archs. This option is only valid if mutate or grad_mutate '
                         'is specified as the pool_strategy')
parser.add_argument('--no_isomorphism', action='store_true', help='Whether to allow mutation/grad_mutation to return'
                                                                  'isomorphic architectures')
parser.add_argument('--maximum_noise', default=0.01, type=float, help='The maximum amount of GP jitter noise variance')
args = parser.parse_args()
options = vars(args)
print(options)

if args.seed is not None:
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

if args.cuda and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

assert args.strategy in ['random', 'gbo']
assert args.pool_strategy in ['random', 'mutate', 'grad_mutate']

# Initialise the objective function. Negative ensures a maximisation task that is assumed by the acquisition function.

# Persistent data structure...
cache_path = 'data/' + args.dataset + '.pickle'

o = None
if args.load_from_cache:
    if os.path.exists(cache_path):
        try:
            o = pickle.load(open(cache_path, 'rb'))
            o.seed = args.fixed_query_seed
            if args.dataset == 'nasbench201':
                o.task = args.task[0]
                o.use_12_epochs_result = args.use_12_epochs_result
        except:
            pass

if o is None:
    if args.dataset == 'nasbench101':
        o = NAS101Cifar10(data_dir=args.data_path, negative=True, seed=args.fixed_query_seed)
    elif args.dataset == 'nasbench201':
        o = NAS201(data_dir=args.data_path, negative=True, seed=args.fixed_query_seed, task=args.task[0],
                   use_12_epochs_result=args.use_12_epochs_result)

    else:
        raise NotImplementedError("Required dataset " + args.dataset + " is not implemented!")

# init_data_list = []
for j in range(args.n_repeat):
    start_time = time.time()
    best_tests = []
    best_vals = []
    # 2. Take n_init_point random samples from the candidate points to warm_start the Bayesian Optimisation
    x, x_config, x_unpruned = random_sampling(args.n_init, benchmark=args.dataset, save_config=True,
                                              return_unpruned_archs=True)
    y_np_list = [o.eval(x_) for x_ in x]
    y = torch.tensor([y[0] for y in y_np_list])
    train_details = [y[1] for y in y_np_list]

    # The test accuracy from NASBench. This is retrieved only for reference, and is not used in BO at all
    test = torch.tensor([o.test(x_) for x_ in x])
    # Initialise the GP surrogate and the acquisition function
    pool = x[:]
    unpruned_pool = x_unpruned[:]
    #
    # pool, unpruned_pool = [], []
    kern = []

    node_weights = None

    for k in args.kernels:
        # Graph kernels
        if k == 'wl':
            k = WeisfilerLehman(h=2,  oa=args.dataset != 'nasbench201',
                                node_weights=node_weights, requires_grad=args.pool_strategy == 'grad_mutate', )
        elif k == 'mlk':
            k = MultiscaleLaplacian(n=1)
        elif k == 'vh':
            k = WeisfilerLehman(h=0, node_weights=node_weights, oa=args.dataset != 'nasbench201',
                                requires_grad=args.pool_strategy == 'grad_mutate',
                                )
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
    if args.strategy != 'random':
        gp = bayesopt.GraphGP(x, y, kern, verbose=args.verbose)
        gp.fit(wl_subtree_candidates=(0,) if args.kernels[0] == 'vh' else tuple(range(1, 3)),
               optimize_lik=args.fixed_query_seed is None,
               max_lik=args.maximum_noise
               # optimize_lik=True,
               )
    else:
        gp = None

    # 3. Main optimisation loop
    columns = ['Iteration', 'Last func val', 'Best func val', 'True optimum in pool',
               'Pool regret', 'Last func test', 'Best func test', 'Time', 'TrainTime']

    res = pd.DataFrame(np.nan, index=range(args.max_iters), columns=columns)
    sampled_idx = []
    for i in range(args.max_iters):
        # Generate a pool of candidates from a pre-specified strategy
        if args.pool_strategy == 'random':
            pool, _, unpruned_pool = random_sampling(args.pool_size, benchmark=args.dataset, return_unpruned_archs=True)
        elif args.pool_strategy == 'mutate':
            pool, unpruned_pool = mutation(x, y, benchmark=args.dataset, pool_size=args.pool_size,
                                           n_best=10,
                                           n_mutate=args.mutate_size if args.mutate_size else args.pool_size // 2,
                                           observed_archs_unpruned=x_unpruned if args.mutate_unpruned_archs else None,
                                           allow_isomorphism=not args.no_isomorphism)
        elif args.pool_strategy == 'grad_mutate':
            index_of_wl = [kern.index(k) for k in kern if isinstance(k, WeisfilerLehman)]
            assert len(index_of_wl) == 1, " If using grad_mutate, need to have exactly one WeisfeilerLehman kernel in" \
                                          "the list of kernels!"
            wl = kern[index_of_wl[0]]
            pool, unpruned_pool = grad_guided_mutation(x, y, wl, gp, n_best=10,
                                                       n_mutate=args.mutate_size if args.mutate_size else args.pool_size // 2,
                                                       pool_size=args.pool_size,
                                                       benchmark=args.dataset,
                                                       observed_archs_unpruned=x_unpruned if args.mutate_unpruned_archs
                                                       else None,
                                                       allow_isomorphism=not args.no_isomorphism)
        else:
            pass

        if args.strategy != 'random':
            if args.acquisition == 'UCB':
                a = bayesopt.GraphUpperConfidentBound(gp)
            elif args.acquisition == 'EI':
                a = bayesopt.GraphExpectedImprovement(gp, in_fill='best', augmented_ei=False)
            elif args.acquisition == 'AEI':
                # Uses the augmented EI heuristic and changed the in-fill criterion to the best test location with
                # the highest *posterior mean*, which are preferred when the optimisation is noisy.
                a = bayesopt.GraphExpectedImprovement(gp, in_fill='posterior', augmented_ei=True)
            else:
                raise ValueError("Acquisition function" + str(args.acquisition) + ' is not understood!')
        else:
            a = None

        # Ask for a location proposal from the acquisition function
        if args.strategy == 'random':
            next_x = random.sample(pool, args.batch_size)
            sampled_idx.append(next_x)
            next_x_unpruned = None
        else:
            next_x, eis, indices = a.propose_location(top_n=args.batch_size, candidates=pool)
            next_x_unpruned = [unpruned_pool[i] for i in indices]
        # Evaluate this location from the objective function
        detail = [o.eval(x_) for x_ in next_x]
        next_y = [y[0] for y in detail]
        train_details += [y[1] for y in detail]
        next_test = [o.test(x_).item() for x_ in next_x]
        # Evaluate all candidates in the pool to obtain the regret (i.e. true best *in the pool* compared to the one
        # returned by the Bayesian optimiser proposal)
        pool_vals = [o.eval(x_)[0] for x_ in pool]
        if gp is not None:
            pool_preds = gp.predict(pool, preserve_comp_graph=args.pool_strategy == 'grad_mutate')
            pool_preds = [p.detach().cpu().numpy() for p in pool_preds]
            pool.extend(next_x)

        # Update the GP Surrogate
        x.extend(next_x)
        if args.pool_strategy in ['mutate', 'grad_mutate']:
            x_unpruned.extend(next_x_unpruned)
        y = torch.cat((y, torch.tensor(next_y).view(-1)))
        test = torch.cat((test, torch.tensor(next_test).view(-1)))

        if args.strategy != 'random':
            gp.reset_XY(x, y)
            gp.fit(wl_subtree_candidates=(0,) if args.kernels[0] == 'vh' else tuple(range(1, 3)),
                   optimize_lik=args.fixed_query_seed is None,
                   max_lik=args.maximum_noise
                   )

            # Compute the GP posterior distribution on the trainning inputs
            train_preds = gp.predict(x, preserve_comp_graph=args.pool_strategy == 'grad_mutate')
            train_preds = [t.detach().cpu().numpy() for t in train_preds]

        zipped_ranked = list(sorted(zip(pool_vals, pool), key=lambda x: x[0]))[::-1]
        true_best_pool = np.exp(-zipped_ranked[0][0])

        # Updating evaluation metrics
        best_val = torch.exp(-torch.max(y))
        pool_regret = np.abs(np.exp(-np.max(next_y)) - true_best_pool)
        best_test = torch.exp(-torch.max(test))

        end_time = time.time()
        # Compute the cumulative training time.
        try:
            cum_train_time = np.sum([item['train_time'] for item in train_details]).item()
        except TypeError:  # cause by np.nan if an older version of the NAS-Bench-201 dataset is used
            cum_train_time = np.nan
        values = [str(i), str(np.exp(-np.max(next_y))), best_val.item(), true_best_pool, pool_regret,
                  str(np.exp(-np.max(next_test))), best_test.item(), str(end_time - start_time),
                  str(cum_train_time)]
        table = tabulate.tabulate([values], headers=columns, tablefmt='simple', floatfmt='8.4f')
        best_vals.append(best_val)
        best_tests.append(best_test)

        if i % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)
        res.iloc[i, :] = values

    if args.save_path is not None:
        kernels_str = ",".join(args.kernels)
        save_path = args.save_path + args.dataset + "/"
        if args.dataset == 'nasbench201':
            save_path += str(args.task) + "/"
        save_path += args.strategy + '_Pool' + str(args.pool_size) + \
                     '_Batch' + str(args.batch_size) + '_Kernel' + kernels_str + "/"
        if args.strategy != 'random':
            save_path += "_Acq" + args.acquisition + "_PoolStrategy" + args.pool_strategy + "_ninit" + str(args.n_init)
        if args.mutate_unpruned_archs and args.pool_strategy in ['mutate', 'grad_mutate']:
            save_path += '_unpruned_mutate'
        if args.fixed_query_seed is not None:
            save_path += '_queryseed' + str(args.fixed_query_seed)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        res.to_csv(save_path + '/Iter_' + str(j) + random_generator() + ".csv")
        # Save the args to a txt file
        option_file = open(save_path + "/command.txt", "w+")
        option_file.write(str(options))
        option_file.close()
    if args.seed is not None:
        args.seed += 1024

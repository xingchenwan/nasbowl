import argparse
import time

import matplotlib.pyplot as plt
import pandas as pd
import pickle
import tabulate

from bayesopt import GraphGP, random_sampling
from benchmarks import NAS201, NAS101Cifar10
from kernels import *
from perf_metrics import *

parser = argparse.ArgumentParser(description='Regression')
parser.add_argument('--n_train', type=int, default=50)
parser.add_argument('--n_test', type=int, default=400)
parser.add_argument('--n_repeat', type=int, default=20)
parser.add_argument('--dataset', default='nasbench101')
parser.add_argument('--data_path', default='data/')
parser.add_argument('--task', default='cifar10-valid')
parser.add_argument('--load_from_cache', action='store_true')
parser.add_argument('-p', '--plot', action='store_true')
parser.add_argument('-k', '--kernels', default=['wl'], nargs="+")
parser.add_argument('-w', '--weights', default=None, type=float, nargs='+')
# ##############################################

args = parser.parse_args()
options = vars(args)
print(options)

# ## Set parameters of regression experiments
graph_k = []
for k in args.kernels:
    if k == 'wl':
        graph_k.append(WeisfilerLehman(h=2, oa=args.dataset != 'nasbench201', ))
    elif k == 'mlp':
        graph_k.append(MultiscaleLaplacian())
    else:
        print('Unrecognised kernel', k)
        pass
weights = args.weights
if weights is not None:
    assert len(weights) == len(graph_k), "when weights variable is specified, its length must match the number of " \
                                         "kernels supplied."
cache_path = args.data_path + args.dataset + '.pickle'
o = None
if args.load_from_cache:
    try:
        o = pickle.load(open(cache_path, 'rb'))
        o.seed = 3
        if args.dataset == 'nasbench201':
            o.task = args.task
    except:
        print('Error in loading pickle')
        o = None

if o is None:
    if args.dataset == 'nasbench101':
        o = NAS101Cifar10(args.data_path, seed=3)
    elif args.dataset == 'nasbench201':
        o = NAS201('data/', task=args.task, seed=3)
    else:
        raise NotImplementedError
    pickle.dump(o, open(cache_path, 'wb'))

table_heading = ['RMSE', 'Spearman', 'NLL', 'Time']
res = pd.DataFrame(np.nan, columns=table_heading, index=np.arange(args.n_repeat))

for i in range(args.n_repeat):
    start = time.time()
    n_samples = args.n_train + args.n_test
    X, _, _ = random_sampling(n_samples, args.dataset,)
    Y = [-o.eval(x_)[0] for x_ in X]

    X_train, X_val = X[:args.n_train], X[args.n_train:]
    Y_train, Y_val = Y[:args.n_train], Y[args.n_train:]

    gp = GraphGP(X_train, torch.tensor(Y_train, ), kernels=graph_k, weights=weights, verbose=False)
    gp.fit(optimize_lik=True, )
    time_taken = time.time() - start

    Y_preds, Y_pred_stds = gp.predict(X_s=X_val)
    Y_train_preds, Y_train_pred_stds = gp.predict(X_s=X_train)

    # PLOT
    # Detach from the graph and numpify the arrays for plotting
    Y_train_preds = Y_train_preds.detach().numpy()
    Y_preds = Y_preds.detach().numpy()
    Y_pred_stds = np.sqrt(np.diag(Y_pred_stds.detach().numpy()))
    Y_train_pred_stds = np.sqrt(np.diag(Y_train_pred_stds.detach().numpy()))

    if args.plot:
        import matplotlib

        matplotlib.rcParams.update({'font.size': 15})

        plt.figure(figsize=(4, 4))
        x = np.linspace(np.min(Y_train), np.max(Y_train), 100)
        plt.plot(x, x, ":", color='black')
        plt.xlabel('Train Target')
        plt.ylabel('GP prediction at training points')
        plt.errorbar(Y_train, Y_train_preds, fmt='.', yerr=np.array(Y_train_pred_stds), capsize=2, color='gray', )
        plt.show()

        plt.figure(figsize=(4, 4))
        x = np.linspace(np.min(Y_val), np.max(Y_val), 100)

        plt.plot(x, x, ":", color='black')
        plt.xlabel('Validation Target')
        plt.ylabel('Validation Prediction')
        plt.errorbar(Y_val, Y_preds, fmt='.', yerr=np.array(Y_pred_stds), capsize=2, color='gray',
                     markerfacecolor='blue', markersize=8, alpha=0.3)
        print('RMSE: ', rmse(Y_preds, Y_val))
        print('Spearman: ', spearman(Y_preds, Y_val))
        print('NLL: ', nll(Y_preds, Y_pred_stds, Y_val))
        print('Average prediction error', average_error(Y_preds, Y_val))
        plt.show()

    res.iloc[i, :] = [rmse(Y_preds, Y_val), spearman(Y_preds, Y_val), nll(Y_preds, Y_pred_stds, Y_val), time_taken]
    values = [str(i), str(rmse(Y_preds, Y_val)), str(spearman(Y_preds, Y_val)), str(nll(Y_preds, Y_pred_stds, Y_val)),
              str(time_taken)]
    table = tabulate.tabulate([values], headers=table_heading, tablefmt='simple', floatfmt='8.4f')
    if i % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)
path = 'results/regression_' + args.dataset
import os

if not os.path.exists(path):
    os.makedirs(path)
res.to_csv(path + "/" + str(args.kernels) + "_" + args.task + str(args.n_train) + "weights_" + str(args.weights)
           + '.csv')

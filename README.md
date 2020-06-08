## Code Repository for NAS-BOWL (Neural Architecture Search using Bayesian Optimisation with Weisfeiler-Lehman Kernel)


#  Prerequisites and Dependencies
This code repository contains the submission of NAS-BOWL. To run the codes, please see the prerequisites below:
1. Download the NAS-Bench-101 and NAS-Bench-201 datasets and place the data
files under ```data/``` path. We expect these files:
    
    NAS-Bench-101: ```nasbench_only102.tfrecord```
    
    NAS-Bench-201: ```NAS-Bench-201-v1_0-e61699.pth``` OR ```NAS-Bench-201-v1_1-096897.pth```
    
    (Please note if the older NAS-Bench-201 data is used, training time statistics will not be available.)

    and install the relevant NAS-Bench-101 and NAS-Bench-201 APIs.

2. Install the prerequisite packages via ```pip``` or ```conda```. We used Anaconda Python 3.7 for our experiments.
```bash
ConfigSpace==0.4.12
Cython==0.29.16
future==0.18.2
gensim==3.8.0
gpytorch>=0.3.5
grakel==0.1b7
graphviz>=0.5.1
matplotlib==3.1.1
numpy==1.16.4
pandas==0.25.1
scikit-learn==0.21.2
scipy==1.3.1
seaborn==0.9.0
six==1.12.0
statsmodels==0.10.1
tabulate==0.8.3
tensorboard==1.14.0
tensorflow==1.14.0
tensorflow-estimator==1.14.0
torch==1.3.1
tqdm==4.32.1
networkx
```

# Running Experiments
To reproduce the experiments in the paper, see below

1. Search on NAS-Bench-101
    ```bash
    python3 -u nasbench_optimisation.py --dataset nasbench101 --strategy gbo -k wloa --pool_size 200 --batch_size 5 --max_iters 30 --n_repeat 20 --n_init 10
    ```

    This by default runs the NAS-Bench-101 on the stochastic validation error (i.e. randomness in the objective function). For the 
    deterministic version, append ```--fixed_query_seed 3``` to the command above.

2. Search on NAS-Bench-201 (by default on the CIFAR-10 valid dataset.)
    ```bash
    python3 -u nasbench_optimisation.py  --dataset nasbench201 --strategy gbo -k wl --pool_size 200 --mutate_size 200 --batch_size 5 --n_init 10 --max_iters 30
    ```
    Again, append ```--fixed_query_seed 3``` for deterministic objective function. Append ```--task cifar100```
    for CIFAR-100 dataset, and similarly ```--task ImageNet16-120``` for ImageNet16 dataset.
      
3. To reproduce regression examples on NAS-Bench, use
    ```bash
   # For NAS-Bench-101
    python3 -u nas_regression.py --n_repeat 20 --n_train 50 --n_test 400 -k wloa --dataset nasbench101
   
   # For NAS-Bench-201
    python3 -u nas_regression.py --n_repeat 20 --n_train 50 --n_test 400 -k wl --dataset nasbench201
    python3 -u nas_regression.py --n_repeat 20 --n_train 50 --n_test 400 -k wl --dataset nasbench201 --task cifar100
    python3 -u nas_regression.py --n_repeat 20 --n_train 50 --n_test 400 -k wl --dataset nasbench201 --task ImageNet16-120
    ```

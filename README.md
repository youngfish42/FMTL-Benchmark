# FMTL-Benchmark
This repository is used for benchmark testing in the field of Federated Multi-Task Learning (FMTL).

By using the code of this project, we can conduct experiments from three perspectives: data, model and algorithm.





Currently, it supports 2 datasets, 2 types of model architectures and 11 algorithms.
- Dataset: PASCAL-Context, NYUD-v2
- Model architecture: multi-decoder architecture MD, single-decoder architecture TC based on task conditions
- Algorithm:
    - Local training: Local
    - Classic federated learning algorithm: FedAvg
    - Personalized federated learning algorithms: FedProx, FedAMP, FedRep, Ditto
    - Multi-task learning algorithms: CAGrad, PCGrad
    - Federated multi-task learning algorithms: MaT-FL, FedMTL, $\text{FedHCA}^2$



This repository provides the official implementation code of the ICMR'24 paper "Federated Multi-Task Learning on Non-IID Data Silos: An Experimental Study".

Instructions for reproducing the paper's experiments will be added as soon as possible.



To-do list:
- [ ] Provides an experimental running script that reproduces the results of the paper.
- [ ] Improve the code to make it easier to add new optimization algorithms.
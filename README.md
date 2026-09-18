# Discrete Bayesian Neural Networks

This repository contains a PyTorch implementation of Discrete Bayesian Neural Networks based on The Neural Testbed.

Deep Ensembles, as a type of Bayesian Neural Networks, can be used to estimate uncertainty on the prediction of multiple neural networks by collecting votes from each network and computing the difference in those predictions. In this paper, we introduce a method for uncertainty estimation that considers a set of independent categorical distributions for each layer of the network, giving many more possible samples with overlapped layers than in the regular Deep Ensembles. We further introduce an optimized inference procedure that reuses common layer outputs, achieving up to 19× speed up and reducing memory usage quadratically. We also show that the method can be further improved by ranking samples, resulting in models that require less memory and time to run while achieving higher uncertainty quality than Deep Ensembles.

## Run

Use `run_example.sh` to train a single model and evaluate its uncertainty quality. Results are saved in `results` folder and can be visualized by `tools/create_enn_plots.py`

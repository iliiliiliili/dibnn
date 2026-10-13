# Discrete Bayesian Neural Networks

This repository contains a PyTorch implementation of Discrete Bayesian Neural Networks based on The Neural Testbed.


## Install

The following script creates a `conda` environment and installs needed requirements.

```bash
bash -i install.sh
```

## Run

Use `run_example.sh` to train a single model and evaluate its uncertainty quality. Results are saved in `results-discrete ` folder and can be visualized by `tools/create_enn_plots.py`


Use `run_experiments_dbnn.sh` to run full experiments. Results are saved in `results-discrete ` folder and can be visualized by `tools/create_enn_plots.py` as follows:
```bash
python tools/create_enn_plots.py create_ranked_dbnn_tables --summary_input_dims=\[\[1,10,100\]\] --max_num_samples=\[100\] --results_folder=results-discrete --table_folder=tables
```
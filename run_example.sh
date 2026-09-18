mkdir -p ../results/single_runs
ln -s ../results
ln -s ../results/single_runs

PYTHONPATH=. python ./src/experiments/run_testbed_discrete_bnn_parallel.py --input_dim=10 --data_ratio=1 --noise_std=0.1 --agent_name=vnn_best --processes_per_device="${1:-1}" --devices="${2:-7}" --results_folder=results-discrete --seeds=[2605]
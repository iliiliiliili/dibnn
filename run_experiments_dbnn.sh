PYTHONPATH=. python ./src/experiments/run_testbed_discrete_bnn_parallel.py --agent_name=vnn_best --processes_per_device="${1:-1}" --devices="${2:-7}" --results_folder=results-discrete
PYTHONPATH=. python ./src/experiments/run_testbed_discrete_bnn_parallel.py --agent_name=bbb_best --processes_per_device="${1:-1}" --devices="${2:-7}" --results_folder=results-discrete
PYTHONPATH=. python ./src/experiments/run_testbed_discrete_bnn_parallel.py --agent_name=layer_ensembles_best --processes_per_device="${1:-1}" --devices="${2:-7}" --results_folder=results-discrete
PYTHONPATH=. python ./src/experiments/run_testbed_discrete_bnn_parallel.py --agent_name=ensemble_best --processes_per_device="${1:-1}" --devices="${2:-7}" --results_folder=results-discrete 
PYTHONPATH=. python ./src/experiments/run_testbed_discrete_bnn_parallel.py --agent_name=hypermodel_best --processes_per_device="${1:-1}" --devices="${2:-7}" --results_folder=results-discrete --reduce_batch_dims=[1000,100]
PYTHONPATH=. python ./src/experiments/run_testbed_discrete_bnn_parallel.py --agent_name=dropout_best --processes_per_device="${1:-1}" --devices="${2:-7}" --results_folder=results-discrete

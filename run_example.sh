mkdir -p ../results/single_runs
ln -s ../results
ln -s ../results/single_runs

python ./enn/experiments/neurips_2021/run_testbed_best_selected.py --input_dim=10 --data_ratio=1 --noise_std=0.1 --agent_id_start=3 --agent_id_end=5 --agent=true_layer_ensemble_einsum_cor --seed=2605
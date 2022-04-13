#! /bin/bash

if false; then
	# to manually define list of GPUs to use, change false to true in above line, and specify here:
	gpu_inds=( 4 5 6 7 ) 
else
	# otherwise, use all GPUs in the system
	n_gpus_in_system=$( nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
	let max_gpu_ind=$n_gpus_in_system-1
	gpu_inds=($(seq 0 $max_gpu_ind))
fi

n_gpus=${#gpu_inds[@]}
	
file_base=$(mktemp)
pids=""
for i in ${!gpu_inds[@]}; do
    CUDA_VISIBLE_DEVICES=${gpu_inds[$i]} python -m scripts.test_kitti $file_base $n_gpus ${i} &
done

wait < <(jobs -p)

python -m scripts.test_kitti $file_base $n_gpus analyze 

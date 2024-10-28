#!/bin/bash

#SBATCH --mail-user=kchow1@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/kchow1/gpu/proj3-KyleLee95/%j.%N.stdout
#SBATCH --error=/home/kchow1/gpu/proj3-KyleLee95/%j.%N.stderr
#SBATCH --chdir=/home/kchow1/gpu/proj3-KyleLee95
#SBATCH --partition=gpu-all
#SBATCH --gres=gpu:1
#SBATCH --job-name=proj3_job

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib

make clean
make all

# Define image sizes and the data directory
image_sizes=("400" "600" "800" "1000" "1600" "3200")
data_dir="data"
output_dir="output"

# Loop over all .txt files in the data directory
for scene_file in ${data_dir}/scene_*.txt; do
	scene_filename=$(basename "$scene_file" .txt)
	for size in "${image_sizes[@]}"; do
		# Run the sequential raytracer
		./sequential_raytracer -d $size $size $scene_file

		# Run the CUDA raytracer
		./cuda_raytracer -d $size $size $scene_file
		./cuda_raytracer -a -d $size $size $scene_file

		# Define output file paths
		seq_output="${output_dir}/output_sequential_${scene_filename}.ppm"
		cuda_output="${output_dir}/output_cuda_${scene_filename}.ppm"
		cuda_unified_output="${output_dir}/output_cuda_unified_${scene_filename}.ppm"

		# Compare images
		./compare_images $seq_output $cuda_output $scene_file
		./compare_images $seq_output $cuda_unified_output $scene_file

	done
done

python3 create_execution_time_graphs.py
python3 create_mismatched_pixels_graph.py

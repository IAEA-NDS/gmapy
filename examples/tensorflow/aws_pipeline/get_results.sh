#!/bin/bash

example_name="$1"
local_example_dir="../$example_name"

output_files=(
    "01_model_preparation_output.pkl"
    "02_parameter_optimization_output.pkl"
    "03_mcmc_sampling_output.pkl"
)

last_output_file="${output_files[-1]}"

while true; do
    echo Checking if results for "$example_name" are already available...
    aws s3 ls "s3://gmapy-results/$example_name/" | grep "$last_output_file"
    if [ $? -eq 0 ]; then
        break
    fi
    echo Waiting for results of "$example_name"...
    sleep 120
done

if [ ! -d "$local_example_dir" ]; then
    mkdir "$local_example_dir/output"
fi

for outfile in "${output_files[@]}"; do
    local_outfile="$local_example_dir/output/$outfile"
    if [ ! -e "$local_outfile" ]; then
        aws s3 cp "s3://gmapy-results/$example_name/$outfile" "$local_outfile"
    else
        echo "Output file "$local_outfile" already exists. Skipping its download."
    fi
done

#!/bin/bash

example_name="$1"
wait_arg="$2"
instance_info_file="../$example_name/aws_instance_info.json"

if [ -z "$example_name" ]; then
    echo "Please provide an example name as argument." >&2
    exit 1
fi

if [ -e "$instance_info_file" ]; then
    echo "Instance file '$instance_info_file' already exists."
    echo "Assuming that calculation is already running."
    if [ "$wait_arg" == "wait" ]; then
        bash get_results.sh "$example_name"
    fi
    exit 0
fi

echo "prepare and run tensorflow example '${example_name}' on AWS machine..."
echo starting machine...
bash start_aws_machine.sh $instance_info_file
login="ubuntu@$(bash get_instance_ip.sh $instance_info_file)"
echo obtained login $login for ssh connection
echo prepare and execute pipeline...

rem_home="/home/ubuntu/"
pipe_script="setup_and_run_gmapy_pipeline.sh"
ssh -o StrictHostKeyChecking=no -i ephemeral-key.pem $login \
    "cat > $rem_home/$pipe_script" \
    < "$pipe_script" 

ssh -o StrictHostKeyChecking=no -i ephemeral-key.pem -t $login \
    "bash $rem_home/$pipe_script remote $example_name"

if [ "$wait_arg" == "wait" ]; then
    bash get_results.sh "$example_name"
fi

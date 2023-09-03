#!/bin/bash

example_name="$1"

if [ -z "$example_name" ]; then
    echo "Please provide an example name as argument." >&2
    exit 1
fi

echo "prepare and run tensorflow example '${example_name}' on AWS machine..."
echo starting machine...
login=$(bash start_aws_machine.sh)
echo obtained login $login for ssh connection
echo prepare and execute pipeline...

rem_home="/home/ubuntu/"
pipe_script="setup_and_run_gmapy_pipeline.sh"
ssh -o StrictHostKeyChecking=no -i ephemeral-key.pem $login \
    "cat > $rem_home/$pipe_script" \
    < "$pipe_script" 
ssh -o StrictHostKeyChecking=no -i ephemeral-key.pem -t $login \
    "bash $rem_home/$pipe_script remote $example_name"

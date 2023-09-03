#!/bin/bash

instance_info_file="$1"

if [ -z "$instance_info_file" ]; then
    echo "Please provide instance information file as argument." >&2
    exit 1
fi

instance_id=$(cat "$instance_info_file" | jq -r .Instances[0].InstanceId)
new_state="$(aws ec2 terminate-instances --instance-ids ${instance_id})"
state_str=$(echo "$new_state" | jq -r .TerminatingInstances[0].CurrentState.Name)
if [ "$state_str" == "shutting-down"  ]; then
    echo "Terminating the machine with instance id $instance_id ..."
    rm "$instance_info_file"
else
    echo "An error occurred, unable to shut down machine" >&2  
fi

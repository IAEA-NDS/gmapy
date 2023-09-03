#!/bin/bash

machine_info_file="instance_info.json"

instance_id=$(cat "$machine_info_file" | jq -r .Instances[0].InstanceId)
new_state="$(aws ec2 terminate-instances --instance-ids ${instance_id})"
state_str=$(echo "$new_state" | jq -r .TerminatingInstances[0].CurrentState.Name)
if [ "$state_str" == "shutting-down"  ]; then
    echo "Terminating the machine with instance id $instance_id ..."
    rm "$machine_info_file"
else
    echo "An error occurred, unable to shut down machine" >&2  
fi

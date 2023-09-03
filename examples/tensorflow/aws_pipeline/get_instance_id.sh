#!/bin/bash
instance_info_file="$1"
if [ -z "$instance_info_file" ]; then
    echo Please provide instance info file. Aborting. >&2
    exit 1
fi
instance_id=$(cat "$instance_info_file" | jq -r .Instances[0].InstanceId)
echo $instance_id

#!/bin/bash

instance_info_file="$1"
machine_config_file="machine_config.json"

if [ -z "$machine_config_file" ]; then
    echo Please provide a machine configuration filename. >&2
    exit 1
fi

if [ -e "$instance_info_file" ]; then
    echo Instance configuration file $instance_info_file already exists. Aborting. >&2
    echo This may indicate that the machine is already running. >&2
    exit 1
fi

echo "Information about machine will be stored in $instance_info_file"

aws ec2 run-instances --cli-input-json file://${machine_config_file} \
  > "$instance_info_file"

retcode=$?
if [ $retcode -ne 0 ]; then
    rm "$instance_info_file"
    echo "An error occurred starting the AWS machine." >&2
    exit $retcode
fi

instance_id=$(cat "$instance_info_file" | jq -r .Instances[0].InstanceId)
while true; do
    public_ip=$(aws ec2 describe-instances --instance-ids $instance_id \
                --query 'Reservations[0].Instances[0].PublicIpAddress' \
                --output text)
    if [ "$?" -eq 0 ]; then
        break
    fi
    sleep 10
done

# wait for the ssh server to get ready
count=0
while [ $count -le 5 ]; do
    login="ubuntu@$public_ip"
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -i ephemeral-key.pem $login "sleep 0" 
    if [ "$?" -eq 0 ]; then
        break
    fi
    sleep 10
    ((count++))
done

echo $login

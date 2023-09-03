#!/bin/bash
machine_info_file="instance_info.json"

instance_id=$(cat "$machine_info_file" | jq -r .Instances[0].InstanceId)
public_ip=$(aws ec2 describe-instances --instance-ids $instance_id \
            --query 'Reservations[0].Instances[0].PublicIpAddress' \
            --output text)
echo $public_ip

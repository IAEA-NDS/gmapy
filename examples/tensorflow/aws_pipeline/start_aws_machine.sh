#!/bin/bash

machine_info_file="instance_info.json"

if [ -e "$machine_info_file" ]; then
    echo "It seems a machine is already running." >&2
    echo "Check the $machine_info_file for more info." >&2
    exit 1
fi

aws ec2 run-instances \
  --region us-east-1 \
  --image-id ami-053b0d53c279acc90 \
  --instance-type r6i.2xlarge \
  --key-name ephemeral-key \
  --security-group-ids sg-0322b8b8a6675e869 \
  --block-device-mappings \
    '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":30,"VolumeType":"gp2"}}]' \
  > "$machine_info_file"

retcode=$?
if [ $retcode -ne 0 ]; then
    rm "$machine_info_file"
    echo "An error occurred starting the AWS machine." >&2
    exit $retcode
fi

instance_id=$(cat "$machine_info_file" | jq -r .Instances[0].InstanceId)
public_ip=$(aws ec2 describe-instances --instance-ids $instance_id \
            --query 'Reservations[0].Instances[0].PublicIpAddress' \
            --output text)

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

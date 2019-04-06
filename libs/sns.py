import os
import boto3
import json
if os.environ.get('STAGE') is None:
    from .config import *

#If the env keys are set the get from env
if os.environ.get('STAGE') is not None:
    STAGE = os.environ.get('STAGE')
    ACC_ARN = os.environ.get('ACC_ARN')

client = boto3.client('sns', region_name='ap-southeast-2')

def send_sns(topic, message):
    if STAGE == 'prod':
        response = client.publish(
            TopicArn='arn:aws:sns:ap-southeast-2:{}:{}'.format(ACC_ARN, topic),
            Message=message,
        )
        print("response",response)

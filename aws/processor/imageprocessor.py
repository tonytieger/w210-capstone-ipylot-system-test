import numpy as np
import cv2
import base64
import paho.mqtt.client as mqtt
import time
import pickle
import boto3
from botocore.exceptions import NoCredentialsError
import requests
import mimetypes

#LOCAL_MQTT_HOST="mosquitto-service"
#LOCAL_MQTT_HOST="mosquitto-service"
LOCAL_MQTT_HOST="localhost"
LOCAL_MQTT_PORT=1883
LOCAL_MQTT_TOPIC="face_detector_topic"

#NOTE: This is NOT the right way to do this. A better practice is to simply assign an IAM Role to the instance with appropriate permissions. The credentials will automatically be used.

ACCESS_KEY_ID = 'AKIA235PFUFLQMRWMEBS'
SECRET_ACCESS_KEY = '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'


message_count = int(0)
def on_connect_local(client, userdata, flags, rc):
        print("Image processor : connected to aws broker with rc: " + str(rc))
        client.subscribe(LOCAL_MQTT_TOPIC)

def on_message(client,userdata, msg):
        try:
            print("Image processor : message received: ", get_datetime())
            image = pickle.loads(msg.payload)
            png = cv2.imdecode(image, 0)
            filename_str = '/apps/data/img_' + str(message_count) + '.png'
            print("Image processor : filename ok " + filename_str)
            cv2.imwrite(filename_str, png)

            fname = 'img_' + str(message_count) + '.png'
            client = boto3.client('s3', aws_access_key_id=ACCESS_KEY_ID, aws_secret_access_key=SECRET_ACCESS_KEY)
            client.upload_file(filename_str,'251hw3', fname)


        except Exception as e: # work on python 3.x
            print('Exception: '+ str(e))


def get_datetime():
    global message_count
    message_count += 1
    time_tuple = time.localtime()
    date_str = time.strftime("%m/%d/%Y", time_tuple)
    time_str = time.strftime("%H:%M:%S", time_tuple)
    return {"date": date_str, "time": time_str, "count": message_count}

local_mqttclient = mqtt.Client()
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
local_mqttclient.on_message = on_message
# go into a loop
local_mqttclient.loop_forever()


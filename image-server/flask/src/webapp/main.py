from curses.ascii import US
from flask import Flask, jsonify, request
from flask_restful import Resource, Api
import boto3 #AWS 연결라이브러리
import tinify
import os
import asyncio

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

app = Flask(__name__)

os.environ['AWS_DEFAULT_REGION'] = 'ap-northeast-1'

S3_CLIENT=boto3.client('s3',
             aws_access_key_id="",
             aws_secret_access_key=''
            )
tinify.key = ""

@app.route('/<ID_BUCKET>/<UserID>/<url>')
def test(ID_BUCKET,UserID,url):
    createFolder(UserID)
    S3_url = UserID + "/" + url
    S3_CLIENT.download_file(Filename = S3_url, Bucket= ID_BUCKET, Key= S3_url)
    source = tinify.from_file(S3_url)
    source.to_file(S3_url)
    S3_CLIENT.upload_file(Filename = S3_url, Bucket= ID_BUCKET, Key= S3_url)
    result = {
        'compress': 'True'
    }


    return jsonify(result)
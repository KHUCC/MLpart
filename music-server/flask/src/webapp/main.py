from curses.ascii import US
from flask import Flask, jsonify, request
from flask_restful import Resource, Api
import boto3 #AWS 연결라이브러리
import os
from sre_constants import CATEGORY
import numpy as np
import pandas as pd
import librosa
from xgboost import XGBClassifier
import joblib
import gc



app = Flask(__name__)

os.environ['AWS_DEFAULT_REGION'] = 'ap-northeast-1'

S3_CLIENT=boto3.client('s3',
             aws_access_key_id="",
             aws_secret_access_key=''
            )





CATEGORY = ['Dance','Rap and Hip-Hop','Rock and Metal','Ballad','Folk and Blues','R&B and Soul']

FEATURES = [
    'mean_stft', 'var_stft','tempo', 'rms_mean', 'rms_var',
    'centroid_mean', 'centroid_var', 'bandwidth_mean', 'bandwidth_var',
    'rolloff_mean', 'rolloff_var', 'crossing_mean', 'crossing_var',
    'harmonic_mean', 'harmonic_var', 'contrast_mean', 'contrast_var',
    'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var', 'mfcc3_mean',
    'mfcc3_var', 'mfcc4_mean', 'mfcc4_var', 'mfcc5_mean', 'mfcc5_var',
    'mfcc6_mean', 'mfcc6_var', 'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean',
    'mfcc8_var', 'mfcc9_mean', 'mfcc9_var', 'mfcc10_mean', 'mfcc10_var',
    'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var', 'mfcc13_mean',
    'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean', 'mfcc15_var',
    'mfcc16_mean', 'mfcc16_var', 'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean',
    'mfcc18_var', 'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var',
    ]

def feature_extract(filepath):
    pitches = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    ls_mean = []
    ls_var =[]
    

    y,sr = librosa.load(filepath,res_type='kaiser_fast')
    length = len(y)/sr
    mean_stft = np.mean(librosa.feature.chroma_stft(y=y,sr=sr))
    var_stft = np.var(librosa.feature.chroma_stft(y=y,sr=sr))
    tempo = librosa.beat.tempo(y,sr=sr)[0]
    
    S,phase = librosa.magphase(librosa.stft(y))
    rms = librosa.feature.rms(S=S)
    rms_mean = np.mean(rms)
    rms_var = np.var(rms)
    
    centroid = librosa.feature.spectral_centroid(S=S)
    centroid_mean = np.mean(centroid)
    centroid_var = np.var(centroid)
    
    bandwidth = librosa.feature.spectral_bandwidth(S=S)
    bandwidth_mean = np.mean(bandwidth)
    bandwidth_var = np.var(bandwidth)
    
    rolloff = librosa.feature.spectral_rolloff(y=y,sr=sr,roll_percent=0.85)
    rolloff_mean = np.mean(rolloff)
    rolloff_var = np.var(rolloff)
    
    zerocrossing = librosa.feature.zero_crossing_rate(y=y)
    crossing_mean = np.mean(zerocrossing)
    crossing_var = np.var(zerocrossing)
    
    y_harmonic = librosa.effects.harmonic(y=y)
    harmonic_mean = np.mean(y_harmonic)
    harmonic_var = np.var(y_harmonic)
    
    contrast = librosa.feature.spectral_contrast(S=S,sr=sr)
    contrast_mean = np.mean(contrast)
    contrast_var = np.var(contrast)

    mfcc= librosa.feature.mfcc(y=y,sr=sr)
    for i in range(0,20):
        ls_mean.append(np.mean(mfcc[i]))
        ls_var.append(np.var(mfcc[i]))
        
    key = librosa.feature.chroma_stft(y=y,sr=sr).sum(axis=1).argmax()
    key_name = pitches[librosa.feature.chroma_stft(y=y,sr=sr).sum(axis=1).argmax()]
    
    if librosa.feature.chroma_stft(y=y,sr=sr).sum(axis=1)[(key+3)%12] > librosa.feature.chroma_stft(y=y,sr=sr).sum(axis=1)[(key+4)%12]:
            scale = 'minor'
    else:
            scale = 'Major'
        

        
    return [mean_stft,var_stft,tempo,rms_mean,rms_var,centroid_mean,centroid_var,\
bandwidth_mean,bandwidth_var,rolloff_mean,rolloff_var, crossing_mean,crossing_var,\
harmonic_mean,harmonic_var,contrast_mean,contrast_var,ls_mean[0],ls_var[0],ls_mean[1],ls_var[1],ls_mean[2],ls_var[2],\
ls_mean[3],ls_var[3],ls_mean[4],ls_var[4],ls_mean[5],ls_var[5],ls_mean[6],ls_var[6],ls_mean[7],ls_var[7],ls_mean[8],ls_var[8],\
ls_mean[9],ls_var[9],ls_mean[10],ls_var[10],ls_mean[11],ls_var[11],ls_mean[12],ls_var[12],ls_mean[13],ls_var[13],ls_mean[14],ls_var[14],\
ls_mean[15],ls_var[15],ls_mean[16],ls_var[16],ls_mean[17],ls_var[17],ls_mean[18],ls_var[18],ls_mean[19],ls_var[19]]


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

@app.route('/<ID_BUCKET>/<UserID>/<url>')
def test(ID_BUCKET,UserID,url):
    createFolder(UserID)
    S3_url = UserID + "/" + url
    S3_CLIENT.download_file(Filename = S3_url, Bucket= ID_BUCKET, Key= S3_url)
    inputs = feature_extract(S3_url)
    inputs = pd.DataFrame(columns= FEATURES, data = np.array([inputs]))
    model = joblib.load("XGB_MusicClassfy")
    pred = model.predict(inputs)

    pred_CAT = CATEGORY[pred[0]]

    out_S3_url = UserID + "/" + pred_CAT + '/' + url

    S3_CLIENT.upload_file(Filename = S3_url, Bucket= ID_BUCKET, Key= out_S3_url)

    S3_CLIENT.delete_object(Bucket= ID_BUCKET, Key= S3_url)

    result = {
        'category':f'{pred_CAT}'
    }


    return jsonify(result)
 
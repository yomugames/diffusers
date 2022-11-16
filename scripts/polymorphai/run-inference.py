#!/usr/bin/env python
# coding: utf-8

import os
import sys
from IPython.display import clear_output
from IPython.utils import capture
import wget
import time

import fileinput
from subprocess import getoutput
from IPython.display import HTML
import random

import argparse
import boto3


parser = argparse.ArgumentParser(description="Dreambooth test script.")

parser.add_argument(
    "--session",
    type=str,
    help="the session name for dreambooth train + inference"
)

opt = parser.parse_args()

Session_Name = opt.session #@param{type: 'string'}

Session_Name=Session_Name.replace(" ","_")
WORKSPACE='/content/gdrive/MyDrive/Fast-Dreambooth'
SESSION_DIR=WORKSPACE+'/Sessions/'+Session_Name
INF_OUTPUT_DIR=SESSION_DIR+'/output'

INSTANCET=Session_Name
INSTANCET=INSTANCET.replace(" ","_")

path_to_trained_model=SESSION_DIR+"/"+INSTANCET+'.ckpt'

# clean output dir
get_ipython().system('rm -r $INF_OUTPUT_DIR')

prompt='photo of user_xyz person working out in a gym'


def run_inference(prompt, output_dir, path_to_trained_model):
  with capture.capture_output() as cap:
    get_ipython().run_line_magic('cd', '/content/gdrive/MyDrive/sd/stable-diffusion/')
  
  get_ipython().system('python scripts/txt2img.py --prompt "$prompt" --outdir "$output_dir" --ddim_steps 50 --scale 15  --H 512 --W 512 --n_samples 5 --seed 332 --ckpt "$path_to_trained_model"')

def upload_to_s3(session_name, output_dir):
  s3 = boto3.client('s3')
  files = os.listdir(output_dir)
  image_files = [f for f in files if os.path.isfile(output_dir+'/'+f) and f.endswith(".png")]
  for image_file in image_files:
    s3key = session_name + "/output/" + image_file
    s3.upload_file(output_dir + '/' + image_file, "polymorph-ai", s3key)  

run_inference(prompt, INF_OUTPUT_DIR, path_to_trained_model)
upload_to_s3(Session_Name, INF_OUTPUT_DIR)

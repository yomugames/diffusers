#!/usr/bin/env python
# coding: utf-8

import os
import sys
from IPython.display import clear_output
from IPython.utils import capture
import wget
import time
import uuid

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
get_ipython().system('rm -rf $INF_OUTPUT_DIR')

def run_inference(prompt, output_dir, path_to_trained_model):
  with capture.capture_output() as cap:
    get_ipython().run_line_magic('cd', '/content/gdrive/MyDrive/sd/stable-diffusion/')
  
    get_ipython().system('python scripts/txt2img.py --prompt "$prompt" --seed 332 --outdir "$output_dir" --ddim_steps 60 --scale 15  --H 512 --W 512 --n_samples 1 --n_iter 1 --skip_grid --ckpt "$path_to_trained_model"')

def upload_to_s3(session_name, output_dir):
  s3 = boto3.client('s3')
  files = os.listdir(output_dir)
  image_files = [f for f in files if os.path.isfile(output_dir+'/'+f) and f.endswith(".png")]
  for image_file in image_files:
    s3key = session_name + "/output/" + image_file
    s3.upload_file(output_dir + '/' + image_file, "polymorph-ai", s3key)  

#prompt='photo of user_xyz person working out in a gym'
#prompt='Centered fine studio photograph of a user_xyz wearing only a futuristic mecha mayan helmet with bright lights, chest and face, ultra-realistic, white background, 8k hdr, shallow depth of field, intricate'

prompt_instance = 'user_xyz'

f = open("prompts.txt","r")
lines = f.readlines()

for line in lines:
    prompt = line.replace("<instance>",prompt_instance)
    run_inference(prompt, INF_OUTPUT_DIR, path_to_trained_model)
    output_file_path = INF_OUTPUT_DIR + "/samples/00000.png"
    new_file_path = INF_OUTPUT_DIR + "/" + str(uuid.uuid4()) + ".png"

    get_ipython().system("cp " + output_file_path + " " + new_file_path)


upload_to_s3(Session_Name, INF_OUTPUT_DIR)

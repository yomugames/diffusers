#!/usr/bin/env python
# coding: utf-8

import os
import sys
from IPython.display import clear_output
from IPython.utils import capture
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

parser.add_argument(
    "--gender",
    type=str,
    help="the gender of subject"
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

def run_inference(prompt, negative_prompt, output_dir, path_to_trained_model, ddim_steps, cfg_scale):
  with capture.capture_output() as cap:
    get_ipython().run_line_magic('cd', '/content/gdrive/MyDrive/sd/stable-diffusion/')
  
    get_ipython().system('python scripts/txt2img.py --prompt "$prompt" --negative_prompt "$negative_prompt" --seed 332 --outdir "$output_dir" --ddim_steps "$ddim_steps" --scale "$cfg_scale"  --H 512 --W 512 --n_samples 1 --n_iter 3 --skip_grid --ckpt "$path_to_trained_model"')

def rename_files(output_dir):
  files = os.listdir(output_dir)
  image_files = [f for f in files if os.path.isfile(output_dir+'/'+f) and f.endswith(".png")]
  for image_file in image_files:
    old_path = output_dir + "/" + image_file
    new_path = output_dir + "/" + str(uuid.uuid4()) + ".png"
    cmd = "mv " + old_path + " " + new_path
    print(cmd)
    get_ipython().system(cmd)


def upload_to_s3(session_name, output_dir):
  s3 = boto3.client('s3')

  # upload individual
  files = os.listdir(output_dir)
  image_files = [f for f in files if os.path.isfile(output_dir+'/'+f) and f.endswith(".png")]
  for image_file in image_files:
    s3key = "results/" + session_name + "/" + image_file
    s3.upload_file(output_dir + '/' + image_file, "polymorph-ai", s3key)  

  # upload zip
  file_name = "polymorf.zip" 
  zip_file = output_dir + "/" + file_name
  get_ipython().system("cd " + output_dir + " && zip -r  " + file_name + " *.png")
  s3key = "results/" + session_name + "/" + file_name
  s3.upload_file(zip_file, "polymorph-ai", s3key)  


#prompt='photo of user_xyz person working out in a gym'
#prompt='Centered fine studio photograph of a user_xyz wearing only a futuristic mecha mayan helmet with bright lights, chest and face, ultra-realistic, white background, 8k hdr, shallow depth of field, intricate'

prompt_instance = 'user_xyz person'


ddim_configs = [10,25,40,60]
scale_configs = [4,6,8,10,15]

f = open("prompts.txt","r")
lines = f.readlines()
f.close()

outdir = INF_OUTPUT_DIR 

steps = 25
scale = 7

negative_prompt = "bad anatomy, bad proportions, blurry, cloned face, deformed, disfigured, duplicate, extra arms, extra fingers, extra limbs, extra legs, fused fingers, gross proportions, long neck, malformed limbs, missing arms, missing legs, mutated hands, mutation, mutilated, morbid, out of frame, poorly drawn hands, poorly drawn face, too many fingers, ugly"

if opt.gender == "Male":
  negative_prompt += ", female"
elif opt.gender == "Female":
  negative_prompt += ", male"

#for steps in ddim_configs:
#  for scale in scale_configs:
#    outdir = INF_OUTPUT_DIR + '/' + str(steps) + '_ddim_' + str(scale) + '_scale'

for line in lines:
  prompt = line.replace("<instance>",prompt_instance)
  run_inference(prompt, negative_prompt, outdir, path_to_trained_model, steps, scale)

samples_outdir = outdir + "/samples" 
rename_files(samples_outdir)
upload_to_s3(Session_Name, samples_outdir)


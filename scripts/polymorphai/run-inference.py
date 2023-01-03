#!/usr/bin/env python
# coding: utf-8

import os
import os.path
from os import path
import json
import math
import sys
from IPython.display import clear_output
from IPython.utils import capture
import time
import uuid
import urllib.request


import fileinput
from subprocess import getoutput
from IPython.display import HTML
import random
import subprocess
import signal
import traceback

import argparse
import boto3
import re
import requests
import io
import base64
from gradio.processing_utils import encode_pil_to_base64
from PIL import Image
import socket
from contextlib import closing
import mysql.connector

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
INF_OUTPUT_DIR=SESSION_DIR+'/output/'

INSTANCET=Session_Name
INSTANCET=INSTANCET.replace(" ","_")

path_to_trained_model=SESSION_DIR+"/"+INSTANCET+'.ckpt'

# clean output dir
get_ipython().system('rm -rf $INF_OUTPUT_DIR')

webui_host = '127.0.0.1'
webui_port = 7860
automatic_web_url = f'http://{webui_host}:{webui_port}'
webui_proc = ""

image_count = 1

def setup_automatic1111():
  global webui_proc
  get_ipython().run_line_magic('cd', '/content/stable-diffusion-webui')
  if not path.exists(path_to_trained_model):
    print(path_to_trained_model + " missing...downloading from s3 instead")
    ckpt_s3_path = "https://polymorph-ai.s3.amazonaws.com/inputs/" + Session_Name + "/" + Session_Name + ".ckpt"
    urllib.request.urlretrieve(ckpt_s3_path, path_to_trained_model)

  webui_cmd = f'/content/stable-diffusion-webui/webui.sh  --disable-safe-unpickle --share --ckpt {path_to_trained_model} --api --port {webui_port} > /content/stable-diffusion-webui/webui.log 2>&1'
  webui_proc = subprocess.Popen(webui_cmd, shell=True, stdin=subprocess.PIPE, preexec_fn=os.setsid)
  print("setup:" + str(webui_proc))
  get_ipython().run_line_magic('cd', '/content/diffusers/scripts/polymorphai')

def kill_automatic1111():
  print("destroy:" + str(webui_proc))
  os.killpg(os.getpgid(webui_proc.pid), signal.SIGTERM)  # Send the signal to all the process groups


def is_socket_open(host, port):
  with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
    return sock.connect_ex((host, port)) == 0

def run_inference(prompt_instance, prompt, negative_prompt, output_dir, path_to_trained_model, ddim_steps, cfg_scale, n_iter, seed):
  global image_count

  is_img2img_flow = 'img2img_prompt' in prompt and prompt['img2img_prompt'] is not None
  print ("is img2img: " + str(is_img2img_flow))

  actual_prompt = prompt['prompt'].replace("<instance>",prompt_instance)
  payload = {
    "prompt": actual_prompt,
    "negative_prompt": negative_prompt,
    "steps": ddim_steps,
    "cfg_scale": cfg_scale,
    "sampler_index": "DDIM",
    "seed": seed,
    "width": 512,
    "height": 512,
    "n_iter": n_iter,
    "restore_faces": True
  }
  #print ("txt params: " + json.dumps(payload, indent = 2))
  response = requests.post(url=f'{automatic_web_url}/sdapi/v1/txt2img', json=payload)
  json_response = response.json()
  extension = ".png" if not is_img2img_flow else ".tmp.png"
  for i in json_response['images']:
    image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
    image.save(output_dir + '/' + str(image_count) + extension)
    image_count += 1

  if is_img2img_flow:
    # run img2img
    actual_prompt = prompt['img2img_prompt'].replace("<instance>",prompt_instance)
    files = os.listdir(output_dir)
    image_files = [f for f in files if os.path.isfile(output_dir+'/'+f) and f.endswith(".tmp.png")]
    for input_image in image_files:
      with capture.capture_output() as cap:
        full_base64 = encode_pil_to_base64(Image.open(output_dir + '/' + input_image, "r"))

        payload = {
          "init_images": [full_base64],
          "prompt": actual_prompt,
          "negative_prompt": negative_prompt,
          "steps": 35,
          "denoising_strength": 0.28,
          "cfg_scale": cfg_scale,
          "sampler_index": "Euler a",
          "seed": seed,
          "width": 512,
          "height": 512,
          "n_iter": 1,
          "restore_faces": True
        }
        response = requests.post(url=f'{automatic_web_url}/sdapi/v1/img2img', json=payload)
        json_response = response.json()

        for i in json_response['images']:
          output_filename = input_image[:-8] + ".png"
          image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
          image.save(output_dir + '/' + output_filename)
        os.rename(output_dir + '/' + input_image, output_dir + '/tmps/' + input_image)

def rename_files_and_write_metadata(output_dir, prompts, steps, scale, n_iter, seed):
  files = os.listdir(output_dir)
  image_files = [f for f in files if os.path.isfile(output_dir+'/'+f) and f.endswith(".png") and not f.endswith(".tmp.png")]

  metadata = { "steps": steps, "scale": scale, "seed": seed, "prompts": {} }

  for image_file in image_files:
    image_index = int(image_file.replace(".png","")) - 1
    prompt_index = math.floor(image_index / n_iter)
    prompt = prompts[prompt_index]

    new_image_uid = str(uuid.uuid4())

    old_path = output_dir + "/" + image_file
    new_path = output_dir + "/" + new_image_uid + ".png"
    cmd = "mv " + old_path + " " + new_path
    print(cmd)
    get_ipython().system(cmd)
    metadata["prompts"][new_image_uid] = prompt['uid']

  print("saving image uid -> prompt map in metadata.txt")
  metadata_path = output_dir + "/metadata.txt"
  f = open(metadata_path, "w")
  f.write(json.dumps(metadata))
  f.close()


def get_prompts(prompt_instance, gender):
  mydb = mysql.connector.connect(
      host=os.getenv("DB_HOST"),
      user=os.getenv("DB_USER"),
      password=os.getenv("DB_PASS"),
      database=os.getenv("DB_DATABASE")
  )

  mycursor = mydb.cursor(dictionary=True)
  sql = "SELECT * FROM prompts WHERE enabled = '1' and gender = '" + gender + "'"
  mycursor.execute(sql)
  records = mycursor.fetchall()
  mycursor.close()

  result = []
  for record in records:
    result.append(record)

  return result


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

prompts = get_prompts(prompt_instance, opt.gender)


outdir = INF_OUTPUT_DIR

steps = 25
scale = 7
n_iter = 8
seed = -1

negative_prompt = "leg, feet, legs, bad anatomy, bad proportions, blurry, cloned face, deformed, disfigured, duplicate, extra arms, extra fingers, extra limbs, extra legs, fused fingers, gross proportions, long neck, malformed limbs, missing arms, missing legs, mutated hands, mutation, mutilated, morbid, out of frame, poorly drawn hands, poorly drawn face, too many fingers, ugly"

if opt.gender == "Male":
  negative_prompt += ", female"
elif opt.gender == "Female":
  negative_prompt += ", male"

#for steps in ddim_configs:
#  for scale in scale_configs:
#    outdir = INF_OUTPUT_DIR + '/' + str(steps) + '_ddim_' + str(scale) + '_scale'

samples_outdir = outdir + "/samples"
os.makedirs(samples_outdir)
os.makedirs(samples_outdir + "/tmps")

if is_socket_open(webui_host, webui_port):
  print(f"Cannot run program because another program is listening to {webui_port}")
  sys.exit()

try:
  setup_automatic1111()

  while(not is_socket_open(webui_host, webui_port)):
    print("Webui not ready yet, sleeping for 5 secs...")
    time.sleep(5)

  print("Sleep another 15 secs to make sure it's ready")
  time.sleep(15)

  for prompt in prompts:
    run_inference(prompt_instance, prompt, negative_prompt, samples_outdir, path_to_trained_model, steps, scale, n_iter, seed)
  rename_files_and_write_metadata(samples_outdir, prompts, steps, scale, n_iter, seed)

  upload_to_s3(Session_Name, samples_outdir)
except Exception as err:
  traceback.print_exc()
finally:
  kill_automatic1111()
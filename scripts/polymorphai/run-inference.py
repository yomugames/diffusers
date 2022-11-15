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

Session_Name = "" #@param{type: 'string'}
while Session_Name=="":
  print('[1;31mInput the Session Name:') 
  Session_Name=input('')

Session_Name=Session_Name.replace(" ","_")
WORKSPACE='/content/gdrive/MyDrive/Fast-Dreambooth'
SESSION_DIR=WORKSPACE+'/Sessions/'+Session_Name
INF_OUTPUT_DIR=SESSION_DIR+'/output'

print('^[[1;31mIt seems that you did not perform training during this session ^[[1;32mor you chose to use a custom path,\nprovide the full path to the model (including the name of the model):\n')
path_to_trained_model=input()

with capture.capture_output() as cap:
  get_ipython().run_line_magic('cd', '/content/gdrive/MyDrive/sd/stable-diffusion/')

prompt='photo of law_xyz person working out in a gym'


get_ipython().system('python scripts/txt2img.py --prompt "$prompt" --outdir "$INF_OUTPUT_DIR" --ddim_steps 50  --H 512 --W 512 --n_samples 5 --seed 332 --ckpt "$path_to_trained_model"')

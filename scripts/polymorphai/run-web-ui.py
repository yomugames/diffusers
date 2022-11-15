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

Use_Gradio_Server = True #@param {type:"boolean"}
#@markdown  - Only if you have trouble connecting to the local server

print('^[[1;31mIt seems that you did not perform training during this session ^[[1;32mor you chose to use a custom path,\nprovide the full path to the model (including the name of the model):\n')
path_to_trained_model=input()


share=''
if Use_Gradio_Server:
  share='--share'
  for line in fileinput.input('/opt/conda/envs/pytorch/lib/python3.9/site-packages/gradio/blocks.py', inplace=True):
    if line.strip().startswith('self.server_name ='):
        line = '            self.server_name = server_name\n'
    if line.strip().startswith('self.server_port ='):
        line = '            self.server_port = server_port\n'
    if line.strip().startswith('self.protocol = "https"'):
        line = '            self.protocol = "https" if self.local_url.startswith("https") else "http"\n'
    sys.stdout.write(line)
  clear_output()
  
else:
  share=''
  get_ipython().system('nohup lt --port 7860 > srv.txt 2>&1 &')
  time.sleep(2)
  get_ipython().system("grep -o 'https[^ ]*' /content/srv.txt >srvr.txt")
  time.sleep(2)
  srv= getoutput('cat /content/srvr.txt')

  for line in fileinput.input('/opt/conda/envs/pytorch/lib/python3.9/site-packages/gradio/blocks.py', inplace=True):
    if line.strip().startswith('self.server_name ='):
        line = f'            self.server_name = "{srv[8:]}"\n'
    if line.strip().startswith('self.server_port ='):
        line = '            self.server_port = 443\n'
    if line.strip().startswith('self.protocol = "https"'):
        line = '            self.protocol = "https"\n'
    sys.stdout.write(line)

  get_ipython().system('sed -i \'13s@.*@    "PUBLIC_SHARE_TRUE": "\x1b[32mConnected",@\' /opt/conda/envs/pytorch/lib/python3.9/site-packages/gradio/strings.py')
  
  get_ipython().system('rm /content/srv.txt')
  get_ipython().system('rm /content/srvr.txt')
  clear_output()

with capture.capture_output() as cap:
  get_ipython().run_line_magic('cd', '/content/gdrive/MyDrive/sd/stable-diffusion/')

if os.path.isfile(path_to_trained_model):
  get_ipython().system('python /content/gdrive/MyDrive/sd/stable-diffusion-webui/webui.py $share --disable-safe-unpickle --ckpt "$path_to_trained_model"')
else:
  get_ipython().system('python /content/gdrive/MyDrive/sd/stable-diffusion-webui/webui.py $share --disable-safe-unpickle --ckpt-dir "$path_to_trained_model"')

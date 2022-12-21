#!/usr/bin/env python
# coding: utf-8

# # **fast-DreamBooth colab From https://github.com/TheLastBen/fast-stable-diffusion, if you face any issues, feel free to discuss them.** 
# Keep your notebook updated for best experience. [Support](https://ko-fi.com/thelastben)
# 

# In[ ]:


# # Downloading the model

# In[ ]:


import os
import time
import sys
from IPython.display import clear_output
from IPython.utils import capture

#@markdown - Skip this cell if you are loading a previous session

#@markdown ---

with capture.capture_output() as cap: 
  get_ipython().run_line_magic('cd', '/content/')

Update_repo = True #@param {type:"boolean"}
Huggingface_Token = os.environ.get('HUGGINGFACE_TOKEN') or ""
token=Huggingface_Token

#@markdown - Download the original v1.5 model.

#@markdown (Make sure you've accepted the terms in https://huggingface.co/runwayml/stable-diffusion-v1-5)

#@markdown ---

Path_to_HuggingFace= "" #@param {type:"string"}

#@markdown - Load and finetune a model from Hugging Face, use the format "profile/model" like : runwayml/stable-diffusion-v1-5

#@markdown Or

CKPT_Path = "" #@param {type:"string"}

#@markdown Or

CKPT_Link = "" #@param {type:"string"}

#@markdown - A CKPT direct link, huggingface CKPT link or a shared CKPT from gdrive.
#@markdown ---

Compatiblity_Mode="" #@param {type:"boolean"}
#@markdown - Enable only if you're getting conversion errors.


def downloadmodel():
  print("downloadmodel");
  token=Huggingface_Token
  if token=="":
      token=input("Insert your huggingface token :")
  if os.path.exists('/content/stable-diffusion-v1-5'):
    get_ipython().system('rm -r /content/stable-diffusion-v1-5')
  clear_output()

  get_ipython().run_line_magic('cd', '/content/')
  clear_output()
  get_ipython().system('mkdir /content/stable-diffusion-v1-5')
  get_ipython().run_line_magic('cd', '/content/stable-diffusion-v1-5')
  get_ipython().system('git init')
  get_ipython().system('git lfs install --system --skip-repo')
  get_ipython().system('git remote add -f origin  "https://USER:{token}@huggingface.co/runwayml/stable-diffusion-v1-5"')
  get_ipython().system('git config core.sparsecheckout true')
  get_ipython().system('echo -e "feature_extractor\\nsafety_checker\\nscheduler\\ntext_encoder\\ntokenizer\\nunet\\nmodel_index.json" > .git/info/sparse-checkout')
  get_ipython().system('git pull origin main')
  if os.path.exists('/content/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin'):
    get_ipython().system('git clone "https://USER:{token}@huggingface.co/stabilityai/sd-vae-ft-mse"')
    get_ipython().system('mv /content/stable-diffusion-v1-5/sd-vae-ft-mse /content/stable-diffusion-v1-5/vae')
    get_ipython().system('sed -i \'s@"clip_sample": false@@g\' /content/stable-diffusion-v1-5/scheduler/scheduler_config.json')
    get_ipython().system('sed -i \'s@"trained_betas": null,@"trained_betas": null@g\' /content/stable-diffusion-v1-5/scheduler/scheduler_config.json')
    get_ipython().system('sed -i \'s@"sample_size": 256,@"sample_size": 512,@g\' /content/stable-diffusion-v1-5/vae/config.json')
    get_ipython().run_line_magic('cd', '/content/')
    clear_output()
    print('[1;32mDONE !')
  else:
    while not os.path.exists('/content/stable-diffusion-v1-5'):
         print('[1;31mMake sure you accepted the terms in https://huggingface.co/runwayml/stable-diffusion-v1-5')
         time.sleep(5)

downloadmodel()


         
with capture.capture_output() as cap:
    get_ipython().run_line_magic('cd', '/content/gdrive/MyDrive/')
    get_ipython().run_line_magic('mkdir', 'sd')
    get_ipython().run_line_magic('cd', 'sd')
    get_ipython().system('git clone https://github.com/Stability-AI/stablediffusion.git stable-diffusion')
    get_ipython().system('mkdir -p cache/{huggingface,torch}')
    get_ipython().run_line_magic('cd', '/content/')

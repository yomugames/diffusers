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

def downloadmodel_hf():
  print("downloadmodel_hf");
  if os.path.exists('/content/stable-diffusion-v1-5'):
    get_ipython().system('rm -r /content/stable-diffusion-v1-5')
  clear_output()

  get_ipython().run_line_magic('cd', '/content/')
  clear_output()
  get_ipython().system('mkdir /content/stable-diffusion-v1-5')
  get_ipython().run_line_magic('cd', '/content/stable-diffusion-v1-5')
  get_ipython().system('git init')
  get_ipython().system('git lfs install --system --skip-repo')
  get_ipython().system('git remote add -f origin  "https://USER:{token}@huggingface.co/{Path_to_HuggingFace}"')
  get_ipython().system('git config core.sparsecheckout true')
  get_ipython().system('echo -e "feature_extractor\\nsafety_checker\\nscheduler\\ntext_encoder\\ntokenizer\\nunet\\nmodel_index.json" > .git/info/sparse-checkout')
  get_ipython().system('git pull origin main')
  if os.path.exists('/content/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin'):
    get_ipython().system('git clone "https://USER:{token}@huggingface.co/stabilityai/sd-vae-ft-mse"')
    get_ipython().system('mv /content/stable-diffusion-v1-5/sd-vae-ft-mse /content/stable-diffusion-v1-5/vae')
    get_ipython().system('rm -r /content/stable-diffusion-v1-5/.git')
    get_ipython().system('sed -i \'s@"clip_sample": false@@g\' /content/stable-diffusion-v1-5/scheduler/scheduler_config.json')
    get_ipython().system('sed -i \'s@"trained_betas": null,@"trained_betas": null@g\' /content/stable-diffusion-v1-5/scheduler/scheduler_config.json')
    get_ipython().system('sed -i \'s@"sample_size": 256,@"sample_size": 512,@g\' /content/stable-diffusion-v1-5/vae/config.json')
    get_ipython().run_line_magic('cd', '/content/')
    clear_output()
    print('[1;32mDONE !')
  else:
    while not os.path.exists('/content/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin'):
         print('[1;31mCheck the link you provided')
         time.sleep(5)

if Path_to_HuggingFace != "":
  downloadmodel_hf()

elif CKPT_Path !="":
  if os.path.exists('/content/stable-diffusion-v1-5'):
    get_ipython().system('rm -r /content/stable-diffusion-v1-5')
  if os.path.exists(str(CKPT_Path)):
    get_ipython().system('mkdir /content/stable-diffusion-v1-5')
    with capture.capture_output() as cap:
      if Compatiblity_Mode:
        get_ipython().system('wget https://raw.githubusercontent.com/huggingface/diffusers/039958eae55ff0700cfb42a7e72739575ab341f1/scripts/convert_original_stable_diffusion_to_diffusers.py')
        get_ipython().system('python /content/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path "$CKPT_Path" --dump_path /content/stable-diffusion-v1-5')
        get_ipython().system('rm /content/convert_original_stable_diffusion_to_diffusers.py')
      else:           
        get_ipython().system('python /content/diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path "$CKPT_Path" --dump_path /content/stable-diffusion-v1-5')
    if os.path.exists('/content/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin'):
      get_ipython().system('rm /content/v1-inference.yaml')
      clear_output()
      print('[1;32mDONE !')
    else:
      get_ipython().system('rm /content/convert_original_stable_diffusion_to_diffusers.py')
      get_ipython().system('rm /content/v1-inference.yaml')
      get_ipython().system('rm -r /content/stable-diffusion-v1-5')
      while not os.path.exists('/content/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin'):
        print('[1;31mConversion error, Insufficient RAM or corrupt CKPT, use a 4.7GB CKPT instead of 7GB')
        time.sleep(5)
  else:
    while not os.path.exists(str(CKPT_Path)):
       print('[1;31mWrong path, use the colab file explorer to copy the path')
       time.sleep(5)


elif CKPT_Link !="":   
    if os.path.exists('/content/stable-diffusion-v1-5'):
      get_ipython().system('rm -r /content/stable-diffusion-v1-5')
    get_ipython().system('gdown --fuzzy $CKPT_Link -O model.ckpt')
    if os.path.exists('/content/model.ckpt'):
      if os.path.getsize("/content/model.ckpt") > 1810671599:
        get_ipython().system('mkdir /content/stable-diffusion-v1-5')
        with capture.capture_output() as cap: 
          if Compatiblity_Mode:
            get_ipython().system('wget https://raw.githubusercontent.com/huggingface/diffusers/039958eae55ff0700cfb42a7e72739575ab341f1/scripts/convert_original_stable_diffusion_to_diffusers.py')
            get_ipython().system('python /content/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path /content/model.ckpt --dump_path /content/stable-diffusion-v1-5')
            get_ipython().system('rm /content/convert_original_stable_diffusion_to_diffusers.py')
          else:           
            get_ipython().system('python /content/diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path /content/model.ckpt --dump_path /content/stable-diffusion-v1-5')
        if os.path.exists('/content/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin'):
          clear_output()
          print('[1;32mDONE !')
          get_ipython().system('rm /content/v1-inference.yaml')
          get_ipython().system('rm /content/model.ckpt')
        else:
          if os.path.exists('/content/v1-inference.yaml'):
            get_ipython().system('rm /content/v1-inference.yaml')
          get_ipython().system('rm /content/convert_original_stable_diffusion_to_diffusers.py')
          get_ipython().system('rm -r /content/stable-diffusion-v1-5')
          get_ipython().system('rm /content/model.ckpt')
          while not os.path.exists('/content/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin'):
            print('[1;31mConversion error, Insufficient RAM or corrupt CKPT, use a 4.7GB CKPT instead of 7GB')
            time.sleep(5)
      else:
        while os.path.getsize('/content/model.ckpt') < 1810671599:
           print('[1;31mWrong link, check that the link is valid')
           time.sleep(5)
else:
  downloadmodel()


         
with capture.capture_output() as cap:
    get_ipython().run_line_magic('cd', '/content/gdrive/MyDrive/')
    get_ipython().run_line_magic('mkdir', 'sd')
    get_ipython().run_line_magic('cd', 'sd')
    get_ipython().system('git clone https://github.com/Stability-AI/stablediffusion.git stable-diffusion')
    get_ipython().system('git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui')
    get_ipython().run_line_magic('cd', '/content/gdrive/MyDrive/sd/stable-diffusion-webui/')
    get_ipython().system('mkdir -p cache/{huggingface,torch}')
    get_ipython().run_line_magic('cd', '/content/')
    get_ipython().system('ln -s /content/gdrive/MyDrive/sd/stable-diffusion-webui/cache/huggingface ../root/.cache/')
    get_ipython().system('ln -s /content/gdrive/MyDrive/sd/stable-diffusion-webui/cache/torch ../root/.cache/')

if Update_repo:
  get_ipython().system('rm /content/gdrive/MyDrive/sd/stable-diffusion-webui/webui.sh')
  get_ipython().system('rm /content/gdrive/MyDrive/sd/stable-diffusion-webui/modules/paths.py')
  get_ipython().system('rm /content/gdrive/MyDrive/sd/stable-diffusion-webui/webui.py')
  get_ipython().system('rm /content/gdrive/MyDrive/sd/stable-diffusion-webui/modules/ui.py')
  get_ipython().system('rm /content/gdrive/MyDrive/sd/stable-diffusion-webui/style.css')
  get_ipython().run_line_magic('cd', '/content/gdrive/MyDrive/sd/stable-diffusion-webui/')
  clear_output()
  print('[1;32m')
  get_ipython().system('git pull')


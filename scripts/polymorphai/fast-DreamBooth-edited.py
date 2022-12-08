#!/usr/bin/env python
# coding: utf-8

# # **fast-DreamBooth colab From https://github.com/TheLastBen/fast-stable-diffusion, if you face any issues, feel free to discuss them.** 
# Keep your notebook updated for best experience. [Support](https://ko-fi.com/thelastben)
# 


# # Dreambooth

# In[ ]:


import os
import argparse
import sys
from IPython.display import clear_output
from IPython.utils import capture
import requests
import time
import json
import boto3
import numpy as np
import face_recognition
import math


#@markdown #Create/Load a Session

# params
fp16 = True
IMAGES_FOLDER_OPTIONAL="/content/input_images" #@param{type: 'string'}

MODEL_NAME="/content/stable-diffusion-v1-5"
PT="photo of user_xyz person"
CPT="photo of a person"

Captionned_instance_images = False


parser = argparse.ArgumentParser(description="Dreambooth training script.")
parser.add_argument(
    "--session",
    type=str,
    help="the session name for dreambooth train + inference",
    required=True
)

parser.add_argument(
    "--s3images",
    type=str,
    help="the array of images for dreambooth to train"
)

parser.add_argument(
    "--gender",
    type=str,
    choices=['Male','Female', 'No','Both'],
    help="gender of input images",
    required=True
)

opt = parser.parse_args()

Session_Name = opt.session #@param{type: 'string'}
S3_image_list = json.loads(opt.s3images)

Training_Steps=len(S3_image_list) * 150   #2700 #@param{type: 'number'}
Lr_warmup_steps = 0 # int(Training_Steps / 10)
Seed='332' #@param{type: 'string'}

Session_Name=Session_Name.replace(" ","_")

Contains_faces = opt.gender #@param ["No", "Female", "Male", "Both"]


Session_Link_optional = "" #@param{type: 'string'}

#@markdown - Import a session from another gdrive, the shared gdrive link must point to the specific session's folder that contains the trained CKPT.

WORKSPACE='/content/gdrive/MyDrive/Fast-Dreambooth'

if Session_Link_optional !="":
  print('[1;32mDownloading session...')
  with capture.capture_output() as cap:
    get_ipython().run_line_magic('cd', '/content')
  if Session_Link_optional != "":
    if not os.path.exists(str(WORKSPACE+'/Sessions')):
      get_ipython().run_line_magic('mkdir', "-p $WORKSPACE'/Sessions'")
      time.sleep(1)
    get_ipython().run_line_magic('cd', "$WORKSPACE'/Sessions'")
    get_ipython().system('gdown --folder --remaining-ok -O $Session_Name  $Session_Link_optional')
    get_ipython().run_line_magic('cd', '$Session_Name')
    get_ipython().system('rm -r instance_images')
    get_ipython().system('rm -r Regularization_images')
    get_ipython().system('unzip instance_images.zip')
    get_ipython().system('rm instance_images.zip')
    get_ipython().system('mv *.ckpt $Session_Name".ckpt"')
    get_ipython().run_line_magic('cd', '/content')


INSTANCE_NAME=Session_Name
OUTPUT_DIR="/content/models/"+Session_Name
SESSION_DIR=WORKSPACE+'/Sessions/'+Session_Name
INSTANCE_DIR=SESSION_DIR+'/instance_images'
MDLPTH=str(SESSION_DIR+"/"+Session_Name+'.ckpt')
CLASS_DIR=SESSION_DIR+'/Regularization_images'


def reg():
  with capture.capture_output() as cap:
    if Contains_faces!="No":
      if not os.path.exists(str(CLASS_DIR)):
        get_ipython().run_line_magic('mkdir', '-p "$CLASS_DIR"')
      get_ipython().run_line_magic('cd', '$CLASS_DIR')
      get_ipython().system("cp -R /content/Regularization/Women .")
      get_ipython().system("cp -R /content/Regularization/Men .")
      get_ipython().system("cp -R /content/Regularization/Mix .")
      get_ipython().run_line_magic('cd', '/content')

#@markdown - If you're training on a subject with a face or a movie/style that contains faces. (experimental, still needs some tuning) 

if os.path.exists(str(SESSION_DIR)) and not os.path.exists(str(SESSION_DIR+"/"+Session_Name+'.ckpt')):
  print('[1;32mLoading session with no previous model, using the original model or the custom downloaded model')
  reg()
  if not os.path.exists('/content/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin'):
    if os.path.exists('/content/stable-diffusion-v1-5'):
      get_ipython().system("rm -r '/content/stable-diffusion-v1-5'")
    #fdownloadmodel()
  if not os.path.exists('/content/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin'):
    print('[1;31mError downloading the model, make sure you have accepted the terms at https://huggingface.co/runwayml/stable-diffusion-v1-5')
  else:
    print('[1;32mSession Loaded, proceed to uploading instance images')

elif os.path.exists(str(SESSION_DIR+"/"+Session_Name+'.ckpt')):
  print('[1;32mSession found, loading the trained model ...')
  reg()
  get_ipython().run_line_magic('mkdir', '-p "$OUTPUT_DIR"')
  get_ipython().system('python /content/diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path "$MDLPTH" --dump_path "$OUTPUT_DIR" --session_dir "$SESSION_DIR"')
  if os.path.exists(OUTPUT_DIR+'/unet/diffusion_pytorch_model.bin'):
    resume=True    
    get_ipython().system('rm /content/v1-inference.yaml')
    clear_output()
    print('[1;32mSession loaded.')
  else:     
    get_ipython().system('rm /content/v1-inference.yaml')
    if not os.path.exists(OUTPUT_DIR+'/unet/diffusion_pytorch_model.bin'):
      print('[1;31mConversion error, if the error persists, remove the CKPT file from the current session folder')


elif not os.path.exists(str(SESSION_DIR)):
    get_ipython().run_line_magic('mkdir', '-p "$INSTANCE_DIR"')
    print('[1;32mCreating session...')
    reg()
    if not os.path.exists('/content/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin'):
      if os.path.exists('/content/stable-diffusion-v1-5'):
        get_ipython().system("rm -r '/content/stable-diffusion-v1-5'")
      #fdownloadmodel()
    if os.path.exists('/content/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin'):
      print('[1;32mSession created, proceed to uploading instance images')
    else:
      print('[1;31mError downloading the model, make sure you have accepted the terms at https://huggingface.co/runwayml/stable-diffusion-v1-5')  
    
if Contains_faces == "Female":
  CLASS_DIR=CLASS_DIR+'/Mix'
if Contains_faces == "Male":
  CLASS_DIR=CLASS_DIR+'/Mix'
if Contains_faces == "Both":
  CLASS_DIR=CLASS_DIR+'/Mix'

    #@markdown 

    #@markdown # The most importent step is to rename the instance pictures of each subject to a unique unknown identifier, example :
    #@markdown - If you have 30 pictures of yourself, simply select them all and rename only one to the chosen identifier for example : phtmejhn, the files would be : phtmejhn (1).jpg, phtmejhn (2).png ....etc then upload them, do the same for other people or objects with a different identifier, and that's it.
    #@markdown - Check out this example : https://i.imgur.com/d2lD3rz.jpeg


# In[ ]:

import shutil
from PIL import Image, ImageOps

#@markdown #Instance Images
#@markdown ----

#@markdown
#@markdown - Run the cell to Upload the instance pictures.

Remove_existing_instance_images= True #@param{type: 'boolean'}
#@markdown - Uncheck the box to keep the existing instance images.


def download_images(s3_image_list, dest_dir):
  get_ipython().system('rm -r "$IMAGES_FOLDER_OPTIONAL"')
  get_ipython().system('mkdir -p "$IMAGES_FOLDER_OPTIONAL"')
  
  for url in s3_image_list:
    filename = url
    if url.find('/'):
      filename = url.rsplit('/', 1)[1]
    r = requests.get(url, allow_redirects=True)

    open(dest_dir + '/' + filename, 'wb').write(r.content)

def get_face_center(file):
    image = file.convert('RGB')
    np_image_file = np.array(image)
    face_locations = face_recognition.face_locations(np_image_file)
    print(face_locations)
    if not face_locations:
        return [256, 256]

    top, right, bottom, left =  face_locations[0]
    y = (bottom + top) / 2
    x = (left + right) / 2
    return [x, y]

def min_fit_resize(file, Crop_size):
    width, height = file.size
    min_side_length = min(width, height)

    new_resized_height = 0
    new_resized_width = 0

    # min-fit resize
    if min_side_length == width:
        # width is smaller
        scale = width / Crop_size
        new_resized_width = Crop_size
        new_resized_height = math.floor(height / scale)
    else:
        # height is smaller
        scale = height / Crop_size
        new_resized_height = Crop_size
        new_resized_width = math.floor(width / scale)

    file = file.resize((new_resized_width, new_resized_height))
    return [file, new_resized_width, new_resized_height]

def crop_center(file, facex, facey, new_resized_width, new_resized_height, Crop_size):
    # if facex centering exceed left, anchor left side
    # if facex centering exceeds right, anchor right side
    # else, anchor center
    if facex - (Crop_size / 2) < 0:
        left = 0
        right = Crop_size
    elif facex + (Crop_size / 2) > new_resized_width:
        right = new_resized_width
        left = new_resized_width - Crop_size
    else:
        left = facex - (Crop_size / 2)
        right = facex + (Crop_size / 2)

    # if facey centering exceed top, anchor top side
    # if facey centering exceeds bottom, anchor bottom side
    # else, anchor center
    if facey - (Crop_size / 2) < 0:
        top = 0
        bottom = Crop_size
    elif facey + (Crop_size / 2) > new_resized_height:
        bottom = new_resized_height
        top = new_resized_height - Crop_size
    else:
        top = facey - (Crop_size / 2)
        bottom = facey + (Crop_size / 2)

    return file.crop((left, top, right, bottom))


# get images of faces somewhere, then copy it to instance dir
def prepare_images():
  if Remove_existing_instance_images:
    if os.path.exists(str(INSTANCE_DIR)):
      get_ipython().system('rm -r "$INSTANCE_DIR"')

  if not os.path.exists(str(INSTANCE_DIR)):
    get_ipython().run_line_magic('mkdir', '-p "$INSTANCE_DIR"')


  #@markdown - If you prefer to specify directly the folder of the pictures instead of uploading, this will add the pictures to the existing (if any) instance images. Leave EMPTY to upload.

  Crop_images= True #@param{type: 'boolean'}
  Crop_size=512 #@param{type: 'number'}

  #@markdown - Unless you want to crop them manually in a precise way, you don't need to crop your instance images externally.
  
  global IMAGES_FOLDER_OPTIONAL

  while IMAGES_FOLDER_OPTIONAL !="" and not os.path.exists(str(IMAGES_FOLDER_OPTIONAL)):
    print('[1;31mThe image folder specified does not exist, use the colab file explorer to copy the path :')
    IMAGES_FOLDER_OPTIONAL=input('')

  if IMAGES_FOLDER_OPTIONAL!="":
    with capture.capture_output() as cap:
      if Crop_images:
        for filename in os.listdir(IMAGES_FOLDER_OPTIONAL):
          extension = filename.split(".")[1]
          identifier=filename.split(".")[0]
          new_path_with_file = os.path.join(IMAGES_FOLDER_OPTIONAL, filename)
          file = Image.open(new_path_with_file)
          file = ImageOps.exif_transpose(file) #prevents accidental image rotation
          file, new_resized_width, new_resized_height = min_fit_resize(file, Crop_size)
          facex, facey = get_face_center(file)
          file = crop_center(file, facex, facey, new_resized_width, new_resized_height, Crop_size)

          if (extension.upper() == "JPG"):
              file.save(new_path_with_file, format="JPEG", quality = 100)
          else:
              file.save(new_path_with_file, format=extension.upper())
          get_ipython().run_line_magic('cp', '-r "$IMAGES_FOLDER_OPTIONAL/." "$INSTANCE_DIR"')
          clear_output()      
      else:
        get_ipython().run_line_magic('cp', '-r "$IMAGES_FOLDER_OPTIONAL/." "$INSTANCE_DIR"')

      get_ipython().run_line_magic('cd', '"$INSTANCE_DIR"')
      get_ipython().system('find . -name "* *" -type f | rename \'s/ /_/g\'')
      get_ipython().run_line_magic('cd', '/content')
      if os.path.exists(INSTANCE_DIR+"/.ipynb_checkpoints"):
        get_ipython().run_line_magic('rm', '-r INSTANCE_DIR+"/.ipynb_checkpoints"')
    print('[1;32mDone, proceed to the training cell')

  with capture.capture_output() as cap:
    get_ipython().run_line_magic('cd', '$SESSION_DIR')
    get_ipython().system('rm instance_images.zip')
    get_ipython().system('zip -r instance_images instance_images')
    get_ipython().run_line_magic('cd', '/content')

download_images(S3_image_list, IMAGES_FOLDER_OPTIONAL)
prepare_images()

# # Training

# In[ ]:


#@markdown ---
#@markdown #Start DreamBooth
#@markdown ---
import os
from subprocess import getoutput
from IPython.display import HTML
from IPython.display import clear_output
import random

Resume_Training = False #@param {type:"boolean"}

if not Resume_Training and not os.path.exists('/content/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin'):
  sys.exit("Model not found, exiting...")  

#@markdown  - If you're not satisfied with the result, check this box, run again the cell and it will continue training the current model.

MODELT_NAME=MODEL_NAME

#@markdown - Total Steps = Number of Instance images * 200, if you use 30 images, use 6000 steps, if you're not satisfied with the result, resume training for another 500 steps, and so on ...


if Seed =='' or Seed=='0':
  Seed=random.randint(1, 999999)
else:
  Seed=int(Seed)

#@markdown - Leave empty for a random seed.

if fp16:
  prec="fp16"
else:
  prec="no"

s = getoutput('nvidia-smi')
if 'A100' in s:
  precision="no"
else:
  precision=prec

try:
   resume = false
   if resume and not Resume_Training:
     print('[1;31mOverwrite your previously trained model ?, answering "yes" will train a new model, answering "no" will resume the training of the previous model?  yes or no ?[0m')
     while True:
        ansres=input('')
        if ansres=='no':
          Resume_Training = True
          del ansres
          break
        elif ansres=='yes':
          Resume_Training = False
          resume= False
          break
except:
  pass

if Resume_Training and os.path.exists(OUTPUT_DIR+'/unet/diffusion_pytorch_model.bin'):
  MODELT_NAME=OUTPUT_DIR
  print('[1;32mResuming Training...[0m')
elif Resume_Training and not os.path.exists(OUTPUT_DIR+'/unet/diffusion_pytorch_model.bin'):
  print('[1;31mPrevious model not found, training a new model...[0m') 
  MODELT_NAME=MODEL_NAME

#@markdown ---------------------------

try:
   Contain_f
   pass
except:
   Contain_f=Contains_faces

Enable_text_encoder_training= True #@param{type: 'boolean'}

#@markdown - At least 10% of the total training steps are needed, it doesn't matter if they are at the beginning or in the middle or the end, in case you're training the model multiple times.
#@markdown - For example you can devide 5%, 5%, 5% on 3 training runs on the model, or 0%, 0%, 15%, given that 15% will cover the total training steps count (15% of 200 steps is not enough).

#@markdown - Enter the % of the total steps for which to train the text_encoder
Train_text_encoder_for=100 #@param{type: 'number'}

#@markdown - Keep the % low for better style transfer, more training steps will be necessary for good results.
#@markdown - Higher % will give more weight to the instance, it gives stronger results at lower steps count, but harder to stylize, 

if Train_text_encoder_for>=100:
  stptxt=Training_Steps
elif Train_text_encoder_for==0:
  Enable_text_encoder_training= False
  stptxt=10
else:
  stptxt=int((Training_Steps*Train_text_encoder_for)/100)

if not Enable_text_encoder_training:
  Contains_faces="No"
else:
   Contains_faces=Contain_f

if Enable_text_encoder_training:
  Textenc="--train_text_encoder"
else:
  Textenc=""

#@markdown ---------------------------
Save_Checkpoint_Every_n_Steps = False #@param {type:"boolean"}
Save_Checkpoint_Every=500 #@param{type: 'number'}
if Save_Checkpoint_Every==None:
  Save_Checkpoint_Every=1
#@markdown - Minimum 200 steps between each save.
stp=0
Start_saving_from_the_step=500 #@param{type: 'number'}
if Start_saving_from_the_step==None:
  Start_saving_from_the_step=0
if (Start_saving_from_the_step < 200):
  Start_saving_from_the_step=Save_Checkpoint_Every
stpsv=Start_saving_from_the_step
if Save_Checkpoint_Every_n_Steps:
  stp=Save_Checkpoint_Every
#@markdown - Start saving intermediary checkpoints from this step.

Caption=''
if Captionned_instance_images:
  Caption='--image_captions_filename'


def txtenc_train(Caption, stpsv, stp, MODELT_NAME, INSTANCE_DIR, CLASS_DIR, OUTPUT_DIR, PT, Seed, precision, Training_Steps):
  _, _, files = next(os.walk(CLASS_DIR))
  class_count = len(files)
  print('[1;33mTraining the text encoder with regularization using the model (' + MODEL_NAME + ')...[0m')
  get_ipython().system('accelerate launch --num_cpu_threads_per_process 4 /content/diffusers/examples/dreambooth/train_dreambooth.py      $Caption    --train_text_encoder         --pretrained_model_name_or_path="$MODEL_NAME"  --instance_data_dir="$INSTANCE_DIR"      --class_data_dir="$CLASS_DIR"      --output_dir="$OUTPUT_DIR"      --with_prior_preservation --prior_loss_weight=1.0      --instance_prompt="$PT" --class_prompt="$CPT"    --seed=$Seed      --resolution=512      --mixed_precision=$precision      --train_batch_size=2      --gradient_accumulation_steps=1 --gradient_checkpointing      --use_8bit_adam      --learning_rate=1e-6      --lr_scheduler="constant"      --lr_warmup_steps="$Lr_warmup_steps"      --max_train_steps=$Training_Steps      --num_class_images=$class_count')

def unet_train(Caption, SESSION_DIR, stpsv, stp, MODELT_NAME, INSTANCE_DIR, OUTPUT_DIR, PT, Seed, precision, Training_Steps):
  clear_output()
  print('[1;33mTraining the unet using the model (' + MODELT_NAME + ')...[0m')
  get_ipython().system('accelerate launch /content/diffusers/examples/dreambooth/train_dreambooth.py      $Caption      --train_only_unet     --pretrained_model_name_or_path="$MODELT_NAME"      --instance_data_dir="$INSTANCE_DIR"      --output_dir="$OUTPUT_DIR"      --instance_prompt="$PT"      --seed=$Seed      --resolution=512      --mixed_precision=$precision      --train_batch_size=1      --gradient_accumulation_steps=1    --use_8bit_adam      --learning_rate=3e-6      --lr_scheduler="polynomial"      --lr_warmup_steps=0      --max_train_steps=$Training_Steps')

if Contains_faces!="No":
  
  txtenc_train(Caption, stpsv, stp, MODELT_NAME, INSTANCE_DIR, CLASS_DIR, OUTPUT_DIR, PT, Seed, precision, Training_Steps=stptxt)
  #unet_train(Caption, SESSION_DIR, stpsv, stp, MODELT_NAME, INSTANCE_DIR, OUTPUT_DIR, PT, Seed, precision, Training_Steps)

else:
  get_ipython().system('accelerate launch /content/diffusers/examples/dreambooth/train_dreambooth.py      $Caption      $Textenc      --save_starting_step=$stpsv      --stop_text_encoder_training=$stptxt      --save_n_steps=$stp      --Session_dir=$SESSION_DIR      --pretrained_model_name_or_path="$MODELT_NAME"      --instance_data_dir="$INSTANCE_DIR"      --output_dir="$OUTPUT_DIR"      --instance_prompt="$PT"      --seed=$Seed      --resolution=512      --mixed_precision=$precision      --train_batch_size=1      --gradient_accumulation_steps=1      --use_8bit_adam      --learning_rate=3e-6      --lr_scheduler="polynomial"      --lr_warmup_steps=0      --max_train_steps=$Training_Steps')


if os.path.exists('/content/models/'+INSTANCE_NAME+'/unet/diffusion_pytorch_model.bin'):
  print("Almost done ...")
  get_ipython().run_line_magic('cd', '/content')
  get_ipython().system('wget -O convertosd.py https://github.com/TheLastBen/fast-stable-diffusion/raw/main/Dreambooth/convertosd.py')
  clear_output()
  if precision=="no":
    get_ipython().system("sed -i '226s@.*@@' /content/convertosd.py")
  get_ipython().system('sed -i \'201s@.*@    model_path = "{OUTPUT_DIR}"@\' /content/convertosd.py')
  get_ipython().system('sed -i \'202s@.*@    checkpoint_path= "{SESSION_DIR}/{Session_Name}.ckpt"@\' /content/convertosd.py')
  get_ipython().system('python /content/convertosd.py')
  clear_output()
  if os.path.exists(SESSION_DIR+"/"+INSTANCE_NAME+'.ckpt'):
    if not os.path.exists(str(SESSION_DIR+'/tokenizer')):
      get_ipython().system('cp -R \'/content/models/\'$INSTANCE_NAME\'/tokenizer\' "$SESSION_DIR"')
    print("[1;32mDONE, the CKPT model is in your Gdrive in the sessions folder")

    #upload model to s3
    s3 = boto3.client('s3')
    s3key = "inputs/" + INSTANCE_NAME + "/" + INSTANCE_NAME + ".ckpt"
    s3.upload_file(MDLPTH, "polymorph-ai", s3key, ExtraArgs={'ACL': 'private'})
    print("Uploaded the CKPT model to S3")

  else:
    print("[1;31mSomething went wrong")
    
else:
  print("[1;31mSomething went wrong")





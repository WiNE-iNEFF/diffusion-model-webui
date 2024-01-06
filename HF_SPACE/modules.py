from huggingface_hub import HfApi, ModelCard, create_repo, get_full_repo_name
from diffusers import DDIMScheduler, DDIMPipeline
from torchvision import transforms
from huggingface_hub import login
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
from modules import *
import torchvision.datasets as datasets
import torch.nn.functional as F
import gradio as gr
import numpy as np
import torchvision
import accelerate
import torch
import wandb
import time
import re
import os


if os.getenv("COLAB_RELEASE_TAG"): RunningInCOLAB = True
else: RunningInCOLAB = False

not_wandb = False
hf_success = False

def future_update():
  update = gr.Markdown("""
                      # This page can't work
                      Now this page can't work. Please wait future update.
                      """)
  return update

def image_crop(image):
  width, height = image.size
  save = []
  for w in range(width//(width//2)):
    for h in range(height//(height//2)):
      box = (w*(width//2), h*(height//2), (w+1)*(width//2), (h+1)*(height//2))
      save.append(image.crop(box))
  return save

def pushhub(model, model_name, model_description):
  model.save_pretrained("my_pipeline")
  hub_model_id = get_full_repo_name(model_name)
  try: create_repo(hub_model_id)
  except: pass
  api = HfApi()
  api.upload_folder(folder_path="my_pipeline/scheduler", path_in_repo="scheduler/", repo_id=hub_model_id)
  api.upload_folder(folder_path="my_pipeline/unet", path_in_repo="unet/", repo_id=hub_model_id)
  api.upload_file(path_or_fileobj="my_pipeline/model_index.json", path_in_repo="model_index.json",repo_id=hub_model_id)
  card = ModelCard(model_description)
  card.push_to_hub(hub_model_id)

def show_images(x):
  grid = torchvision.utils.make_grid(x, nrow=2, padding=0)
  grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
  grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
  return grid_im

def image_generate(ch_num, scheduler, pretrained_pipeline, image_size):
  image_size = tuple(map(int, image_size.replace('(', '').replace(')', '').split(',')))
  x = torch.randn(4, ch_num, *image_size).to(device)
  for i, t in tqdm(enumerate(scheduler.timesteps)):
    model_input = scheduler.scale_model_input(x, t)
    with torch.no_grad():
      noise_pred = pretrained_pipeline.unet(model_input, t)["sample"]
    x = scheduler.step(noise_pred, t, x).prev_sample
  return show_images(x)

def hf_login(huggingface_write_token):
  global hf_success
  if not huggingface_write_token: raise gr.Error('Please write Huggingface access token with write permission')
  else:
    try:
      gr.Info('Try login in HuggingFace')
      if RunningInCOLAB == True:
        from google.colab import userdata
        login(token=userdata.get(huggingface_secret_token), add_to_git_credential=True)
      else: login(token=huggingface_write_token, add_to_git_credential=True)
      time.sleep(5)
      hf_success = True
      gr.Info('Login success')
    except Exception as e: gr.Error(e)

def wandb_setup(wandb_write_token, wandb_project_name, wandb_run_name):
  global not_wandb
  if hf_success == False: raise gr.Warning('Please login in HuggingFace before starting other code')

  if not wandb_write_token:
    not_wandb = True
    raise gr.Error('Please write WanDB api token')

  else:
    gr.Info('Start setup WanDB')
    if RunningInCOLAB:
      from google.colab import userdata
      wandb.login(key=userdata.get(wandb_secret_token))
    else: wandb.login(key=wandb_write_token)

    if not wandb_project_name: raise gr.Error('Please write wandb project name')
    elif not wandb_run_name:
      gr.Info("Wandb run name not choose. Using random name")
      wandb.init(project=wandb_project_name)
    else: wandb.init(project=wandb_project_name, name=wandb_run_name)
    time.sleep(5)
    gr.Info('WanDB setup success')

def start_setup(pretrained_model, dataset_name, image_size, batch_size, color_value, epoch, grad_accumulation_steps, save_to_hub, image_example, model_name, model_description):
  if hf_success == False: raise gr.Warning('Please login in HuggingFace before starting other code')

  if not re.search(r'^[A-Za-z0-9/-]+/[A-Za-z0-9/-]+$', pretrained_model): raise gr.Error('Please write link of pretrained model')
  else:
    gr.Info("Try to load pretrained model")
    try: pretrained_pipeline = DDIMPipeline.from_pretrained(pretrained_model).to(device)
    except Exception as e: raise gr.Error(e)
  time.sleep(5)

  if not re.search(r'^[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$', dataset_name): raise gr.Error('Please write link of dataset')
  else:
    gr.Info("Try to load dataset")
    try: dataset = load_dataset(dataset_name, split="train")
    except Exception as e: raise gr.Error(e)
  time.sleep(5)

  if not re.search(r'^\(\d+,\s*\d+\)$', image_size): raise gr.Error('Please write image size: or check correct form of "image size" value')
  else:
    try: preprocess = transforms.Compose([transforms.Resize(eval(image_size)), transforms.ToTensor(),])
    except Exception as e: raise gr.Error(e)

  if not batch_size or batch_size <= 0: raise gr.Error('Please write batch size or check correct form of "batch size" value') #re.search(r'^\d+$', batch_size)
  time.sleep(5)

  if not color_value: raise gr.Error('Please write what type of image use: RGBA, RGB')
  else:
    try:
      def transform(examples):
        images = [preprocess(image.convert(color_value)) for image in examples["image"]]
        return {"images": images}
      dataset.set_transform(transform)
      ch_num = (3 if color_value == 'RGB' else 4)
      train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    except Exception as e: raise gr.Error(e)
  time.sleep(5)

  gr.Info('Load training setting')
  if epoch <= 0 or not epoch: raise gr.Error('Please write count of training epoch')
  if grad_accumulation_steps < 0 or not grad_accumulation_steps: raise gr.Error('Please write count of gradient accumulation step: Minimum = 0')
  if save_to_hub < 0 or not save_to_hub: raise gr.Error('Please write what number of epoch need to push checkpoint in hub: Minimum = 0')
  if image_example < 0 or not image_example: raise gr.Error('Please write what number of epoch need to save image example: Minimum = 0')
  if not model_name: raise gr.Error('Please write model name to push in hub')
  if not model_description:
    model_description = f"""
    ---
    license: mit
    tags:
    - pytorch
    - diffusers
    - unconditional-image-generation
    - diffusion-model
    ---
    """

  try: scheduler = DDIMScheduler.from_pretrained(pretrained_model)
  except: scheduler = DDIMScheduler.from_pretrained(pretrained_model, subfolder='scheduler')
  scheduler.set_timesteps(num_inference_steps=35)
  lr = 1e-5
  optimizer = torch.optim.AdamW(pretrained_pipeline.unet.parameters(), lr=lr)
  losses = []

  gr.Info('Start training')
  for epoch in range(epoch):
    gr.Info(f'Epoch: {epoch}')
    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
      clean_images = batch['images'].to(device)
      #clean_images = batch[0].to(device)
      noise = torch.randn(clean_images.shape).to(clean_images.device)
      bs = clean_images.shape[0]

      timesteps = torch.randint(0, pretrained_pipeline.scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()
      noisy_images = pretrained_pipeline.scheduler.add_noise(clean_images, noise, timesteps)
      noise_pred = pretrained_pipeline.unet(noisy_images, timesteps, return_dict=False)[0]
      loss = F.mse_loss(noise_pred, noise)
      losses.append(loss.item())
      loss.backward(loss)

      if (step + 1) % grad_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

    gr.Info(f'average loss: {sum(losses[-len(train_dataloader):])/len(train_dataloader)}')

    if epoch % save_to_hub == 0:
        try: pushhub(pretrained_pipeline, model_name, model_description)
        except Exception as e: raise gr.Error(e)

    if not_wandb == False:
      wandb.log({'Epoch loss':sum(losses[-len(train_dataloader):])/len(train_dataloader)})
      if epoch % image_example == 0:
        wandb.log({'Sample generations': wandb.Image(image_generate(ch_num, scheduler, pretrained_pipeline, image_size))})

  return image_crop(image_generate(ch_num, scheduler, pretrained_pipeline, image_size))
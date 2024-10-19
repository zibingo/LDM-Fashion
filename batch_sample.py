import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from torchvision.utils import save_image
from tqdm import tqdm
import os



def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

# Placeholder for your neural network (replace with your actual model)
def load_neural_network():
    model = load_model_from_config(config, ckpt_path)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = DDIMSampler(model)
    return sampler

        
def sample(sketch ,texture, scale_1, scale_2,ddim_steps, ddim_eta=0.0):
    f = 8 # first stage downsampling factor
    model=sampler.model
    c = {
        "sketch":sketch,
        "texture":texture
    }
    cond = model.get_learned_conditioning(c)
    cond = {"cond":cond}
    unconditional_guidance_scale = {
        "scale_1":scale_1,
        "scale_2":scale_2,
    }
    unconditional_conditioning = {
        "two_zero":model.get_learned_conditioning(
        {
            "sketch":torch.zeros_like(c["sketch"],device=c["sketch"].device),
            "texture":torch.zeros_like(c["texture"],device=c["texture"].device)
        }),
        "one_zero":model.get_learned_conditioning(
        {
            "sketch":torch.zeros_like(c["sketch"],device=c["sketch"].device),
            "texture":c["texture"]
        }
    )}

    shape = [4, 256//f, 256//f]

    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                     conditioning=cond,
                                     batch_size=1,
                                     shape=shape,
                                     verbose=False,
                                     unconditional_guidance_scale=unconditional_guidance_scale,
                                     unconditional_conditioning=unconditional_conditioning,
                                     eta=ddim_eta
                                     )

    x_samples_ddim = model.decode_first_stage(samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
    
    return x_samples_ddim

def transform_img(img):
    img = img.resize((256,256))
    img = np.array(img).astype(np.uint8)
    # 如果是单通道图像，则添加一个批次维度
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
    img = np.transpose(img, (2, 0, 1))
    img = (img/127.5 - 1.0).astype(np.float32)

    return torch.from_numpy(img[np.newaxis, ...]).cuda()

# Your image processing and neural network inference logic
def generate_images(sketch_img_name, texture_img_name, scale_1, scale_2, ddim_steps):

    sketch_img_path = os.path.join(sample_dataset_path, "sketch/", sketch_img_name+".png")
    texture_img_path = os.path.join(sample_dataset_path, "texture/", texture_img_name+".jpg")


    sketch_img = transform_img(Image.open(sketch_img_path)).cuda()
    texture_img = transform_img(Image.open(texture_img_path)).cuda()
    predictions = sample(sketch_img, texture_img,scale_1,scale_2, ddim_steps)

    return predictions

if __name__ == "__main__":
    config = OmegaConf.load("configs/latent-diffusion/mydata_ldm-vq-f8-test.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    ckpt_path = "logs/2024-06-09T08-45-27_mydata_ldm-vq-f8/checkpoints/add_style_loss_epoch=000034.ckpt"
    
    sampler = load_neural_network()
    
    sample_dataset_path = "mydataset1/test"
    # 指定目录路径
    generated_images_folder = "gen_data"
    if not os.path.exists(generated_images_folder):
        os.makedirs(generated_images_folder)
    if not os.path.exists("small_gt"):
        os.makedirs("small_gt")

    # 去除文件扩展名并打印文件名列表
    for file in tqdm(os.listdir(os.path.join(sample_dataset_path, "gt"))):
        filename, _ = os.path.splitext(file)
        gen = generate_images(filename,filename,
                        config["model"]["params"]["scale_1"],
                        config["model"]["params"]["scale_2"],
                        ddim_steps=200)
        save_image(gen, os.path.join(generated_images_folder, filename+".jpg"))
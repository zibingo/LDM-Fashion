import os
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image


class MyDataSetTrain(Dataset):
    def __init__(self, size, path):
        super().__init__()
        gt_path=os.path.join(path,"training","gt")
        sketch_path=os.path.join(path,"training","sketch")
        texture_path=os.path.join(path,"training","texture")

        #images path contains subfolders
        gt_images=os.listdir(gt_path)
    
        self.gt_images = [os.path.join(gt_path, img) for img in gt_images]
        self.sketch_images = [os.path.join(sketch_path, img) for img in gt_images]
        self.texture_images = [os.path.join(texture_path, img) for img in gt_images]

        assert len(self.gt_images) == len(self.sketch_images) == len(self.texture_images)

    def __len__(self):
        return len(self.gt_images)
    def check_pair(self,a,b,c):
        # 检查文件名是否一致
        a = a.split('/')[-1].split('.')[0]
        b = b.split('/')[-1].split('.')[0]
        c = c.split('/')[-1].split('.')[0]
        assert a == b == c
    def __getitem__(self, i):

        self.check_pair(self.gt_images[i],self.sketch_images[i],self.texture_images[i])

        gt_image = Image.open(self.gt_images[i])
        gt_image = np.array(gt_image).astype(np.uint8)
        gt_image = (gt_image/127.5 - 1.0).astype(np.float32)

        sketch_image = Image.open(self.sketch_images[i])
        sketch_image = np.array(sketch_image).astype(np.uint8)
        sketch_image = (sketch_image/127.5 - 1.0).astype(np.float32)

        texture_image = Image.open(self.texture_images[i])
        texture_image = np.array(texture_image).astype(np.uint8)
        texture_image = (texture_image/127.5 - 1.0).astype(np.float32)

        return {
            "gt":gt_image, 
            "sketch":sketch_image,
            "texture":texture_image
            }

class MyDataSetValidation(Dataset):
    def __init__(self, size, path):
        super().__init__()
        gt_path=os.path.join(path,"validation","gt")
        sketch_path=os.path.join(path,"validation","sketch")
        texture_path=os.path.join(path,"validation","texture")

        #images path contains subfolders
        gt_images=os.listdir(gt_path)

        self.gt_images = [os.path.join(gt_path, img) for img in gt_images]
        self.sketch_images = [os.path.join(sketch_path, img) for img in gt_images]
        self.texture_images = [os.path.join(texture_path, img) for img in gt_images]

        
        assert len(self.gt_images) == len(self.sketch_images) == len(self.texture_images)

    def __len__(self):
        return len(self.gt_images)
    def check_pair(self,a,b,c):
        # 检查文件名是否一致
        a = a.split('/')[-1].split('.')[0]
        b = b.split('/')[-1].split('.')[0]
        c = c.split('/')[-1].split('.')[0]
        assert a == b == c
    def __getitem__(self, i):

        self.check_pair(self.gt_images[i],self.sketch_images[i],self.texture_images[i])

        gt_image = Image.open(self.gt_images[i])
        gt_image = np.array(gt_image).astype(np.uint8)
        gt_image = (gt_image/127.5 - 1.0).astype(np.float32)

        sketch_image = Image.open(self.sketch_images[i])
        sketch_image = np.array(sketch_image).astype(np.uint8)
        sketch_image = (sketch_image/127.5 - 1.0).astype(np.float32)

        texture_image = Image.open(self.texture_images[i])
        texture_image = np.array(texture_image).astype(np.uint8)
        texture_image = (texture_image/127.5 - 1.0).astype(np.float32)

        return {
            "gt":gt_image, 
            "sketch":sketch_image,
            "texture":texture_image
            }


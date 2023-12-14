import torch
import json
import os
from imageio import imread
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
from timm.data import create_transform
from torch.utils.data import Dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def img_normalize(image):
    if len(image.shape)==2:
        channel = (image[:, :, np.newaxis] - 0.485) / 0.229
        image = np.concatenate([channel,channel,channel], axis=2)
    else:
        image = (image-np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3)))\
                /np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
    return image
class Imgtrain(Dataset):
    def __init__(self):
        path = "../../../DataSet/ImageNet/"  #
        #path = "/home/huatianyu/ImageNet/"#
        #path = "/data0/Data/ImageNet/"
        with open("../label.json") as f:
            self.label = json.load(f)
        self.img = []
        for folder in list(self.label.keys()):
            file = os.listdir(os.path.join(path, "train", folder))
            for i in file:
                self.img.append(os.path.join(path, "train", folder, i))
        self.transform = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,)
    def __len__(self):
        return len(self.img)
    def __getitem__(self, item):
        img = self.transform(Image.open(self.img[item]).convert("RGB"))
        label = self.label[self.img[item].split("/")[-1].split("_")[0]]
        return {"img": img, "label":label}

class ImgVal(Dataset):
    def __init__(self, path = "../../../DataSet/ImageNet/"):
        with open("../label.json") as f:
            self.label = json.load(f)
        self.img = []
        self.index = list(np.arange(1000))
        for folder in list(self.label.keys()):
            file = os.listdir(os.path.join(path, "val", folder))
            for i in file:
                self.img.append(os.path.join(path, "val", folder, i))
        self.transform = transforms.Compose([transforms.Resize((256,256),interpolation=Image.BICUBIC),
                                             transforms.CenterCrop((224,224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
    def __len__(self):
        return len(self.img)
    def __getitem__(self, item):
        img = Image.open(self.img[item]).convert('RGB')
        img = self.transform(img)
        label = self.label[self.img[item].split("/")[-2]]
        return {"img": img, "label": label}


class ImgA(Dataset):
    def __init__(self):
        path = "/mnt/data1/rsc/DataSet/ImageNet-A"
        with open("../label.json") as f:
            self.label = json.load(f)
        self.img = []
        folders = sorted(os.listdir(path))
        self.index = []
        for folder in folders:
            self.index.append(self.label[folder])
            file = os.listdir(os.path.join(path, folder))
            for i in file:
                self.img.append(os.path.join(path, folder, i))
        print(self.index)
        self.transform = transforms.Compose([transforms.Resize((256,256),interpolation=Image.BICUBIC),
                                             transforms.CenterCrop((224,224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
    def __len__(self):
        return len(self.img)
    def __getitem__(self, item):
        img = Image.open(self.img[item]).convert('RGB')
        img = self.transform(img)
        label = self.label[self.img[item].split("/")[-2]]
        return {"img": img, "label": self.index.index(label)}

class ImgR(Dataset):
    def __init__(self):
        path = "/mnt/data1/rsc/DataSet/ImageNet-R"
        with open("../label.json") as f:
            self.label = json.load(f)
        self.img = []
        folders = sorted(os.listdir(path))
        self.index = []
        for folder in folders:
            self.index.append(self.label[folder])
            file = os.listdir(os.path.join(path, folder))
            for i in file:
                self.img.append(os.path.join(path, folder, i))
        print(self.index)
        self.transform = transforms.Compose([transforms.Resize((256,256),interpolation=Image.BICUBIC),
                                             transforms.CenterCrop((224,224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
    def __len__(self):
        return len(self.img)
    def __getitem__(self, item):
        img = Image.open(self.img[item]).convert('RGB')
        img = self.transform(img)
        label = self.label[self.img[item].split("/")[-2]]
        return {"img": img, "label": self.index.index(label)}


class ImgC(Dataset):
    def __init__(self, path = "/mnt/data1/rsc/DataSet/ImageNet/"):
        with open("../label.json") as f:
            self.label = json.load(f)
        self.img = []
        self.index = list(np.arange(1000))
        for folder in list(self.label.keys()):
            file = os.listdir(os.path.join(path, folder))
            for i in file:
                self.img.append(os.path.join(path, folder, i))
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
    def __len__(self):
        return len(self.img)
    def __getitem__(self, item):
        img = Image.open(self.img[item]).convert('RGB')
        img = self.transform(img)
        label = self.label[self.img[item].split("/")[-2]]
        return {"img": img, "label": label}
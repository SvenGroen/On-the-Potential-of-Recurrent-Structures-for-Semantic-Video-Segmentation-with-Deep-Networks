import random
import sys
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch
import json
from torch.utils import data
import numpy as np


def vstack(images):
    if len(images) == 0:
        raise ValueError("Need 0 or more images")

    if isinstance(images[0], np.ndarray):
        images = [Image.fromarray(img) for img in images]
    width = max([img.size[0] for img in images])
    height = sum([img.size[1] for img in images])
    stacked = Image.new(images[0].mode, (width, height))

    y_pos = 0
    for img in images:
        stacked.paste(img, (0, y_pos))
        y_pos += img.size[1]
    return stacked


def hstack(images):
    if len(images) == 0:
        raise ValueError("Need 0 or more images")

    if isinstance(images[0], np.ndarray):
        images = [Image.fromarray(img) for img in images]
    width = sum([img.size[0] for img in images])
    height = max([img.size[1] for img in images])
    stacked = Image.new(images[0].mode, (width, height))

    x_pos = 0
    for img in images:
        stacked.paste(img, (x_pos, 0))
        x_pos += img.size[0]
    return stacked


class YT_Greenscreen(data.Dataset):

    def __init__(self, train=True, start_index=torch.tensor([0]), batch_size=1):
        import os
        import sys
        sys.stderr.write("\nin dataset: " + os.getcwd() + "\n")
        print("in dataset: ", os.getcwd())

        self.train = train
        self.mode = "train" if train else "test"
        with open("src/dataset/data/images/YT_4sec/" + self.mode + "/out_log.json", "r") as json_file:
            self.data = json.load(json_file)
        self.start_index = start_index if isinstance(start_index, int) else start_index[0].item()
        self.seed = random.randint(-999, 999)  # makes sure the transformations are applied equally
        self.transform = Segmentation_transform(seed=self.seed)
        self.batch_size = batch_size
        self.cur_idx = self.start_index
        self.zeros_inp = None
        self.zeros_lbl = None

    def __len__(self):
        length = len(self.data["inputs"])
        rest = length % self.batch_size

        return length - rest

    def set_start_index(self, idx):
        if isinstance(idx, int):
            self.start_index = idx
        else:
            self.start_index = idx[0].item()

    def __getitem__(self, idx):
        idx = idx + self.start_index
        if idx >= self.__len__():
            return 0, False, (self.zeros_inp, self.zeros_lbl)
        video_start = bool(int(self.data["inputs"][idx][1]))
        if video_start:
            self.transform = Segmentation_transform(seed=random.randint(-999, 999))
            self.transform.random_apply()
            self.seed = random.randint(-999, 999)
        img = Image.open(self.data["inputs"][idx][0])
        lbl = Image.open(self.data["labels"][idx][0]).convert("L")
        random.seed(self.seed)
        inp = self.transform(img)
        random.seed(self.seed)
        lbl = (self.transform(lbl, label=True)).squeeze(0)
        self.cur_idx = idx
        if self.zeros_inp is None or self.zeros_lbl is None:
            self.zeros_inp = torch.zeros_like(inp)
            self.zeros_lbl = torch.zeros_like(lbl).long()
        return idx, video_start, (inp, lbl.round().long())

    def show(self, num_images, start_idx: int = 0, random_images=False):

        out = []
        to_PIL = T.ToPILImage()
        for i in range(num_images):
            if random_images:
                indx = np.random.randint(0, len(self))
            else:
                indx = start_idx + i
            video_start = bool(int(self.data["inputs"][indx][1]))
            if video_start:
                self.transform = Segmentation_transform(seed=random.randint(-999, 999))
            img = Image.open(self.data["inputs"][indx][0])
            lbl = Image.open(self.data["labels"][indx][0]).convert("L")
            img = self.transform(img)
            lbl = self.transform(lbl, label=True)
            out.append(hstack([to_PIL(img), to_PIL(lbl)]))
        result = vstack(out)
        result.show()


# --------- Visualization functions -----------

class Segmentation_transform:
    def __init__(self, seed):
        self.seed = seed
        self.angle = 0
        self.translate = (0, 0)
        self.shear = random.randint(-7, 7)
        self.scale = 1
        self.hflip = random.randint(0, 1)
        self.brightness = random.choice([0.6, 0.8, 1.2, 1.4])
        self.random_apply()
        self.apply_transform = True

    def deactivate_transform(self):
        self.apply_transform = False

    def __call__(self, img, label=False):
        if self.apply_transform:
            random.seed(self.seed)
            if self.hflip:
                img = TF.hflip(img=img)
            img = TF.affine(img=img, angle=self.angle, translate=self.translate, shear=self.shear, scale=self.scale)
            if not label:
                img = TF.adjust_brightness(img=img, brightness_factor=self.brightness)
        img = TF.to_tensor(img)
        if not label:
            # Normalize according to the ImageNet dataset
            img = TF.normalize(img, mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
            # bring the values back to a range of [0-1]
            invTrans = T.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            )
            img = invTrans(img)
        return img

    def renormalize(self, tensor):
        minFrom = tensor.min()
        maxFrom = tensor.max()
        minTo = 0
        maxTo = 1
        return minTo + (maxTo - minTo) * ((tensor - minFrom) / (maxFrom - minFrom))

    def random_apply(self):
        if random.random() < 0.4:
            self.angle = random.randint(-10, 10)
            self.scale = random.choice([1.2, 1.1, 1.3])
        if random.random() > 0.6:
            self.translate = (random.randint(-10, 10), random.randint(-10, 10))
            self.scale = random.choice([1.2, 1.1, 1.3])
        if random.random() > 0.4:
            self.shear = 0
        if random.random() > 0.7:
            self.brightness = 1


if __name__ == "__main__":
    from torch.utils.data import Dataset, DataLoader

    dataset = YT_Greenscreen(train=True, start_index=0)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    to_pil = T.ToPILImage()
    iter = 0
    # for i in loader:
    #     idx, next_vid, (inp, label) = i
    #     inp = to_pil(inp.squeeze(0).float())
    #     label = to_pil(label.float())
    #     inp.show()
    #     label.show()
    #     iter += 1
    #     if iter == 10: break

    # for img in inp[0,:,:,:,:]:
    #     img=to_pil(img)
    #     img.show()
    # lbl = to_pil(label[0,:,:])
    # lbl.show()

    dataset.show(4, start_idx=230)

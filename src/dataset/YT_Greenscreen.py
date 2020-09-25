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
    """
    stacks images along the vertical axis
    :param images: list of images to be stacked
    :return: the resulting stack of images as one image
    """
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
    """
     stacks images along the horizontal axis
     :param images: list of images to be stacked
     :return: the resulting stack of images as one image
     """
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
    """
    This dataset class implements data.Dataset and is supposed to be used in conjunction with PyTorch Dataloader.
    It is designed to work on the Grid Network and can be interrupted and restarted at any epoch and any batch.
    Unfortunately, it is not possible to provide the Dataloader with a "Batch start index".
    Therefore, this class has a work around that allows to continue training at any specific batch index wanted.
    It is important to consider that this class returns more than just the input and ground truth image.
    For more details pls read the __getitem__ documentation.

    The Images need to be saved in "src/dataset/data/images/YT_4sec/[train|test]" and a log file located at:
    "src/dataset/data/images/YT_4sec/[train|test]/out_log.json" needs to be provided.
    This log file contains the paths of the image files and a flag that is true if a new 4 sec. video clip starts.

    :param train: if true returns the training dataset else returns the testing dataset
    :param start_index: from which dataset index should the training start
    :param batch_size: batch size provided to the loader
    :param seed: enables reproducability
    :param apply_transform: turn on data augmentation
    """
    def __init__(self, train=True, start_index=torch.tensor([0]), batch_size=1, seed=0, apply_transform=True):
        """
                Please see help(YT_Greenscreen) for more information.
        """
        self.train = train
        self.mode = "train" if train else "test"
        with open("src/dataset/data/images/YT_4sec/" + self.mode + "/out_log.json", "r") as json_file:
            self.data = json.load(json_file)
        self.start_index = start_index if isinstance(start_index, int) else start_index[0].item()
        self.seed = seed  # makes sure the transformations are applied equally
        self.cur_idx = self.start_index
        self.transform = Segmentation_transform(seed=self.seed)
        self.batch_size = batch_size
        self.apply_transform = apply_transform
        self.zeros_inp = None
        self.zeros_lbl = None
        self.set_seeds(self.seed)

    def __len__(self):
        """
        returns the length of the dataset (implementation required).
        Cuts off the last few images that would not fit in the last batch
        :return: length of the dataset
        """
        length = len(self.data["inputs"])
        rest = length % self.batch_size

        return length - rest  # self.batch_size * 500 #length - rest

    def set_seeds(self, seed):
        """
        Ensures reproducibility
        :param seed: int: value that should be used as seed
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def set_start_index(self, idx):
        """
        helper function to set the start index

        :param idx: int or torch tensor
        """
        if isinstance(idx, int):
            self.start_index = idx
        else:
            self.start_index = idx[0].item()

    def __getitem__(self, idx):
        """
        this method is automatically called by the dataloader and returns the current idx.
        To ensure that training is continued at the desired position the idx = idx + start_index.
        The dataloader class is only finished if it returend len(dataset) batches, meaning it is necessary to manually
        check each batch if the current idx exceeds the dataset size.
        If this is the case, idx = 0 will be returned.

        in order to end the finished train batch properly use the following inside your batch loop:
        for batch in loader:
            idx, video_start, (images, labels) = batch
            # (only works for bs > 1)
            if torch.sum(idx == 0) > 1: # if more than one 0 is present in the current batch, end the batch
                self.dataset.start_index = 0  # reset start index for the next batch
                break

        :param idx: the indx to be returned (automatically called by dataloader)
        :return: idx, video_start, (images, labels)
        """
        idx = idx + self.start_index
        if idx >= self.__len__():
            return 0, False, (self.zeros_inp, self.zeros_lbl)
        video_start = bool(int(self.data["inputs"][idx][1])) # indicates whether a new video has started
        if video_start:
            # change the random augmentation with each new video start
            self.seed = random.randint(0, 999)
            print("random seed:", self.seed)
            self.transform = Segmentation_transform(seed=self.seed, activate=self.apply_transform)
            self.transform.random_apply()
        img = Image.open(self.data["inputs"][idx][0])
        lbl = Image.open(self.data["labels"][idx][0]).convert("L")
        random.seed(self.seed) # makes sure the same transforms are applied to input and lbl
        inp = self.transform(img)
        random.seed(self.seed)
        lbl = (self.transform(lbl, label=True)).squeeze(0)
        self.cur_idx = idx
        if self.zeros_inp is None or self.zeros_lbl is None: # black screen will be returned if end of dataset is reached
            self.zeros_inp = torch.zeros_like(inp)
            self.zeros_lbl = torch.zeros_like(lbl).long()
        return idx, video_start, (inp, lbl.round().long())

    def show(self, num_images, start_idx: int = 0, random_images=False):
        """
        Displayes a few example
        :param num_images: how many images should be displayed
        :param start_idx: which images should be the first one to be displayed
        :param random_images: if true displays random images
        """
        out = []
        to_PIL = T.ToPILImage()
        for i in range(num_images):
            if random_images:
                indx = np.random.randint(0, len(self))
            else:
                indx = start_idx + i
            video_start = bool(int(self.data["inputs"][indx][1]))
            if video_start:
                self.transform = Segmentation_transform(seed=random.randint(0, 20))
            img = Image.open(self.data["inputs"][indx][0])
            lbl = Image.open(self.data["labels"][indx][0]).convert("L")
            img = self.transform(img)
            lbl = self.transform(lbl, label=True)
            out.append(hstack([to_PIL(img), to_PIL(lbl)]))
        result = vstack(out)
        result.show()

class Segmentation_transform:
    """
    Custom transform function that will transform the inp and lbl
    :param seed: ensures reproducability
    :param activate: if True, augmentations are applied, else only necessary transformations are applied.
    """
    def __init__(self, seed, activate=True):
        """
        see help(Segmentation_transform)
        """
        self.seed = seed
        self.angle = 0
        self.translate = (0, 0)
        self.shear = random.randint(-7, 7)
        self.scale = 1
        self.hflip = random.randint(0, 1)
        self.brightness = random.choice([0.6, 0.8, 1.2, 1.4])
        self.random_apply()
        self.apply_transform = activate

    def deactivate_transform(self):
        """
        deactivate the augmentation transforms
        """
        self.apply_transform = False

    def __call__(self, img, label=False):
        """
        will apply random augmentations depending on if the input is a label or not
        :param img: img to be transformed
        :param label: should be True if img is a label, else false
        :return: the transformed image
        """
        random.seed(self.seed)
        if self.apply_transform:

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
        """
        normalizes images to 0 and 1
        :param tensor: input tensor
        :return: normalized tensor
        """
        minFrom = tensor.min()
        maxFrom = tensor.max()
        minTo = 0
        maxTo = 1
        return minTo + (maxTo - minTo) * ((tensor - minFrom) / (maxFrom - minFrom))

    def random_apply(self):
        """
        determines the augmentation values
        """
        if random.random() < 0.5:
            self.angle = random.randint(-10, 10)
            self.scale = random.choice([1, 1.2, 1.1, 1.3])
        if random.random() > 0.5:
            self.translate = (random.randint(-10, 10), random.randint(-10, 10))
            self.scale = random.choice([1, 1.2, 1.1, 1.3])
        if random.random() > 0.5:
            self.shear = 0
        if random.random() > 0.5:
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

from collections import defaultdict

from src.models.custom_deeplabs import *
from src.utils import SegLoss
import torch

def initiate_model(config):
    detach_interval = 1
    if config["model"] == "Deep+_mobile":
        net = Deeplabv3Plus_base(backbone="mobilenet")
        # upper_lr_bound = 1e-2
        # lower_lr_bound = upper_lr_bound / 6
        upper_lr_bound = 1e-3
        lower_lr_bound = 1e-5
        wd = 1e-8
    elif config["model"] == "Deep_mobile_lstmV1":
        net = Deeplabv3Plus_lstmV1(backbone="mobilenet")
        upper_lr_bound = 1e-3
        lower_lr_bound = 8e-5
        wd = 0
    elif config["model"] == "Deep_mobile_lstmV2":
        net = Deeplabv3Plus_lstmV2(backbone="mobilenet", activate_3d=False)
        upper_lr_bound = 1e-3
        lower_lr_bound = 8e-5
        wd = 0
    elif config["model"] == "Deep_mobile_lstmV3":
        net = Deeplabv3Plus_lstmV3(backbone="mobilenet")
        upper_lr_bound = 2e-4
        lower_lr_bound = 1e-6
        wd = 0
    elif config["model"] == "Deep_mobile_lstmV4":
        net = Deeplabv3Plus_lstmV4(backbone="mobilenet")
        upper_lr_bound = 1e-3
        lower_lr_bound = 1e-6
        wd = 0
    elif config["model"] == "Deep_mobile_lstmV5":
        net = Deeplabv3Plus_lstmV5(backbone="mobilenet", store_previous=False)
        upper_lr_bound = 2e-4
        lower_lr_bound = 2e-6
        wd = 0
    elif config["model"] == "Deep_mobile_lstmV6":
        net = Deeplabv3Plus_lstmV5(backbone="mobilenet", store_previous=True)
        upper_lr_bound = 2e-4
        lower_lr_bound = 2e-6
        wd = 0
    elif config["model"] == "Deep_mobile_lstmV7":
        net = Deeplabv3Plus_lstmV7(backbone="mobilenet")
        upper_lr_bound = 2e-4
        lower_lr_bound = 2e-6
        wd = 0
    elif config["model"] == "Deep_mobile_gruV1":
        net = Deeplabv3Plus_gruV1(backbone="mobilenet")
        upper_lr_bound = 1e-3
        lower_lr_bound = 7e-5
        wd = 1e-8
    elif config["model"] == "Deep_mobile_gruV2":
        net = Deeplabv3Plus_gruV2(backbone="mobilenet")
        upper_lr_bound = 1e-3
        lower_lr_bound = 7e-5
        wd = 1e-8
    elif config["model"] == "Deep_mobile_gruV3":
        net = Deeplabv3Plus_gruV3(backbone="mobilenet")
        upper_lr_bound = 1e-3
        lower_lr_bound = 2e-6
        wd = 0
    elif config["model"] == "Deep_mobile_gruV4":
        net = Deeplabv3Plus_gruV4(backbone="mobilenet")
        upper_lr_bound = 4e-2
        lower_lr_bound = 2e-6
        wd = 0
    elif config["model"] == "Deep_mobile_gruV5":
        net = Deeplabv3Plus_gruV5(backbone="mobilenet", store_previous=False)
        upper_lr_bound = 1e-3
        lower_lr_bound = 2e-6
        wd = 1e-8
    elif config["model"] == "Deep_mobile_gruV6":
        net = Deeplabv3Plus_gruV5(backbone="mobilenet", store_previous=True)
        upper_lr_bound = 4e-2
        lower_lr_bound = 8e-6
        wd = 1e-8
    elif config["model"] == "Deep+_resnet50":
        net = Deeplabv3Plus_base(backbone="resnet50")
        upper_lr_bound = 1e-3
        lower_lr_bound = 1e-6
        wd = 1e-8
    elif config["model"] == "Deep_resnet50_lstmV1":
        net = Deeplabv3Plus_lstmV1(backbone="resnet50")
        upper_lr_bound = 1e-3
        lower_lr_bound = 6e-5
        wd = 0
    elif config["model"] == "Deep_resnet50_lstmV2":
        net = Deeplabv3Plus_lstmV2(backbone="resnet50", activate_3d=False)
        upper_lr_bound = 1e-3
        lower_lr_bound = 1e-5
        wd = 0
    elif config["model"] == "Deep_resnet50_lstmV3":
        net = Deeplabv3Plus_lstmV3(backbone="resnet50")
        upper_lr_bound = 3e-4
        lower_lr_bound = 6e-7
        wd = 0
    elif config["model"] == "Deep_resnet50_lstmV4":
        net = Deeplabv3Plus_lstmV4(backbone="resnet50")
        upper_lr_bound = 2e-4
        lower_lr_bound = 1e-6
        wd = 1e-8
    elif config["model"] == "Deep_resnet50_lstmV5":
        net = Deeplabv3Plus_lstmV5(backbone="resnet50", store_previous=False)
        upper_lr_bound = 2e-4
        lower_lr_bound = 2e-6
        wd = 0
    elif config["model"] == "Deep_resnet50_lstmV6":
        net = Deeplabv3Plus_lstmV5(backbone="resnet50", store_previous=True)
        upper_lr_bound = 2e-4
        lower_lr_bound = 2e-6
        wd = 0
    elif config["model"] == "Deep_resnet50_gruV1":
        net = Deeplabv3Plus_gruV1(backbone="resnet50")
        upper_lr_bound = 1e-3
        lower_lr_bound = 1e-6
        wd = 1e-8
    elif config["model"] == "Deep_resnet50_gruV2":
        net = Deeplabv3Plus_gruV2(backbone="resnet50")
        upper_lr_bound = 1e-3
        lower_lr_bound = 2e-6
        wd = 0
    elif config["model"] == "Deep_resnet50_gruV3":
        net = Deeplabv3Plus_gruV3(backbone="resnet50")
        upper_lr_bound = 2e-4
        lower_lr_bound = 5e-7
        wd = 0
    elif config["model"] == "Deep_resnet50_gruV4":
        net = Deeplabv3Plus_gruV4(backbone="resnet50")
        upper_lr_bound = 2e-4
        lower_lr_bound = 6e-7
        wd = 0
    elif config["model"] == "Deep_resnet50_gruV5":
        net = Deeplabv3Plus_gruV5(backbone="resnet50", store_previous=False)
        upper_lr_bound = 1e-3
        lower_lr_bound = 2e-6
        wd = 0
    elif config["model"] == "Deep_resnet50_gruV6":
        net = Deeplabv3Plus_gruV5(backbone="resnet50", store_previous=True)
        upper_lr_bound = 1e-3
        lower_lr_bound = 2e-6
        wd = 0
    else:
        net = None
        lower_lr_bound = None
        upper_lr_bound = None
        wd = 0

    net.train()
    for param in net.base.backbone.parameters():
        param.requires_grad = False

    return net, wd, (lower_lr_bound, upper_lr_bound), detach_interval


def initiate_criterion(config):
    if config["loss"] == "SoftDice":
        criterion = SegLoss.dice_loss.SoftDiceLoss(smooth=0.0001, apply_nonlin=torch.nn.Softmax(dim=1))
    elif config["loss"] == "Focal":
        criterion = SegLoss.focal_loss.FocalLoss(smooth=0.0001, apply_nonlin=torch.nn.Softmax(dim=1))
    elif config["loss"] == "CrossDice":
        dice = SegLoss.dice_loss.SoftDiceLoss(smooth=0.0001, apply_nonlin=torch.nn.Softmax(dim=1))
        entropy = torch.nn.CrossEntropyLoss()
        criterion = lambda x, y: (dice(x, y) + entropy(x, y)) / 2.
    elif config["loss"] == "CrossEntropy":
        criterion = torch.nn.CrossEntropyLoss()
    return criterion


def initiate_logger(lr):
    logger = defaultdict(list)
    logger["epochs"].append(0)
    logger["lrs"].append(lr)
    logger["batch_index"] = 0
    return logger

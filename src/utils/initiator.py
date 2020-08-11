from collections import defaultdict

from src.models.custom_deeplabs import *
from src.utils import SegLoss


def initiate_model(config):
    if config["model"] == "Deep+_mobile":
        net = Deeplabv3Plus_base(backbone="mobilenet")
        upper_lr_bound = 0.0001
        lower_lr_bound = upper_lr_bound / 6
    elif config["model"] == "Deep_mobile_lstmV1":
        net = Deeplabv3Plus_lstmV1(backbone="mobilenet")
        upper_lr_bound = 5e-5
        lower_lr_bound = upper_lr_bound / 6
    elif config["model"] == "Deep_mobile_lstmV2_1":
        net = Deeplabv3Plus_lstmV2(backbone="mobilenet", activate_3d=False)
        upper_lr_bound = 2e-5
        lower_lr_bound = upper_lr_bound / 6
    elif config["model"] == "Deep_mobile_lstmV2_2":
        net = Deeplabv3Plus_lstmV2(backbone="mobilenet", activate_3d=True)
        upper_lr_bound = 3e-3
        lower_lr_bound = upper_lr_bound / 6
    elif config["model"] == "Deep_mobile_lstmV3":
        net = Deeplabv3Plus_lstmV3(backbone="mobilenet")
        upper_lr_bound = 1e-2
        lower_lr_bound = upper_lr_bound / 6
    elif config["model"] == "Deep_mobile_lstmV4":
        net = Deeplabv3Plus_lstmV4(backbone="mobilenet")
        upper_lr_bound = 3e-5
        lower_lr_bound = upper_lr_bound / 6
    elif config["model"] == "Deep_mobile_lstmV5_1":
        net = Deeplabv3Plus_lstmV5(backbone="mobilenet", keep_hidden=True)
        upper_lr_bound = 0.0023
        lower_lr_bound = upper_lr_bound / 6
    elif config["model"] == "Deep_mobile_lstmV5_2":
        net = Deeplabv3Plus_lstmV5(backbone="mobilenet", keep_hidden=False)
        upper_lr_bound = 1e-5
        lower_lr_bound = upper_lr_bound / 6
    elif config["model"] == "Deep_mobile_gruV1":
        net = Deeplabv3Plus_gruV1(backbone="mobilenet")
        upper_lr_bound = 1e-5
        lower_lr_bound = upper_lr_bound / 6
    elif config["model"] == "Deep_mobile_gruV2":
        net = Deeplabv3Plus_gruV2(backbone="mobilenet")
        upper_lr_bound = 5e-4
        lower_lr_bound = upper_lr_bound / 6
    elif config["model"] == "Deep_mobile_gruV3":
        net = Deeplabv3Plus_gruV3(backbone="mobilenet")
        upper_lr_bound = 1e-4  # 1e-5
        lower_lr_bound = upper_lr_bound / 6
    elif config["model"] == "Deep_mobile_gruV4":
        net = Deeplabv3Plus_gruV4(backbone="mobilenet")
        upper_lr_bound = 7e-5
        lower_lr_bound = upper_lr_bound / 6
    elif config["model"] == "Deep+_resnet50":
        net = Deeplabv3Plus_base(backbone="resnet50")
        upper_lr_bound = 0.00055
        lower_lr_bound = upper_lr_bound / 6
    elif config["model"] == "Deep_resnet50_lstmV1":
        net = Deeplabv3Plus_lstmV1(backbone="resnet50")
        upper_lr_bound = 0.002
        lower_lr_bound = upper_lr_bound / 6
    elif config["model"] == "Deep_resnet50_lstmV2_1":
        net = Deeplabv3Plus_lstmV2(backbone="resnet50", activate_3d=False)
        upper_lr_bound = 0.002
        lower_lr_bound = upper_lr_bound / 6
    elif config["model"] == "Deep_resnet50_lstmV2_2":
        net = Deeplabv3Plus_lstmV2(backbone="resnet50", activate_3d=True)
        upper_lr_bound = 0.002
        lower_lr_bound = upper_lr_bound / 6
    elif config["model"] == "Deep_resnet50_lstmV3":
        net = Deeplabv3Plus_lstmV3(backbone="resnet50")
        upper_lr_bound = 0.00055
        lower_lr_bound = upper_lr_bound / 6
    elif config["model"] == "Deep_resnet50_lstmV4":
        net = Deeplabv3Plus_lstmV4(backbone="resnet50")
        upper_lr_bound = 0.00025
        lower_lr_bound = upper_lr_bound / 6
    elif config["model"] == "Deep_resnet50_lstmV5_1":
        net = Deeplabv3Plus_lstmV5(backbone="resnet50", keep_hidden=True)
        upper_lr_bound = 0.0023
        lower_lr_bound = upper_lr_bound / 6
    elif config["model"] == "Deep_resnet50_lstmV5_2":
        net = Deeplabv3Plus_lstmV5(backbone="resnet50", keep_hidden=False)
        upper_lr_bound = 0.001
        lower_lr_bound = upper_lr_bound / 6
    elif config["model"] == "Deep_resnet50_gruV1":
        net = Deeplabv3Plus_gruV1(backbone="resnet50")
        upper_lr_bound = 0.00055
        lower_lr_bound = upper_lr_bound / 6
    elif config["model"] == "Deep_resnet50_gruV2":
        net = Deeplabv3Plus_gruV2(backbone="resnet50")
        upper_lr_bound = 0.001
        lower_lr_bound = upper_lr_bound / 6
    elif config["model"] == "Deep_resnet50_gruV3":
        net = Deeplabv3Plus_gruV3(backbone="resnet50")
        upper_lr_bound = 0.000055
        lower_lr_bound = upper_lr_bound / 6
    elif config["model"] == "Deep_resnet50_gruV4":
        net = Deeplabv3Plus_gruV4(backbone="resnet50")
        upper_lr_bound = 0.00026
        lower_lr_bound = upper_lr_bound / 6
    else:
        net = None
        lower_lr_bound = None
        upper_lr_bound = None

    for param in net.base.backbone.parameters():
        param.requires_grad = False

    return net, (lower_lr_bound, upper_lr_bound)


def initiate_criterion(config):
    if config["loss"] == "SoftDice":
        criterion = SegLoss.dice_loss.SoftDiceLoss(smooth=0.0001, apply_nonlin=F.softmax)
    elif config["loss"] == "Focal":
        criterion = SegLoss.focal_loss.FocalLoss(smooth=0.0001, apply_nonlin=F.softmax)
    elif config["loss"] == "Boundary":
        criterion = SegLoss.boundary_loss.BDLoss()
    elif config["loss"] == "CrossDice":
        dice = SegLoss.dice_loss.SoftDiceLoss(smooth=0.0001, apply_nonlin=F.softmax)
        entropy = torch.nn.CrossEntropyLoss()
        criterion = lambda x, y: (dice(x, y) + entropy(x, y)) / 2.
    return criterion


def initiate_logger(lr):
    logger = defaultdict(list)
    logger["epochs"].append(0)
    logger["lrs"].append(lr)
    logger["batch_index"] = 0
    return logger

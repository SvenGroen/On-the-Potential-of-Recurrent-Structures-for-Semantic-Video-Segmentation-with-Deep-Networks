import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.recurrent_modules import *
from src.models.network import *
from src.models.network._deeplab import DeepLabHeadV3PlusLSTM, DeepLabHeadV3PlusGRU, DeepLabHeadV3PlusLSTMV2, \
    DeepLabHeadV3PlusGRUV2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
This File contains all Deeplab alternations.
contains:
V1 - 5 for lstm and gru
(V6 is created through different V5 initialization)
"""

# BASE
class Deeplabv3Plus_base(nn.Module):
    """
    base model with either a mobilenet or resnet backbone.

    :param backbone: mobilenet or resnet50
    """
    def __init__(self, backbone="mobilenet"):
        super().__init__()
        if backbone == "mobilenet":
            self.base = deeplabv3plus_mobilenet(num_classes=2, pretrained_backbone=True)
        elif backbone == "resnet50":
            self.base = deeplabv3plus_resnet50(num_classes=2, pretrained_backbone=True)
    def detach(self):
        pass

    def reset(self):
        pass

    def start_eval(self):
        pass

    def end_eval(self):
        pass

    def forward(self, x, *args):
        return self.base(x)


# --- LSTMs ---

class Deeplabv3Plus_lstmV1(nn.Module):
    """
    Base model with lstm that receives no additional timesteps.
    Lstm is located at the end of the model.
    """
    def __init__(self, backbone="mobilenet"):
        super().__init__()
        if backbone == "mobilenet":
            self.base = deeplabv3plus_mobilenet(num_classes=2, pretrained_backbone=True)
        elif backbone == "resnet50":
            self.base = deeplabv3plus_resnet50(num_classes=2, pretrained_backbone=True)

        self.lstm = ConvLSTM(input_dim=2, hidden_dim=[2], kernel_size=(3, 3), num_layers=1, batch_first=True,
                             bias=True,
                             return_all_layers=False)
        self.hidden = None
        self.tmp_hidden = None

    def detach(self):
        self.hidden = [tuple(state.detach() for state in i) for i in self.hidden]

    def reset(self):
        self.hidden = None

    def start_eval(self):
        self.tmp_hidden = self.hidden
        self.hidden = None

    def end_eval(self):
        self.hidden = self.tmp_hidden
        self.tmp_hidden = None

    def forward(self, x, *args):
        input_shape = x.shape[-2:]
        out = self.base(x)
        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)
        out = out.unsqueeze(1)
        out, self.hidden = self.lstm(out, self.hidden)
        self.hidden = [tuple(state.detach() for state in i) for i in self.hidden]
        return out[-1].squeeze(1)

class Deeplabv3Plus_lstmV2(nn.Module):
    """
    Base model with lstm that receives 2 additional timesteps.
    Lstm is located at the end of the model.
    """
    def __init__(self, backbone="mobilenet"):
        super().__init__()
        if backbone == "mobilenet":
            self.base = deeplabv3plus_mobilenet(num_classes=2, pretrained_backbone=True)
        elif backbone == "resnet50":
            self.base = deeplabv3plus_resnet50(num_classes=2, pretrained_backbone=True)

        self.lstm = ConvLSTM(input_dim=2, hidden_dim=[2], kernel_size=(3, 3), num_layers=1, batch_first=True,
                             bias=True,
                             return_all_layers=True)

        self.hidden = None
        self.tmp_hidden = None
        self.tmp_old_pred = [None, None]
        self.old_pred = [None, None]

    def detach(self):
        pass

    def reset(self):
        self.hidden = None
        self.old_pred = [None, None]

    def start_eval(self):
        self.tmp_hidden = self.hidden
        self.hidden = None
        self.tmp_old_pred = self.old_pred
        self.old_pred = [None, None]

    def end_eval(self):
        self.hidden = self.tmp_hidden
        self.tmp_hidden = None
        self.old_pred = self.tmp_old_pred
        self.tmp_old_pred = [None, None]

    def forward(self, x, *args):
        input_shape = x.shape[-2:]
        x = self.base(x)
        out = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        out = out.unsqueeze(1)
        if None in self.old_pred:
            for i in range(len(self.old_pred)):
                self.old_pred[i] = torch.zeros_like(out)
        # match shape
        elif len(self.old_pred[0].shape) != len(out.shape):
            for i in range(len(self.old_pred)):
                self.old_pred[i] = self.old_pred[i].unsqueeze(1)
        out = self.old_pred + [out]
        out = torch.cat(out, dim=1)

        out, self.hidden = self.lstm(out, self.hidden)
        out = out[0]
        out = out[:, -1, :, :, :]  # <--- not to sure if 0 or -1
        self.hidden = [tuple(state.detach() for state in i) for i in self.hidden]
        self.old_pred[0] = self.old_pred[1]  # oldest at 0 position
        self.old_pred[1] = out.unsqueeze(1).detach()  # newest at 1 position
        return out

class Deeplabv3Plus_lstmV3(nn.Module):
    """
    Base model with lstm that receives no additional timesteps.
    Lstm is located after concatenation of encoder output and low level features.
    """
    def __init__(self, backbone="mobilenet"):
        super().__init__()
        if backbone == "mobilenet":
            self.base = deeplabv3plus_mobilenet(num_classes=2, pretrained_backbone=True)
            in_channels = 320
            low_level_channels = 24
        elif backbone == "resnet50":
            self.base = deeplabv3plus_resnet50(num_classes=2, pretrained_backbone=True)
            in_channels = 2048
            low_level_channels = 256
        self.base.classifier = DeepLabHeadV3PlusLSTM(in_channels, low_level_channels, 2, [12, 24, 36])
        self.tmp_old_pred = [None, None]
        self.tmp_hidden = None

    def detach(self):
        pass

    def reset(self):
        self.base.classifier.hidden = None
        self.base.classifier.old_pred = [None, None]

    def start_eval(self):
        self.tmp_hidden = self.base.classifier.hidden
        self.tmp_old_pred = self.base.classifier.old_pred
        self.reset()

    def end_eval(self):
        self.base.classifier.hidden = self.tmp_hidden
        self.base.classifier.old_pred = self.tmp_old_pred
        self.tmp_hidden = None
        self.tmp_old_pred = [None, None]

    def forward(self, x, *args):
        input_shape = x.shape[-2:]
        out = self.base(x)

        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)
        return out

class Deeplabv3Plus_lstmV4(nn.Module):
    """
    Base model with lstm that receives two additional timesteps.
    Lstm is located after concatenation of encoder output and low level features.
    """
    def __init__(self, backbone="mobilenet"):
        super().__init__()
        if backbone == "mobilenet":
            self.base = deeplabv3plus_mobilenet(num_classes=2, pretrained_backbone=True)
            in_channels = 320
            low_level_channels = 24
        elif backbone == "resnet50":
            self.base = deeplabv3plus_resnet50(num_classes=2, pretrained_backbone=True)
            in_channels = 2048
            low_level_channels = 256
        self.base.classifier = DeepLabHeadV3PlusLSTM(in_channels, low_level_channels, 2, [12, 24, 36],
                                                     store_previous=True)
        self.tmp_old_pred = [None, None]
        self.tmp_hidden = None

    def detach(self):
        pass

    def reset(self):
        self.base.classifier.hidden = None
        self.base.classifier.old_pred = [None, None]

    def start_eval(self):
        self.tmp_hidden = self.base.classifier.hidden
        self.tmp_old_pred = self.base.classifier.old_pred
        self.reset()

    def end_eval(self):
        self.base.classifier.hidden = self.tmp_hidden
        self.base.classifier.old_pred = self.tmp_old_pred
        self.tmp_hidden = None
        self.tmp_old_pred = [None, None]

    def forward(self, x, *args):
        input_shape = x.shape[-2:]
        out = self.base(x)
        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)
        return out

class Deeplabv3Plus_lstmV5(nn.Module):
    """
    Base model with lstm that uses 1x1 convolutions to reduce complexity.
    Lstm is located after concatenation of encoder output and low level features.
    """
    def __init__(self, backbone="mobilenet", store_previous=False):
        super().__init__()
        if backbone == "mobilenet":
            self.base = deeplabv3plus_mobilenet(num_classes=2, pretrained_backbone=True)
            in_channels = 320
            low_level_channels = 24
        elif backbone == "resnet50":
            self.base = deeplabv3plus_resnet50(num_classes=2, pretrained_backbone=True)
            in_channels = 2048
            low_level_channels = 256
        self.base.classifier = DeepLabHeadV3PlusLSTMV2(in_channels, low_level_channels, 2, [12, 24, 36],
                                                     store_previous=store_previous)
        self.tmp_old_pred = [None, None]
        self.tmp_hidden = None

    def detach(self):
        pass

    def reset(self):
        self.base.classifier.hidden = None
        self.base.classifier.old_pred = [None, None]

    def start_eval(self):
        self.tmp_hidden = self.base.classifier.hidden
        self.tmp_old_pred = self.base.classifier.old_pred
        self.reset()

    def end_eval(self):
        self.base.classifier.hidden = self.tmp_hidden
        self.base.classifier.old_pred = self.tmp_old_pred
        self.tmp_hidden = None
        self.tmp_old_pred = [None, None]

    def forward(self, x, *args):
        input_shape = x.shape[-2:]
        out = self.base(x)
        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)
        return out

class Deeplabv3Plus_lstmV7(nn.Module):
    """
    test version;
    """
    def __init__(self, backbone="mobilenet", keep_hidden=True):
        super().__init__()
        if backbone == "mobilenet":
            self.base = deeplabv3plus_mobilenet(num_classes=2, pretrained_backbone=True)
        elif backbone == "resnet50":
            self.base = deeplabv3plus_resnet50(num_classes=2, pretrained_backbone=True)
        if keep_hidden:
            return_all_layers = True
        else:
            return_all_layers = False
        self.lstm = ConvLSTM(input_dim=2, hidden_dim=[2], kernel_size=(3, 3), num_layers=1, batch_first=True,
                             bias=True,
                             return_all_layers=return_all_layers)
        self.hidden = None
        self.tmp_hidden = None
        self.keep_hidden = keep_hidden
        self.tmp_old_pred = [None, None]
        self.old_pred = [None, None]

    def detach(self):
        pass

    def reset(self):
        self.hidden = None
        self.old_pred = [None, None]

    def start_eval(self):
        self.tmp_hidden = self.hidden
        self.hidden = None
        self.tmp_old_pred = self.old_pred
        self.old_pred = [None, None]

    def end_eval(self):
        self.hidden = self.tmp_hidden
        self.tmp_hidden = None
        self.old_pred = self.tmp_old_pred
        self.tmp_old_pred = [None, None]

    def forward(self, x, *args):
        input_shape = x.shape[-2:]
        x = self.base(x)
        out = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        out = out.unsqueeze(1)

        # initialize if necessary
        if None in self.old_pred:
            for i in range(len(self.old_pred)):
                self.old_pred[i] = torch.zeros_like(out)
        # match shape
        elif len(self.old_pred[0].shape) != len(out.shape):
            for i in range(len(self.old_pred)):
                self.old_pred[i] = self.old_pred[i].unsqueeze(1)
        out = self.old_pred + [out]
        out = torch.cat(out, dim=1)
        if self.keep_hidden:
            out, self.hidden = self.lstm(out, self.hidden)
        else:
            out, self.hidden = self.lstm(out)
        out = out[0][:, -1, :, :, :]  # <--- not to sure if 0 or -1
        self.hidden = [tuple(state.detach() for state in i) for i in self.hidden]
        self.old_pred[0] = self.old_pred[1]  # oldest at 0 position
        self.old_pred[1] = out.unsqueeze(1).detach()
        return out

# --- GRU ---
class Deeplabv3Plus_gruV1(nn.Module):
    """
    Base model with gru that receives no additional timesteps.
    Gru is located at the end of the model.
    """
    def __init__(self, backbone="mobilenet"):
        super().__init__()
        if backbone == "mobilenet":
            self.base = deeplabv3plus_mobilenet(num_classes=2, pretrained_backbone=True)
        elif backbone == "resnet50":
            self.base = deeplabv3plus_resnet50(num_classes=2, pretrained_backbone=True)

        self.gru = ConvGRU(input_size=(270, 512), input_dim=2, hidden_dim=[2], kernel_size=(3, 3), num_layers=1,
                           dtype=torch.FloatTensor, batch_first=True, bias=True, return_all_layers=True)
        self.hidden = [None]
        self.tmp_hidden = None

    def detach(self):
        pass

    def reset(self):
        self.hidden = [None]

    def start_eval(self):
        self.tmp_hidden = self.hidden
        self.hidden = [None]

    def end_eval(self):
        self.hidden = self.tmp_hidden
        self.tmp_hidden = [None]

    def forward(self, x, *args):
        x = self.base(x)
        x = x.unsqueeze(1)
        out, self.hidden = self.gru(x, self.hidden[-1])
        self.hidden = [tuple(state.detach() for state in i) for i in self.hidden]
        out = out[0][:, -1, :, :, :]
        return out

class Deeplabv3Plus_gruV2(nn.Module):
    """
    Base model with gru that receives two additional timesteps.
    Gru is located at the end of the model.
    """
    def __init__(self, backbone="mobilenet"):
        super().__init__()
        if backbone == "mobilenet":
            self.base = deeplabv3plus_mobilenet(num_classes=2, pretrained_backbone=True)
        elif backbone == "resnet50":
            self.base = deeplabv3plus_resnet50(num_classes=2, pretrained_backbone=True)

        self.gru = ConvGRU(input_size=(270, 512), input_dim=2, hidden_dim=[2], kernel_size=(3, 3), num_layers=1,
                           dtype=torch.FloatTensor, batch_first=True, bias=True, return_all_layers=True)
        self.hidden = [None]
        self.tmp_hidden = [None]
        self.old_pred = [None, None]
        self.tmp_old_pred = [None, None]

    def detach(self):
        pass

    def reset(self):
        self.hidden = [None]
        self.old_pred = [None, None]

    def start_eval(self):
        self.tmp_hidden = self.hidden
        self.hidden = [None]
        self.tmp_old_pred = self.old_pred
        self.old_pred = [None, None]

    def end_eval(self):
        self.hidden = self.tmp_hidden
        self.tmp_hidden = [None]
        self.old_pred = self.tmp_old_pred
        self.tmp_old_pred = [None, None]

    def forward(self, x, *args):
        out = self.base(x)
        out = out.unsqueeze(1)  # add "timestep" dimension

        if None in self.old_pred:
            for i in range(len(self.old_pred)):
                self.old_pred[i] = torch.zeros_like(out)
        # match shape
        elif len(self.old_pred[0].shape) != len(out.shape):
            for i in range(len(self.old_pred)):
                self.old_pred[i] = self.old_pred[i].unsqueeze(1)  # add "timestep" dimension
        out = self.old_pred + [out]
        out = torch.cat(out, dim=1)
        out, self.hidden = self.gru(out, self.hidden[-1])
        self.hidden = [tuple(state.detach() for state in i) for i in self.hidden]
        out = out[0]
        # out = self.conv3d(out)
        out = out[:, -1, :, :, :]  # <--- not to sure if 0 or -1
        self.old_pred[0] = self.old_pred[1]  # oldest at 0 position
        self.old_pred[1] = out.unsqueeze(1).detach()  # newest at 1 position
        return out

class Deeplabv3Plus_gruV3(nn.Module):
    """
    Base model with gru that receives no additional timesteps.
    Gru is located after concatenation of encoder output and low level features.
    """
    def __init__(self, backbone="mobilenet"):
        super().__init__()
        if backbone == "mobilenet":
            self.base = deeplabv3plus_mobilenet(num_classes=2, pretrained_backbone=True)
            in_channels = 320
            low_level_channels = 24
        elif backbone == "resnet50":
            self.base = deeplabv3plus_resnet50(num_classes=2, pretrained_backbone=True)
            in_channels = 2048
            low_level_channels = 256
        self.base.classifier = None
        self.classifier = DeepLabHeadV3PlusGRU(in_channels, low_level_channels, 2, [12, 24, 36],
                                               store_previous=False).to(device)
        self.hidden = self.classifier.hidden
        self.tmp_hidden = None
        self.classifier.old_pred = [None, None]
        self.tmp_old_pred = [None, None]
    def detach(self):
        pass

    def reset(self):
        self.classifier.hidden = [None]
        self.classifier.old_pred = [None, None]

    def start_eval(self):
        self.tmp_hidden = self.classifier.hidden
        self.tmp_old_pred = self.classifier.old_pred
        self.classifier.hidden = [None]
        self.classifier.old_pred = [None, None]

    def end_eval(self):
        self.classifier.hidden = self.tmp_hidden
        self.classifier.old_pred = self.tmp_old_pred
        self.tmp_hidden = [None]
        self.tmp_old_pred = [None, None]

    def forward(self, x, *args):
        input_shape = x.shape[-2:]
        features = self.base.backbone(x)
        out = self.classifier(features)
        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)
        return out

class Deeplabv3Plus_gruV4(nn.Module):
    """
    Base model with gru that receives two additional timesteps.
    Gru is located after concatenation of encoder output and low level features.
    """
    def __init__(self, backbone="mobilenet"):
        super().__init__()
        if backbone == "mobilenet":
            self.base = deeplabv3plus_mobilenet(num_classes=2, pretrained_backbone=True)
            in_channels = 320
            low_level_channels = 24
        elif backbone == "resnet50":
            self.base = deeplabv3plus_resnet50(num_classes=2, pretrained_backbone=True)
            in_channels = 2048
            low_level_channels = 256
        self.base.classifier = None
        self.classifier = DeepLabHeadV3PlusGRU(in_channels, low_level_channels, 2, [12, 24, 36],
                                               store_previous=True).to(device)
        self.hidden = self.classifier.hidden
        self.tmp_hidden = None
        self.classifier.old_pred = [None, None]
        self.tmp_old_pred = [None, None]
    def detach(self):
        pass

    def reset(self):
        self.classifier.hidden = [None]
        self.classifier.old_pred = [None, None]

    def start_eval(self):
        self.tmp_hidden = self.classifier.hidden
        self.tmp_old_pred = self.classifier.old_pred
        self.classifier.hidden = [None]
        self.classifier.old_pred = [None, None]

    def end_eval(self):
        self.classifier.hidden = self.tmp_hidden
        self.classifier.old_pred = self.tmp_old_pred
        self.tmp_hidden = [None]
        self.tmp_old_pred = [None, None]

    def forward(self, x, *args):
        input_shape = x.shape[-2:]
        features = self.base.backbone(x)
        out = self.classifier(features)
        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)
        return out

class Deeplabv3Plus_gruV5(nn.Module):
    """
    Base model with gru that uses 1x1 convolutions to reduce complexity.
    Gru is located after concatenation of encoder output and low level features.
    """
    def __init__(self, backbone="mobilenet", store_previous=False):
        super().__init__()
        if backbone == "mobilenet":
            self.base = deeplabv3plus_mobilenet(num_classes=2, pretrained_backbone=True)
            in_channels = 320
            low_level_channels = 24
        elif backbone == "resnet50":
            self.base = deeplabv3plus_resnet50(num_classes=2, pretrained_backbone=True)
            in_channels = 2048
            low_level_channels = 256
        self.base.classifier = DeepLabHeadV3PlusGRUV2(in_channels, low_level_channels, 2, [12, 24, 36],
                                                     store_previous=store_previous)
        self.tmp_old_pred = [None, None]
        self.tmp_hidden = None

    def detach(self):
        pass

    def reset(self):
        self.base.classifier.hidden = [None]
        self.base.classifier.old_pred = [None, None]

    def start_eval(self):
        self.tmp_hidden = self.base.classifier.hidden
        self.tmp_old_pred = self.base.classifier.old_pred
        self.reset()

    def end_eval(self):
        self.base.classifier.hidden = self.tmp_hidden
        self.base.classifier.old_pred = self.tmp_old_pred
        self.tmp_hidden = None
        self.tmp_old_pred = [None, None]

    def forward(self, x, *args):
        input_shape = x.shape[-2:]
        out = self.base(x)
        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)
        return out

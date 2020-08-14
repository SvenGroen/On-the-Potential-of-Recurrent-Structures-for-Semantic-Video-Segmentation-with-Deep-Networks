# Video_Segmentation


# References:
SegLoss:
https://github.com/JunMa11/SegLoss
SoftDiceLoss was used

Metrics:


DeeplabV3+:
https://github.com/VainF/DeepLabV3Plus-Pytorch
custom DeeplabV3-Head added (DeeplabV3-LSTMHead and DeeplabV3-GRUHead)

ConvLSTM:
https://github.com/ndrplz/ConvLSTM_pytorch
modified to keep hidden and cell state until signal is given

ConvGRU:
https://github.com/happyjin/ConvGRU-pytorch
modified to keep hidden and cell state until signal is given

LR_finder:
https://github.com/davidtvs/pytorch-lr-finder
modified to fit custom dataset return values and reseting of the model if a new 4 second clip starts
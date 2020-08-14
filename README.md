# Video_Segmentation


# References and changes:
SegLoss:  
https://github.com/JunMa11/SegLoss  
SoftDiceLoss was used.

Metrics:


DeeplabV3+:  
https://github.com/VainF/DeepLabV3Plus-Pytorch  
custom DeeplabV3-Head added (DeeplabV3-LSTMHead and DeeplabV3-GRUHead).

ConvLSTM:  
https://github.com/ndrplz/ConvLSTM_pytorch  
modified to keep hidden and cell state until signal is given.

ConvGRU:  
https://github.com/happyjin/ConvGRU-pytorch  
modified to keep hidden and cell state until signal is given.

LR_finder:  
https://github.com/davidtvs/pytorch-lr-finder  
use: `pip install torch-lr-finder` to install the LR-finder package.  
I extracted the downloaded "torch_lr_finder" folder from: 
> "C:\ProgramData\Anaconda3\envs\<my_conda_env>\Lib\site-packages\torch_lr_finder" 

placed it into: 
> Video_Segementation/src/utils/

and made changed to `__init__.py` and `lr_finder.py`.
I modified `lr_finder.py` to fit my custom dataset return values and resetting of the model if a new 4 second clip starts.

# Pip list:
Package and Version            

certifi==2020.4.5.1         
cycler ==         0.10.0             
joblib   ==       0.14.1             
kiwisolver ==     1.2.0              
matplotlib   ==   3.2.1              
numpy          == 1.18.3             
opencv-python==   4.2.0.34           
packaging   ==    20.4               
pandas    ==      1.0.3              
Pillow    ==      7.1.1              
pip         ==    20.0.2             
pyparsing     ==  2.4.7              
python-dateutil== 2.8.1              
pytz        ==    2019.3             
PyYAML      ==    5.1.2              
scikit-learn   == 0.22.2.post1       
scipy        ==   1.4.1              
setuptools     == 46.1.3.post20200330
six          ==   1.14.0             
torch         ==  1.4.0              
torch-lr-finder== 0.2.0              
torchsummary  ==  1.5.1              
torchvision   ==  0.5.0              
tqdm         ==   4.45.0             
wheel       ==    0.34.2 

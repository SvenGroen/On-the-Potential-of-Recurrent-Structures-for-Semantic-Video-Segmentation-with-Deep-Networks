import argparse
import json
import sys
import torch
from torch import optim
from src.utils.torch_lr_finder.lr_finder import LRFinder
from torch.utils.data import DataLoader
from src.dataset.YT_Greenscreen import YT_Greenscreen
from src.gridtrainer import GridTrainer
import matplotlib.pyplot as plt

"""
Starts LR range test for multiple hyperparmaeters and plots them
"""

# -cfg src\models\LR_Tests\bs_6\Deep_resnet50_gruV4bs6num_iter100ID19/train_config.json

device = "cuda" if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser()

parser.add_argument("-cfg", "--config",
                    help="Config location", type=str)
args = parser.parse_args()

with open(args.config) as js:
    config = json.load(js)

historys = []
weight_decays = [0, 1e-4, 1e-6, 1e-8]
# batch sizes = [2, 4, 6, 8, 10, 12]


for wd in weight_decays:
    sys.stderr.write(f"\nParam: {str(wd)}, hist: {historys}\n")
    trainer = GridTrainer(config, load_from_checkpoint=False, batch_size=config["batch_size"])
    model = trainer.model
    criterion = trainer.criterion
    val_dataset = YT_Greenscreen(train=False, start_index=0,
                                 batch_size=config["batch_size"])  # batch_size=trainer.config["batch_size"])
    val_loader = DataLoader(val_dataset, val_dataset.batch_size, shuffle=False)
    optimizer = optim.Adam(model.parameters(), lr=1e-7, weight_decay=wd)
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    try:
        lr_finder.range_test(trainer.loader, end_lr=10, num_iter=config["num_epochs"])  # val_loader=val_loader
        historys.append(lr_finder.history)
        lr_finder.plot(skip_start=0, skip_end=0)
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f'| WARNING: ran out of memory for batch size {wd}')
        torch.cuda.empty_cache()
        weight_decays.remove(wd)
    lr_finder.reset()

# creates a plot of the lr range test results
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111)

for wd, hist in zip(weight_decays, historys):
    lrs = hist["lr"]
    losses = hist["loss"]
    ax.plot(lrs, losses, label=str(wd))
    ax.set_xscale("log")
# x_ticks = [i*10 ** - exponent for exponent in range(9, 1, -1) for i in range(1,10,1)]
# plt.xticks(x_ticks)
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
ax.legend(title="Weight Decay")
plt.title("Learning Rate vs Loss for different weight decays")

fig.savefig(str(config["save_files_path"] + "/lr_analysis.png"))

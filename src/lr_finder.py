import argparse
import json
import torch
from torch import optim
from src.utils.torch_lr_finder.lr_finder import LRFinder
from torch.utils.data import DataLoader
from src.dataset.YT_Greenscreen import YT_Greenscreen
from src.gridtrainer import GridTrainer
import matplotlib.pyplot as plt
#-cfg src\models\LR_Tests\bs_6\Deep_resnet50_gruV4bs6num_iter100ID19/train_config.json

device = "cuda" if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser()

parser.add_argument("-cfg", "--config",
                    help="Config location", type=str)
args = parser.parse_args()

with open(args.config) as js:
    config = json.load(js)

historys = []
weight_decays = [0, 1e-2, 1e-4, 1e-6]
for wd in weight_decays:
    trainer = GridTrainer(config, load_from_checkpoint=False)
    model = trainer.model
    criterion = trainer.criterion
    val_dataset = YT_Greenscreen(train=False, start_index=0, batch_size=trainer.config["batch_size"])
    val_loader = DataLoader(val_dataset, val_dataset.batch_size, shuffle=False)
    optimizer = optim.Adam(model.parameters(), lr=1e-7, weight_decay=wd)
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(trainer.loader, end_lr=10, num_iter=config["num_epochs"])  # , val_loader=val_loader
    historys.append(lr_finder.history)
    lr_finder.plot(skip_start=0, skip_end=0)
    lr_finder.reset()

for wd, hist in zip(weight_decays, historys):
    lrs = hist["lr"]
    losses = hist["loss"]
    plt.plot(lrs, losses, label=str(wd))
    plt.xscale("log")
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.legend()
plt.title("Learning Rate vs Loss for different weight decay values")
plt.savefig(str(config["save_files_path"] + "/lr_analysis.png"), dpi=400)

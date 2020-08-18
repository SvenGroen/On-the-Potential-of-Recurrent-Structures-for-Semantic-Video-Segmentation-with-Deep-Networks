import argparse
import json

import torch

from src.gridtrainer import GridTrainer
from src.utils.visualizations import visualize_metric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument("-stps", "--steps",
                    help="The number of evalution steps to be performed", type=int)
parser.add_argument("-rdm", "--random",
                    help="Should the start of the evaluation be at a random position", type=int)
parser.add_argument("-pth", "--path",
                    help="The path of the folder", type=str)
parser.add_argument("-fnl", "--final",
                    help="Is this the last episode", type=int)
args = parser.parse_args()

with open(args.path + "/train_config.json") as js:
    print("Loading config: ", args.path)
    config = json.load(js)
out = args.path + "/intermediate_results" if not args.final else args.path + "/final_results"
# First evaluate on Train set and afterwards on validation dataset
train_trainer = GridTrainer(config=config, train=True, batch_size=1)
train_trainer.eval(random_start=args.random if not args.final else False,
                   eval_length=args.steps if not args.final else len(train_trainer.dataset), save_file_path=out)
val_trainer = GridTrainer(config=config, train=False, batch_size=1)
val_trainer.eval(random_start=args.random if not args.final else False,
                 eval_length=args.steps if not args.final else len(val_trainer.dataset), save_file_path=out)
metric_logger = torch.load(config["save_files_path"] + "/metrics.pth.tar", map_location=device)

visualize_metric(metric_log=metric_logger, step_size=config["evaluation_steps"],
                 epoch=train_trainer.logger["epochs"][-1],
                 save_file_path=out)

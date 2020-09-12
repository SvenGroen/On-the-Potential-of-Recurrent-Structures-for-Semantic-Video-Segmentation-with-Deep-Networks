import argparse
import json
import sys
from src.gridtrainer import GridTrainer
"""
This script will be called by train.sge, which will be called by multiple_train.sge or by gridtrainers restart_script()
method.
It creates a GridTrainer object and trains the model.
"""


parser = argparse.ArgumentParser()
parser.add_argument("-cfg", "--config",
                    help="The Path to the configuration json for the model.\nShould include: model, ID, lr, batchsize,"
                         " num_epochs, scheduler_step_size, save_freq, save_path", type=str)
args = parser.parse_args()
if args.config is not None:
    print("Loading config: ", args.config)
    with open(args.config) as js:
        config = json.load(js)

trainer = GridTrainer(config)
trainer.train()
sys.stderr.write("\nEND OF TRAIN FILE\n")


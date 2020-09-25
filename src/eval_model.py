import argparse
import json
import sys
import torch
import pip
import pkg_resources
import platform, socket, re, uuid, json, logging, sys

from src.gridtrainer import GridTrainer
from src.utils.visualizations import visualize_metric
from subprocess import call

"""
Script is called to start the evaluation process.
Either final evaluation (eval on whole dataset) or intermediate eval (eval on part of dataset)

:param -stps: The number of evalution steps to be performed
:param -rdm: Should the start of the evaluation be at a random position
:param -pth: The path of the folder where the configuration file is located
:param -fnl: Indicator if a final evalutaion has to be perfomed
"""
# -stps 2 -rdm 0 -pth src/models/trained_models/YT_miniV3_3d/Deep+_mobile_wd0e+00bs6num_ep100ev2ID0 -fnl 1

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

# Load Config file
with open(args.path + "/train_config.json") as js:
    print("Loading config: ", args.path)
    config = json.load(js)


def getSystemInfo():
    """
    prints some system informations
    :return: dict with system informations
    """
    try:
        info = {}
        info['platform'] = platform.system()
        info['platform-release'] = platform.release()
        info['platform-version'] = platform.version()
        info['architecture'] = platform.machine()
        info['hostname'] = socket.gethostname()
        info['ip-address'] = socket.gethostbyname(socket.gethostname())
        info['mac-address'] = ':'.join(re.findall('..', '%012x' % uuid.getnode()))
        info['processor'] = platform.processor()
        info["GPU_Name"] = torch.cuda.get_device_name(0)
        info["Python_version"] = sys.version
        return json.dumps(info)
    except Exception as e:
        logging.exception(e)


# save some system information
with open(str(args.path + "/System_information.txt"), "w") as txt_file:
    txt_file.write("System Information:\n{}\n".format(getSystemInfo()))
    call("pip list >> {}/System_information.txt".format(args.path), shell=True)

load = torch.cuda.is_available()  # load data only if executed on grid.
out = args.path + "/intermediate_results" if not args.final else args.path + "/final_results_best_val"

# First evaluate on Train set and afterwards on validation dataset
train_trainer = GridTrainer(config=config, train=True, batch_size=1, load_from_checkpoint=load)
train_trainer.eval(random_start=args.random,
                   eval_length=args.steps if not args.final else len(train_trainer.dataset), save_file_path=out,
                   load_most_recent=load, checkpoint="best_checkpoint.pth.tar" if args.final else "checkpoint.pth.tar",
                   final=args.final)

# Evaluation on Validation set
val_trainer = GridTrainer(config=config, train=False, batch_size=1, load_from_checkpoint=load)
val_trainer.eval(random_start=args.random,
                 eval_length=args.steps if not args.final else len(val_trainer.dataset), save_file_path=out,
                 load_most_recent=load, checkpoint="best_checkpoint.pth.tar" if args.final else "checkpoint.pth.tar",
                 final=args.final)

# load the results that have just been made and visualize
path = config["save_files_path"] + "/metrics.pth.tar" if not args.final else out + "/metrics.pth.tar"
metric_logger = torch.load(path, map_location=device)

visualize_metric(metric_log=metric_logger, step_size=config["evaluation_steps"],
                 epoch=train_trainer.logger["epochs"][-1],
                 save_file_path=out)


# early stopping. Save best performaning model state
current = metric_logger["val"][-1]["Mean IoU"].avg
best = 0
sys.stderr.write(f"\ncurrent: {current}\n")
if not args.final:
    try:
        checkpoint = torch.load(train_trainer.config["save_files_path"] + "/best_checkpoint.pth.tar",
                                map_location=train_trainer.device)
        best = checkpoint["current_best"]
        sys.stderr.write(f"\nbest: {best}\n")
    except IOError:
        best = current

    sys.stderr.write(f"\ncurrent and  best: {current}\t{best}\n")

    if current >= best:
        train_trainer.logger["current_best"] = current
        torch.save(train_trainer.logger, train_trainer.config["save_files_path"] + "/best_checkpoint.pth.tar")

    current_train = metric_logger["train"][-1]["Mean IoU"].avg
    best_train = 0
    try:
        checkpoint_train = torch.load(train_trainer.config["save_files_path"] + "/best_checkpoint_train.pth.tar",
                                      map_location=train_trainer.device)
        best_train = checkpoint_train["current_best_train"]
        best_train = current_train if isinstance(best_train, list) else current_train
        sys.stderr.write(f"\nbest: {best_train}\n")
    except IOError:
        best_train = current_train

    if current_train >= best_train:
        train_trainer.logger["current_best"] = current_train
        torch.save(train_trainer.logger, train_trainer.config["save_files_path"] + "/best_checkpoint_train.pth.tar")

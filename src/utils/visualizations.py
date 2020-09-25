from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys

"""
Some usefull visualization scripts
"""
def visualize_metric(metric_log, step_size=2, epoch=0, save_file_path=None):
    """
    plots the metrics vs epoch for metrices saved in metric log
    :param metric_log: dict containing the metric logger information
    :param step_size: not used anymore
    :param epoch: not used anymore
    :param save_file_path: Where should the images be saved
    """
    for key in metric_log["train"][0]:
        if not key in ["curr_epoch", "hist"]:
            y = defaultdict(list)
            for i in range(len(metric_log["train"])):
                try:
                    cur_epoch = metric_log["train"][i]["curr_epoch"]
                    y["train"].append((cur_epoch, metric_log["train"][i][key].avg))
                    y["val"].append((cur_epoch, metric_log["val"][i][key].avg))
                except IndexError as e:
                    sys.stderr.write(f"\nError:\n{e}\n")
            print(y["train"], y["val"])
            plt.plot(*zip(*y["train"]), color='red', label="train")
            plt.plot(*zip(*y["val"]), color='blue', label="validation")
            plt.legend()
            plt.title('Average Train/Test {} score'.format(key))
            plt.xlabel('Epoch')
            plt.ylabel('Average {}'.format(key))
            plt.savefig(str(Path(save_file_path) / (key + ".jpg")))
            plt.close()


def visualize_logger(logger, path):
    """
    more simple version for visualization.
    Plots Lrs vs epoch, loss vs epoch, miou vs epoch
    :param logger: logger dict containing the lrs, loss and mious values
    :param path: where to save the results
    """
    def save_figure(values, y_label="", x_label="Epoch"):
        # saves values in a plot
        plt.plot(values)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.savefig(path + "/" + y_label + "_" + x_label + ".jpg")
        plt.close()

    save_figure(logger["lrs"], y_label="Learning Rate")
    save_figure(logger["loss"], y_label="Loss")
    save_figure(logger["mious"], y_label="Mean IoU")

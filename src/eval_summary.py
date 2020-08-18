import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import defaultdict
import re
import itertools
import os

import torch

model_path = Path("src/models/trained_models/YT_miniV2_no3d")

results = []
for folder in model_path.glob("*"):
    if not os.path.isdir(folder):
        continue
    eval_folder = folder / "final_results"

    metric_results = torch.load(folder / "metrics.pth.tar", map_location="cpu")
    with open(folder / "train_config.json") as js:
        config = json.load(js)
    pattern = re.search("V[1-9]", config["model"])
    label = "base"
    if "lstm" in config["model"]:
        label += " + LSTM"
    if "gru" in config["model"]:
        label += " + GRU"
    if pattern:
        label += pattern.group(0)

    for i in ["train", "val"]:
        metric_results[i][-1]["Label"] = label
        metric_results[i][-1]["Model"] = str(config["model"]) + "_" + str(config["track_ID"])
        metric_results[i][-1]["model_class"] = "mobile" if "mobile" in str(config["model"]) else "resnet"

    results.append(metric_results)

dfs =[]
for mode in ["train", "val"]:
    data = defaultdict(list)
    for i, category in enumerate(results[0][mode][-1].keys()):
        for model in results:
            if category != "curr_epoch":
                if isinstance(model[mode][-1][category], str):
                    data[category].append(model[mode][-1][category])
                elif torch.is_tensor(model[mode][-1][category].avg):
                    data[category].append(model[mode][-1][category].avg.item())
                else:
                    data[category].append(model[mode][-1][category].avg)
    df = pd.DataFrame.from_dict(data)
    df["Mode"] = mode
    dfs.append(df)
    print(df)

    # df["time_in_ms"] = df["time_taken"] * 1000

    mobile_df = df[df["model_class"] == "mobile"]
    resnet_df = df[df["model_class"] != "mobile"]

    categorys = ["Mean IoU", "Pixel Accuracy", "Per Class Accuracy", "Dice"]  # , "time_in_ms"
    plots = []

    # f, ax = plt.subplots(1, 2, figsize=(30, 10))
    # ax[0].bar(mobile_df["Label"], mobile_df["Jaccard"])
    # ax[1].bar(mobile_df["Label"], mobile_df["time_in_ms"])
    fontdict = {'fontsize': 30,
                'fontweight': 1,
                'verticalalignment': 'baseline',
                'horizontalalignment': "center"}
    tmp = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for name, df in [("mobile", mobile_df), ("resnet", resnet_df)]:
        f, ax = plt.subplots(2, 2, figsize=(50, 20))
        for pos, category in zip(tmp, categorys):
            ax[pos].bar(df["Label"], df[category])
            ax[pos].axhline(y=float(df[category][df["Label"] == "base"]), xmin=-0, xmax=1, color="r")
            ax[pos].set_title(category, fontdict=fontdict)
            plt.savefig(model_path / str(name + "_" + mode + ".png"))
    # plt.show()


result = pd.concat(dfs)
result.to_csv(model_path / "metric_results.csv")
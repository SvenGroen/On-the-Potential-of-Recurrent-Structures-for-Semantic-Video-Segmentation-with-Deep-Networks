import json
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch


def get_change(current, previous):
    if current == previous:
        return 0
    try:
        return ((current - previous) / previous) * 100.0
    except ZeroDivisionError:
        return float('inf')


def autolabel(rects, ax, percentage=True, base=None):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        if percentage:
            text = '{}%'.format(round(height * 100, 2))
        else:
            text = '{}'.format(height)
        if base is not None:
            change = get_change(height, base)
            if change == 0:
                text = '{}'.format(height)
            else:
                text = '{:+}'.format(round(change, 2)) + "%"

        ax.annotate(text,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    size=30,
                    ha='center', va='bottom')


model_path = Path("src/models/trained_models/yt_fullV1")
(model_path / "Metric_Results").mkdir(parents=True, exist_ok=True)

results = []
for folder in model_path.glob("*"):
    if "Metric" in str(folder):
        continue
    if not os.path.isdir(folder):
        continue
    eval_folder = folder / "final_results"

    metric_results = torch.load(folder / "metrics.pth.tar", map_location="cpu")
    with open(folder / "train_config.json") as js:
        config = json.load(js)
    pattern = re.search("V[1-9](_[1-9])?", config["model"])
    label = "base"
    if "lstm" in config["model"]:
        label += " + LSTM"
    if "gru" in config["model"]:
        label += " + GRU"
    if pattern:
        label += pattern.group(0)

    print(label)
    for i in ["train", "val"]:
        metric_results[i][-1]["Label"] = label
        metric_results[i][-1]["Model"] = str(config["model"]) + "_" + str(config["track_ID"])
        metric_results[i][-1]["model_class"] = "mobile" if "mobile" in str(config["model"]) else "resnet"

    results.append(metric_results)

dfs = []
for mode in ["train", "val"]:
    data = defaultdict(list)
    for i, category in enumerate(results[0][mode][-1].keys()):
        for model in results:
            if category not in ["curr_epoch", "hist"]:
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

    categorys = ["Mean IoU", "Pixel Accuracy", "Per Class Accuracy", "Dice", "num_params",
                 "Time_taken"]  # , "time_in_ms"
    metrics = ["Mean IoU", "Pixel Accuracy", "Per Class Accuracy", "Dice", "FIP",
               "FP"]  # <--------------------------------------------------------------------------------------------------------USE LATER WHEN NEW METRICS FILE
    plots = []

    # f, ax = plt.subplots(1, 2, figsize=(30, 10))
    # ax[0].bar(mobile_df["Label"], mobile_df["Jaccard"])
    # ax[1].bar(mobile_df["Label"], mobile_df["time_in_ms"])
    fontdict = {'fontsize': 30,
                'fontweight': 1,
                'verticalalignment': 'baseline',
                'horizontalalignment': "center"}
    tmp = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (1, 2)]

    for name, df in [("mobile", mobile_df), ("resnet", resnet_df)]:  # ,
        (model_path / "Metric_Results" / name).mkdir(parents=True, exist_ok=True)
        f, ax = plt.subplots(2, 3, figsize=(50, 20))
        for pos, category in zip(tmp, categorys):
            ax[pos].set_ylim([0, 1])
            ax[pos].bar(df["Label"], df[category])
            ax[pos].axhline(y=float(df[category][df["Label"] == "base"]), xmin=-0, xmax=1, color="r")
            ax[pos].set_title(category, fontdict=fontdict)
            f.savefig(model_path / "Metric_Results" / name / str(name + "_" + mode + ".png"))
        plt.close(fig=f)
    # plt.show()
    bars = []
    for name, df in [("mobile", mobile_df), ("resnet", resnet_df)]:
        df = df.round(4)
        for category in categorys:
            base_performance = float(df[category][df["Label"] == "base"])
            f_new, ax_new = plt.subplots(figsize=(50, 20))
            bar = ax_new.bar(df["Label"], df[category])
            ax_new.axhline(y=base_performance, xmin=-0, xmax=1, color="r")
            ax_new.set_title(category, fontdict=fontdict)
            autolabel(bar, ax=ax_new, base=base_performance)  # , base=base_performance
            plt.xticks(fontsize=25)
            plt.yticks(fontsize=25)
            f_new.savefig(model_path / "Metric_Results" / name / str(category + "_" + mode + ".png"))

            plt.close(fig=f_new)

result = pd.concat(dfs)
result.round(4).to_csv(model_path / "Metric_Results" / "metric_results.csv", sep=";")
result.round(4).reset_index().to_json(model_path / "Metric_Results" / "metric_results.json")

train = result["Mode"] == "train"
mobile = result["model_class"] == "mobile"
resnet = result["model_class"] == "resnet"
metrics = ["Mean IoU", "Pixel Accuracy", "Per Class Accuracy", "Dice"]
df_resnet = pd.DataFrame(result[resnet], columns=["Label", "Mode"] + metrics)
df_mobile = pd.DataFrame(result[mobile], columns=["Label", "Mode"] + metrics)
df_mobile = df_mobile.set_index(["Label", "Mode"]).unstack()
df_resnet = df_resnet.set_index(["Label", "Mode"]).unstack()
print(df_mobile.round(4).to_latex(index=True, multirow=True, multicolumn=True))
print(df_resnet.round(4).to_latex(index=True, multirow=True, multicolumn=True))

df_param = pd.DataFrame(result[train & mobile], columns=["Label", "model_class", "num_params"])
df_param["percentage"] = df_param["num_params"].pct_change()
df_param2 = pd.DataFrame(result[train & resnet], columns=["Label", "model_class", "num_params"])
df_param2["percentage"] = df_param2["num_params"].pct_change()
df_param = pd.concat([df_param, df_param2])
df_param = df_param.set_index(["Label", 'model_class']).unstack()
print(df_param.round(4).to_latex(index=True, multirow=True, multicolumn=True))


result["Time_taken"] = result["Time_taken"] * 100
df_time = pd.DataFrame(result[train & mobile], columns=["Label", "model_class", "Time_taken"])
df_time["percentage"] = df_time["Time_taken"].pct_change()
df_time2 = pd.DataFrame(result[train & resnet], columns=["Label", "model_class", "Time_taken"])
df_time2["percentage"] = df_time2["Time_taken"].pct_change()
df_time = pd.concat([df_time, df_time2])
df_time = df_time.set_index(["Label", 'model_class']).unstack()
print(df_time.round(4).to_latex(index=True, multirow=True, multicolumn=True, float_format=lambda x: f"{x}ms"))

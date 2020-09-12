import argparse
import json
import os
import sys
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
from src.gridtrainer import GridTrainer


def get_change(current, previous):
    """
    calculates the percentage change
    :param current: current value
    :param previous: base value
    :return: change of current to previous in percentage
    """
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


model_path = Path("src/models/trained_models/yt_fullV4")

rows = []
for folder in model_path.glob("*"):
    print("new folder:", folder)
    if "V5" in str(folder) or "V6" in str(folder) or "V7" in str(folder):
        continue
    try:
        with open(folder / "train_config.json") as js:
            config = json.load(js)
    except FileNotFoundError as e:
        print(e)
        continue
    pattern = re.search("V[1-9](_[1-9])?", config["model"])
    label = "base"
    if "lstm" in config["model"]:
        label += " + LSTM"
    if "gru" in config["model"]:
        label += " + GRU"
    if pattern:
        label += pattern.group(0)
    try:
        train_trainer = GridTrainer(config=config, train=True, batch_size=1,
                                    load_from_checkpoint=True)
        train_trainer.dataset.apply_transform = False
    except Exception as e:
        print(e)
        continue


    time_taken = train_trainer.time_and_image_eval()
    mode = "resnet" if "resnet" in config["model"] else "mobile"
    rows.append([label, mode, time_taken])
print(rows)
df = pd.DataFrame(rows, columns=["Label", "backbone", "time"])
print(df)
mobile = df[df["backbone"] == "mobile"]["time"]
resnet = df[df["backbone"] == "resnet"]["time"]
index = df[df["backbone"] == "mobile"]["Label"]
df = pd.DataFrame({'mobile': list(mobile),
                   'resnet': list(resnet)}, index=index)
df.to_csv(model_path / "time_results.csv")
ax = df.plot.bar(rot=0)
autolabel(ax)
fig = ax.get_figure()
fig.savefig(model_path / "time_results.png")
print("all done")

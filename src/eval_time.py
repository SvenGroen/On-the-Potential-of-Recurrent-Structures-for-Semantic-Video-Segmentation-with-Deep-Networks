import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
from src.gridtrainer import GridTrainer

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
df = pd.DataFrame(rows, columns=["Label", "backbone", "time"])
mobile = df[df["backbone"] == "mobile"]["time"]
resnet = df[df["backbone"] == "resnet"]["time"]
index = df[df["backbone"] == "mobile"]["Label"]
df = pd.DataFrame({'mobile': list(mobile),
                   'resnet': list(resnet)}, index=index)
df.to_csv(model_path / "time_results.csv")
ax = df.round(3).plot.bar(rot=0)
for p in ax.patches:
    ax.annotate(np.round(p.get_height(), decimals=2), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                va='center', xytext=(0, 10), textcoords='offset points')

ax.set_xlabel("Model Versions")
ax.set_ylabel("Time (ms)")
ax.set_title("Time (ms) for different models and backbones")
fig = ax.get_figure()
fig.savefig(model_path / "time_results.png")
print("all done")

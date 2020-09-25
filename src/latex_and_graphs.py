import json
import re
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
visualization of certain graph for better readability in the thesis + prints latex tables from pandas Dataframes.
"""



def strip_base(x):
    """
    removes the "base" part from a given string
    :param x: input string
    :return: output string without "base"
    """
    if x == "base":
        return x
    else:
        return x.lstrip("base")


# visualization of time results
folder_name = "yt_fullV4/"
out = "src/models/trained_models/" + folder_name
df = pd.read_csv(out + "/time_results.csv")

df["Label"] = df["Label"].map(lambda x: strip_base(x))
df.set_index("Label", inplace=True)
ax = df.round(3).plot.bar(rot=0, figsize=(15, 8))
for p in ax.patches:
    ax.annotate(np.round(p.get_height(), decimals=2), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                va='center', xytext=(0, 10), textcoords='offset points', size=12)
fontdict = {'fontsize': 15,
            'fontweight': 1,
            'verticalalignment': 'baseline',
            'horizontalalignment': "center"}
ax.set_xlabel("", fontdict=fontdict)
ax.set_ylabel("Time (ms)", fontdict=fontdict)
ax.set_title("Time (ms) for different models and backbones", fontdict=fontdict)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(12)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(12)
ax.legend(fontsize=12)
plt.savefig(out + "/time_results.png")

# train vs test miou plot
folder_name = folder_name + "Metric_Results"
out = "src/models/trained_models/" + folder_name
df = pd.read_csv(out + "/metric_results.csv", delimiter=";")
df["Label"] = df["Label"].map(lambda x: strip_base(x))

df = df[df["model_class"] == "resnet"]

df = df[['Mean IoU', "Label", "Mode"]]
train = df[df["Mode"] == "train"]

test = df[df["Mode"] == "val"]
index = df[df["Mode"] == "train"]["Label"]

df = pd.DataFrame({'train': list(train['Mean IoU']),
                   'validation': list(test['Mean IoU'])}, index=index)

ax = df.round(4).plot.bar(rot=0, figsize=(15, 8))
for p in ax.patches:
    ax.annotate(str(np.round(p.get_height()*100, decimals=1))+"%", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                va='center', xytext=(0, 10), textcoords='offset points')
ax.set_ylim([0, 1])
ax.set_xlabel("Model Versions")
ax.set_ylabel("Mean IoU")
ax.set_title("Mean IoU for Models with ResNet-50 Backbone",fontdict=fontdict)
ax.legend(loc=2)
plt.savefig(out + "/MioU_train_test")

# same for mobilenet
out = "src/models/trained_models/" + folder_name
df = pd.read_csv(out + "/metric_results.csv", delimiter=";")
df["Label"] = df["Label"].map(lambda x: strip_base(x))

df = df[df["model_class"] == "mobile"]

df = df[['Mean IoU', "Label", "Mode"]]
train = df[df["Mode"] == "train"]

test = df[df["Mode"] == "val"]
index = df[df["Mode"] == "train"]["Label"]

df = pd.DataFrame({'train': list(train['Mean IoU']),
                   'validation': list(test['Mean IoU'])}, index=index)

ax = df.round(4).plot.bar(rot=0, figsize=(15, 8))
for p in ax.patches:
    ax.annotate(str(np.round(p.get_height()*100, decimals=1))+"%", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                va='center', xytext=(0, 10), textcoords='offset points')
ax.set_ylim([0, 1])
ax.set_xlabel("Model Versions")
ax.set_ylabel("Mean IoU")
ax.set_title("Mean IoU for Models with Mobilenet Backbone",fontdict=fontdict)
ax.legend(loc=2)
plt.savefig(out + "/MioU_train_test_mobile")




# plot latex tables:
folder_name = "yt_fullV4/"
out = "src/models/trained_models/" + folder_name
df = pd.read_csv(out + "/time_results.csv")

df["Label"] = df["Label"].map(lambda x: strip_base(x))

df = df.set_index("Label")
print(df.round(2).to_latex(index=True, multirow=True, multicolumn=True))

# Batch sizes and Weight decay plots but with larger labels for better readability
folder_name = "/LR_Tests_weight_decay"
out = Path("src/models/" + folder_name)
for folder in out.glob("*"):
    print(folder)
    with open(folder / "results.json") as js:
        data = json.loads(js.read())
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    for wd, hist in zip(data["wds"], data["hists"]):
        lrs = hist["lr"]
        losses = hist["loss"]
        ax.plot(lrs, losses, label=str(wd))
        ax.set_xscale("log")
        size = 16
        legend = ax.legend(loc=3, prop={"size": size}, title="Weight Decay:")
        legend.get_title().set_fontsize(str(size))
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(size)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(size)
        plt.xlabel("Learning Rate", size=size)
        plt.ylabel("Loss", size=size)
        plt.savefig(folder / "lr_analysis_large.png")
    plt.close()

folder_name = "/LR_Tests_batch_size"
out = Path("src/models/" + folder_name)
for folder in out.glob("*"):
    print(folder)
    with open(folder / "results.json") as js:
        data = json.loads(js.read())
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    for wd, hist in zip(data["wds"], data["hists"]):
        lrs = hist["lr"]
        losses = hist["loss"]
        ax.plot(lrs, losses, label=str(wd))
        ax.set_xscale("log")
        size = 16
        legend = ax.legend(loc=3, prop={"size": size}, title="Batch Size:")
        legend.get_title().set_fontsize(str(size))
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(size)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(size)
        plt.xlabel("Learning Rate", size=size)
        plt.ylabel("Loss", size=size)
        plt.savefig(folder / "lr_analysis_large.png")
    plt.close()

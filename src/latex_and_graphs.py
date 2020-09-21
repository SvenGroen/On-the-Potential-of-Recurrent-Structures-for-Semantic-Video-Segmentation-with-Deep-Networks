import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
folder_name ="yt_fullV4_probably_use/"
out = "src/models/trained_models/" + folder_name
df = pd.read_csv(out +"/time_results.csv")
print(df)
pattern = re.search("V[1-9](_[1-9])?", config["model"])
label = "base"
if "lstm" in config["model"]:
    label += " + LSTM"
if "gru" in config["model"]:
    label += " + GRU"
if pattern:
    label += pattern.group(0)


df.set_index("Label", inplace=True)

ax = df.round(3).plot.bar(rot=0, figsize=(15, 8))
for p in ax.patches:
    ax.annotate(np.round(p.get_height(), decimals=2), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                va='center', xytext=(0, 10), textcoords='offset points')

ax.set_xlabel("Model Versions")
ax.set_ylabel("Time (ms)")
ax.set_title("Time (ms) for different models and backbones")
plt.savefig(out + "/time_results.png")

folder_name= folder_name +"Metric_Results_use_for_final_evaluation"
out = "src/models/trained_models/" + folder_name
df = pd.read_csv(out +"/metric_results.csv", delimiter=";")

df = df[df["model_class"] == "resnet"].sort_values(by=["Label"])

df = df[['Mean IoU', "Label","Mode"]]

train = df[df["Mode"] == "train"].sort_values(by=["Label"])

test = df[df["Mode"] == "val"].sort_values(by=["Label"])
index = df[df["Mode"] == "train"]["Label"].sort_values()

df = pd.DataFrame({'train': list(train['Mean IoU']),
                   'validation': list(test['Mean IoU'])}, index=index)

ax = df.round(3).plot.bar(rot=0, figsize=(15, 8))
for p in ax.patches:
    ax.annotate(np.round(p.get_height(), decimals=2), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                va='center', xytext=(0, 10), textcoords='offset points')
ax.set_ylim([0,1])
ax.set_xlabel("Model Versions")
ax.set_ylabel("Mean IoU")
ax.set_title("Mean IoU for Models with ResNet-50 Backbone")
ax.legend(loc=2)
print(df)




plt.show()


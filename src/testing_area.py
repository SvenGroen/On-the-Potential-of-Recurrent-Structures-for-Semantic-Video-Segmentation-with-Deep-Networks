import pandas as pd

metrics = ["Mean IoU", "Pixel Accuracy", "Per Class Accuracy", "Dice"]

results = pd.read_csv("src/models/trained_models/yt_fullV1/Metric_Results/metric_results.csv", delimiter=";")
mobile = results["model_class"] == "mobile"
resnet = results["model_class"] == "resnet"
train = results["Mode"] == "train"


# df = pd.DataFrame(df[mobile], columns=["Label", "Mode"] + metrics)
# df = df.set_index(["Label",'Mode']).unstack()
def get_change(current, previous):
    if current == previous:
        return 100
    try:
        return ((current - previous) / previous) * 100.0 +100
    except ZeroDivisionError:
        return float('inf')


df_param = pd.DataFrame(results[train & mobile], columns=["Label", "model_class", "num_params"])

df_param["percentage"] = df_param["num_params"].pct_change()

df_param2 = pd.DataFrame(results[train & resnet], columns=["Label", "model_class", "num_params"])
df_param2["percentage"] = df_param2["num_params"].pct_change()

df_param = pd.concat([df_param, df_param2])
df_param = df_param.set_index(["Label", 'model_class']).unstack()


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:.3f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


# print(human_format(2), human_format(200), human_format(2000))
results["Time_taken"] = results["Time_taken"] * 100
df_time = pd.DataFrame(results[train & mobile], columns=["Label", "model_class", "Time_taken"])
df_time["percentage"] = df_time["Time_taken"].pct_change()
df_time2 = pd.DataFrame(results[train & resnet], columns=["Label", "model_class", "Time_taken"])
df_time2["percentage"] = df_time2["Time_taken"].pct_change()
df_time = pd.concat([df_time, df_time2])
df_time = df_time.set_index(["Label", 'model_class']).unstack()
print(df_time.round(4).to_latex(index=True, multirow=True, multicolumn=True, float_format=lambda x: f"{x}ms"))
import json
from pathlib import Path
import sys
import os

"""
This script allows to create an array of jobs for different models with (if wanted) multiple hyperparameters.
models needs to be a list containing at least one of these models:

"Deep_mobile_lstmV4", "Deep_resnet50_lstmV4", "Deep_mobile_gruV4", "Deep_resnet50_gruV4",
"Deep_mobile_lstmV3", "Deep_mobile_gruV3", "Deep_resnet50_lstmV3", "Deep_resnet50_gruV3",
"Deep_mobile_lstmV2", "Deep_mobile_gruV2", "Deep_resnet50_lstmV2", "Deep_resnet50_gruV2",
"Deep_mobile_lstmV1", "Deep_mobile_gruV1", "Deep_resnet50_lstmV1", "Deep_resnet50_gruV1",
"Deep+_resnet50", "Deep+_mobile", "Deep_mobile_lstmV5", "Deep_mobile_lstmV6", "Deep_mobile_lstmV7"
"""
sys.stderr.write("start writing configs\n")

models = ["Deep_mobile_lstmV4", "Deep_resnet50_lstmV4", "Deep_mobile_gruV4", "Deep_resnet50_gruV4",
          "Deep_mobile_lstmV3", "Deep_mobile_gruV3", "Deep_resnet50_lstmV3", "Deep_resnet50_gruV3",
          "Deep_mobile_lstmV2", "Deep_mobile_gruV2", "Deep_resnet50_lstmV2", "Deep_resnet50_gruV2",
          "Deep_mobile_lstmV1", "Deep_mobile_gruV1", "Deep_resnet50_lstmV1", "Deep_resnet50_gruV1",
          "Deep+_resnet50", "Deep+_mobile",
          "Deep_mobile_lstmV5", "Deep_mobile_lstmV6", "Deep_resnet50_lstmV5", "Deep_resnet50_lstmV6",
          "Deep_mobile_gruV5", "Deep_mobile_gruV6", "Deep_resnet50_gruV5", "Deep_resnet50_gruV6"]

batch_sizes = 8
num_epochs = 50
loss = ["SoftDice"]  # "CrossEntropy" or "CrossDice"
eval_steps = 2

config_paths = []
models_name = []
configs = []

# create a config file for each combination
for model in models:

    for i in range(len(loss)):
        config = {
            "model": model,
            "batch_size": batch_sizes,
            "num_epochs": num_epochs,
            "evaluation_steps": eval_steps,
            "loss": loss[i],
            "save_folder_path": "src/models/trained_models/yt_fullV5/"}
        configs.append(config)

# start to call a job for each config file
for i, config in enumerate(configs):
    # print(config)
    from subprocess import call

    config["track_ID"] = i
    unique_name = "ID" + str(config["track_ID"]) + config["model"] \
                  + "_bs" + str(config["batch_size"]) + "num_ep" \
                  + str(config["num_epochs"]) + "ev" + str(config["evaluation_steps"])
    config["unique_name"] = unique_name
    config["save_files_path"] = Path(config["save_folder_path"]) / unique_name

    config["save_files_path"].mkdir(parents=True, exist_ok=True)  # create folder to save results
    (config["save_files_path"] / "log_files").mkdir(parents=True, exist_ok=True)
    config["save_files_path"] = str(config["save_files_path"])

    with open(str(config["save_files_path"] + "/train_config.json"), "w") as js:  # save learn config
        json.dump(config, js)

    vRam = "3.4G"
    option = " -l hostname=!\"(*cippy25*|*cippy19*)\""
    if "V5" in config["model"]:
        vRam = "3.8G"
        option = " -l hostname=!\"(*cippy25*|*cippy19*|*vr*|*grid*)\""
    if "V3" in config["model"] or "V6" in config["model"]:
        vRam = "7.8G"
    if "V4" in config["model"]:
        vRam = "10G"

    job_name = "id" + str(config["track_ID"]).zfill(2) + config["model"]
    recallParameter = 'qsub -N ' + job_name \
                      + ' -l nv_mem_free=' + vRam \
                      + option \
                      + " -o " + config["save_files_path"] + "/log_files/" + job_name + ".o$JOB_ID" \
                      + " -e " + config["save_files_path"] + "/log_files/" + job_name + ".e$JOB_ID" \
                      + ' -v CFG=' + str(config["save_files_path"]) + "/train_config.json" + ' src/train.sge'

    print(recallParameter, "\t" + os.getcwd(), "\n")
    call(recallParameter, shell=True)

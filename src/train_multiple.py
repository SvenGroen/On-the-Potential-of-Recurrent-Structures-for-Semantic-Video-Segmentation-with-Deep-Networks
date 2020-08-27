import json
from pathlib import Path
import sys
import os

# models = ["Deep+_mobile", "Deep_mobile_lstmV1", "Deep_mobile_lstmV2", "Deep_mobile_lstmV3", "Deep_mobile_lstmV4",
# "Deep_mobile_lstmV5_1", "Deep_mobile_lstmV5_2", "Deep_mobile_gruV1", "Deep_mobile_gruV2", "Deep_mobile_gruV3",
# "Deep_mobile_gruV4", "Deep+_resnet50", "Deep_resnet50_lstmV1", "Deep_resnet50_lstmV2", "Deep_resnet50_lstmV3",
# "Deep_resnet50_lstmV4", "Deep_resnet50_gruV1", "Deep_resnet50_gruV2", "Deep_resnet50_gruV3", "Deep_resnet50_gruV4"]

print(os.getcwd())

sys.stderr.write("start writing configs\n")

models = ["Deep+_mobile", "Deep_mobile_lstmV4", "Deep_mobile_gruV4", "Deep_mobile_gruV1", "Deep_mobile_gruV2",
          "Deep_mobile_gruV3", "Deep_mobile_lstmV1", "Deep_mobile_lstmV2_1", "Deep_mobile_lstmV2_2",
          "Deep_mobile_lstmV3","Deep_mobile_lstmV5_2", "Deep+_resnet50", "Deep_resnet50_lstmV1", "Deep_resnet50_lstmV2_2",
          "Deep_resnet50_lstmV3", "Deep_resnet50_lstmV4", "Deep_resnet50_gruV1", "Deep_resnet50_gruV2",
          "Deep_resnet50_gruV3", "Deep_resnet50_gruV4"]
# models = [ "Deep_resnet50_lstmV1", "Deep_resnet50_lstmV2", "Deep_resnet50_lstmV3",
# "Deep_resnet50_lstmV4", "Deep_resnet50_gruV1", "Deep_resnet50_gruV2", "Deep_resnet50_gruV3", "Deep_resnet50_gruV4"]

models= ["Deep+_mobile"]
batch_sizes = 8
num_epochs = 100
loss = ["SoftDice"]  # "CrossEntropy"
wds = [0]
eval_steps = 15

config_paths = []
models_name = []
configs = []

for model in models:

    for i in range(len(loss)):
        config = {}
        config["model"] = model
        config["weight_decay"] = wds[i]
        config["batch_size"] = batch_sizes
        config["num_epochs"] = num_epochs
        config["evaluation_steps"] = eval_steps
        config["loss"] = loss[i]
        config["save_folder_path"] = "src/models/trained_models/yt_fullV2/"

        # print(config) 
        configs.append(config)

for i, config in enumerate(configs):
    # print(config)
    from subprocess import call

    config["track_ID"] = i
    unique_name = config["model"] + "_wd" + format(config["weight_decay"], ".0e") + "bs" + str(
        config["batch_size"]) + "num_ep" \
                  + str(config["num_epochs"]) + "ev" + str(config["evaluation_steps"]) + "ID" + str(config["track_ID"])
    config["unique_name"] = unique_name
    config["save_files_path"] = Path(config["save_folder_path"]) / unique_name

    config["save_files_path"].mkdir(parents=True, exist_ok=True)  # create folder to save results
    (config["save_files_path"] / "log_files").mkdir(parents=True, exist_ok=True)
    config["save_files_path"] = str(config["save_files_path"])

    with open(str(config["save_files_path"] + "/train_config.json"), "w") as js:  # save learn config
        json.dump(config, js)

    vRam = "9G"
    if "V4" in config["model"]:
        vRam = "11G"

    job_name = "id" + str(i).zfill(2) + config["model"]
    recallParameter = 'qsub -N ' + job_name \
                      + ' -l nv_mem_free=' + vRam \
                      + " -o " + config["save_files_path"] + "/log_files/" + job_name + ".o$JOB_ID" + " -e " \
                      + config["save_files_path"] + "/log_files/" + job_name + ".e$JOB_ID" \
                      + ' -v CFG=' + str(config["save_files_path"]) + "/train_config.json" + ' src/train.sge'
    '''
    = 'qsub -N ' + job_name \
                      + ' -l nv_mem_free=' + vRam \
                      + " -o " + config["save_files_path"] + "/log_files/"+job_name+".o$JOB_ID y -j" + " -e " \
                      + config["save_files_path"] + "/log_files/"+job_name+".e$JOB_ID y -j" \
                      + ' -v CFG=' + str(config["save_files_path"]) + "/train_config.json" + ' src/train.sge'
    '''
    print(recallParameter, "\t" + os.getcwd(), "\n")
    call(recallParameter, shell=True)

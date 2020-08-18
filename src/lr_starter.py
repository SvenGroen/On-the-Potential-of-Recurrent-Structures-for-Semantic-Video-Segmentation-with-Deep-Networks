import json
from pathlib import Path
from subprocess import call

import torch

models = ["Deep+_mobile", "Deep_mobile_lstmV1", "Deep_mobile_lstmV2_1","Deep_mobile_lstmV2_2", "Deep_mobile_lstmV3", "Deep_mobile_lstmV4",
          "Deep_mobile_lstmV5_1", "Deep_mobile_lstmV5_2", "Deep_mobile_gruV1", "Deep_mobile_gruV2", "Deep_mobile_gruV3",
          "Deep_mobile_gruV4", "Deep+_resnet50", "Deep_resnet50_lstmV1", "Deep_resnet50_lstmV2_1","Deep_resnet50_lstmV2_2", "Deep_resnet50_lstmV3",
          "Deep_resnet50_lstmV4", "Deep_resnet50_gruV1", "Deep_resnet50_gruV2", "Deep_resnet50_gruV3",
          "Deep_resnet50_gruV4"]
configs = []
for model in models:
    config = {}
    config["model"] = model
    config["weight_decay"] = 0
    config["batch_size"] = 6
    config["num_epochs"] = 100
    config["evaluation_steps"] = 2
    config["loss"] = "CrossDice"
    config["save_folder_path"] = "src/models/LR_Tests_update/bs_" + str(config["batch_size"])

    # print(config)
    configs.append(config)

for i, config in enumerate(configs):
    config["track_ID"] = i
    unique_name = config["model"] + "bs" + str(config["batch_size"]) + "num_iter" \
                  + str(config["num_epochs"]) + "ID" + str(config["track_ID"])
    config["unique_name"] = unique_name
    config["save_files_path"] = Path(config["save_folder_path"]) / unique_name
    config["save_files_path"].mkdir(parents=True, exist_ok=True)  # create folder to save results
    (config["save_files_path"] / "log_files").mkdir(parents=True, exist_ok=True)
    config["save_files_path"] = str(config["save_files_path"])

    with open(str(config["save_files_path"] + "/train_config.json"), "w") as js:  # save learn config
        json.dump(config, js)

    job_name = "LR" + str(config["track_ID"]).zfill(2)
    VRAM = "9G"
    recallParameter = "qsub -N " + job_name + config["model"] + ' -l nv_mem_free=' + str(VRAM) \
                      + " -o " + str(config["save_files_path"]) + "/log_files/" + job_name + ".o$JOB_ID" \
                      + " -e " + str(config["save_files_path"]) + "/log_files/" + job_name + ".e$JOB_ID" \
                      + " -v CFG=" + str(config["save_files_path"] + "/train_config.json") + " src/lr_finder.sge"
    if torch.cuda.is_available():
        call(recallParameter, shell=True)
    else:
        print(recallParameter)
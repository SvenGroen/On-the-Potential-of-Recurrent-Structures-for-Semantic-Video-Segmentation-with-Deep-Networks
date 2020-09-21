import json
from pathlib import Path

model_path = Path("src/models/trained_models/yt_fullV4")
for i, path in enumerate(model_path.glob("*")):
    from subprocess import call
    if not "Deep_resnet50_gruV3" in str(path):
        continue
    # if not ("resnet50_gruV2" in str(path) or "mobile_lstmV4" in str(path) or "mobile_gruV1" in str(path)):
    #     continue
    if "Metric" in str(path) or "V7" in str(path): #or "V6" in str(path) or "V5" in str(path)
        continue 
    with open(path / "train_config.json") as js:
        config = json.load(js)

    job_name = "EV" + str(config["track_ID"]).zfill(2) + "final"
    # + " -l h_rt=04:29:00"\*grid
    recallParameter = "qsub -N " + job_name + config["model"] + ' -l hostname=*' \
                      + " -l h_rt=15:29:00" \
                      + " -o " + str(config["save_files_path"]) + "/log_files/" + job_name + ".o$JOB_ID" \
                      + " -e " + str(config["save_files_path"]) + "/log_files/" + job_name + ".e$JOB_ID" \
                      + " -v STPS=" + str(1) \
                      + " -v RDM=" + str(int(False)) \
                      + " -v FNL=" + str(int(True)) \
                      + " -v PTH=" + str(config["save_files_path"]) + " src/eval_model.sge"

    # print(recallParameter)
    call(recallParameter, shell=True)

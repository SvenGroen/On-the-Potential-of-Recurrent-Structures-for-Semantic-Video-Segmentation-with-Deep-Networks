import json
from pathlib import Path

"""
Allows to start an array of (final) evaluation jobs.
Specify the model_path where all models are located that you want to evaluate.
Some model versions can be excluded like this:

exclude resnet gruv4 and all v4 models:
if not "Deep_resnet50_gruV3" in str(path) or "V4" in str(path):
    continue
"""

model_path = Path("src/models/trained_models/yt_fullV4")
for i, path in enumerate(model_path.glob("*")):
    from subprocess import call
    if "Metric" in str(path):
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
    call(recallParameter, shell=True)

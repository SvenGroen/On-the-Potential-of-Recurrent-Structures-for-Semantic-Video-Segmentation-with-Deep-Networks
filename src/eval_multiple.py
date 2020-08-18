import json
from pathlib import Path

model_path = Path("src/models/trained_models/YT_miniV3_3d")
for i, path in enumerate(model_path.glob("*")):
    from subprocess import call

    with open(path / "train_config.json") as js:
        config = json.load(js)

    job_name = "IE" + str(config["track_ID"]).zfill(2) + "final"
    VRAM = 3.8
    recallParameter = "qsub -N " + job_name + config["model"] + ' -l nv_mem_free=' + str(VRAM) \
                      + " -o " + str(config["save_files_path"]) + "/log_files/" + job_name + ".o$JOB_ID" \
                      + " -e " + str(config["save_files_path"]) + "/log_files/" + job_name + ".e$JOB_ID" \
                      + " -v STPS=" + str(1) \
                      + " -v RDM=" + str(int(False)) \
                      + " -v FNL=" + str(int(True)) \
                      + " -v PTH=" + str(config["save_files_path"]) + " src/eval_model.sge"

    # print(recallParameter)
    call(recallParameter, shell=True)

import argparse
import json
import sys
import os
from pathlib import Path
from src.gridtrainer import GridTrainer

# print(os.getcwd())
# sys.stderr.write("\n" + os.getcwd() + "\n")



parser = argparse.ArgumentParser()
parser.add_argument("-cfg", "--config",
                    help="The Path to the configuration json for the model.\nShould include: model, ID, lr, batchsize,"
                         " num_epochs, scheduler_step_size, save_freq, save_path", type=str)
args = parser.parse_args()
if args.config is not None:
    print("Loading config: ", args.config)
    with open(args.config) as js:
        # print("Loading config: ", args.config)
        config = json.load(js)

trainer = GridTrainer(config)
trainer.train()

'''
else:
    print("Test cfg used")
    model = "Deep_mobile_gruV1"
    weight_decay = 1e-8
    batch_size = 6
    track_id = 00
    num_epochs = 2
    eval_steps = 2
    unique_name = model + "_wd" + format(weight_decay, ".0e") + "bs" + str(batch_size) + "num_ep" \
                  + str(num_epochs) + "ev" + str(eval_steps) + "ID" + str(track_id)
    config = {
        "save_folder_path": "src/models/trained_models/testing",
        "save_files_path": "src/models/trained_models/testing/" + unique_name,
        "model": model,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "evaluation_steps": eval_steps,
        "loss": "CrossDice",
        "track_ID": track_id
    }
    Path(config["save_files_path"]).mkdir(parents=True, exist_ok=True)
    with open(str(Path(config["save_files_path"]) / "train_config.json"), "w") as js:  # save learn config
        json.dump(config, js)
    trainer = GridTrainer(config)
    trainer.train()
'''

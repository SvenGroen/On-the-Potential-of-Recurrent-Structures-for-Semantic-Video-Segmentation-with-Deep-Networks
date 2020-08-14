import json
from pathlib import Path

from torch import optim
from torch_lr_finder import LRFinder
from torch.utils.data import DataLoader
from src.dataset.YT_Greenscreen import YT_Greenscreen
from src.gridtrainer import GridTrainer

model = "Deep+_mobile"
weight_decay = 1e-8
batch_size = 6
track_id = 00
num_epochs = 2
eval_steps = 2
unique_name = model + "_wd" + format(weight_decay, ".0e") + "bs" + str(batch_size) + "num_ep" \
              + str(num_epochs) + "ev" + str(eval_steps) + "ID" + str(track_id)
config = {
    "save_folder_path": "models/trained_models/testing",
    "save_files_path": "models/trained_models/testing/" + unique_name,
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

model = trainer.model
criterion = trainer.criterion
val_dataset = YT_Greenscreen(train=False, start_index=0, batch_size=trainer.config["batch_size"])
val_loader = DataLoader(val_dataset, val_dataset.batch_size, shuffle=False)
optimizer = optim.Adam(model.parameters(), lr=1e-8, weight_decay=0)
optimizer.state_dict()["param_groups"][0]["lr"] = 1e-7
lr_finder = LRFinder(model, optimizer, criterion, device="cpu")
lr_finder.range_test(trainer.loader,val_loader=val_loader, end_lr=1e-1, num_iter=10)
lr_finder.plot(skip_start=0, skip_end=0)
lr_finder.reset()

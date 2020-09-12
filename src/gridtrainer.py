import json
import sys
import torch
import cv2
import numpy as np
import torchvision.transforms as T
import random
from collections import defaultdict
from pathlib import Path
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset.YT_Greenscreen import YT_Greenscreen
from src.utils import initiator, time_logger, AverageMeter, stack, eval_metrics, fast_hist, jaccard_index
from src.utils.visualizations import visualize_logger

from src.utils.metrics import get_gpu_memory_map


class GridTrainer:
    """
    The GridTrainer object holds relevant parameters for training. It is supposed to be used in conjunction with a
    Grid Computing Network where script restart due to time limitations are necessary.
    It allows to restart the training process and start an intermediate evaluation by calling the train.sge or
    eval_model.sge script.
    This class is especially useful for multiple grid jobs and separate computers.

    Uses Adam Optimizer for training, learning rate and weight decay values have been predetermined
    based on a lr range test. They can be changed for each model in initiator.py.
    As a lr scheduler pytorches CyclicLR scheduler is choosen with a "triangular2" strategy and a stepsize of
    7 * len(self.loader)). To change please change self.scheduler and make sure that self.scheduler.step() is in the
    correct position.

    Creates multiple files:
    - checkpoint.pth.tar the most recent checkpoint
    - metrics.pth.tar saves the evaluation metric results
    - Learning Rate_Epoch.jpg visualization of the lr scheduler
    - Loss_Epoch.jpg visualization of the loss

    - /log_files/ folder that saves the log files created by the grid.
    - /intermediate_results/ folder that saves results of intermediate evaluation
    - /final_results/ folder that saves the results after evaluation on the whole dataset

    Important Attributes and instantiations:
        src.utils.initiator.initiate_model(config) returns the model and hyperparameter training values
        config: stores the train config as a dict
        criterion: the loss function used for training
        logger: dictionary of lists. Logs the training progress and model status and enables
        a complete restart of training at a certain position in case the script is restarted.
        metric_logger: like logger but is used to store the evaluation results.
        time_logger: TimeLogger object. Keeps track of the time the script is running and elicits a "restart signal",
        if the Grid kill time is soon to be exceeded.
        dataset: the Youtube Greenscreen dataset.
        loader: DataLoader object from pytorch
        optimizer: Adam optimizer
        scheduler: Cyclic LR with step_size_up of 7* len(loader)
        _Restart: Flag that indicates if the script was just restarted (necessary for initiating certain values).
        cur_idx: current idx of the batch loop. Necessary for the restart


    :param config: dict:
        Configuration parameters stored in a dictionary. Following keys must be filled out:
        "batch_size":           int:    will be used as the batch size if no batch size
                                        is provided in the initialization of the class.
        "tack_id":              int:    is used as a unique id to identify the model and the job. Will be displayed
                                        in the job name quickly identify the model.
        "num_epochs":           int:    The number of epochs that should be trained.
        "save_files_path":      str:    Path to folder where files should be created. The config file needs to be saved
                                        at this location with the name "train_config.json". (Can be automatically
                                        created using the multiple_train.sge/train_multiple.py script).
        "model":                str:    Model Version that should be used. Currently supported:
                                        - Deep+_(resnet50)|(mobile) for the base model
                                        - Deep_(resnet50)|(mobile)(lstm|gru)V[1-4] for the Recurrent unit alternations
        "loss":                 str:    Which loss function should be used: Currently supported:
                                        - "SoftDice"
                                        - "CrossEntropy"
                                        - "CrossDice" ((softdice + Crossentropy) / 2)
                                        - "Focal" (probably needs parameter adjustment,
                                                   default values did not enable good learning)
        "evaluation_steps"      int:    in what interval should a evaluation occur.

    :param train: boolean:
        If True, the training dataset will be used, else the testing dataset will be used.
        Default: True
    :param batch_size:
        Overwrites the batch_size in the config if given.
        (Useful if a different batch size should be used for evaluation)
        Default: None
    :param load_from_checkpoint: boolean:
        If True, the last known checkpoint (config["save_files_path"] + "/checkpoint.pth.tar") will be loaded to resume
        training after script restart.
        Default: True
    :param seed: int > 0:
        Sets seed for reproducibility
    """

    def __init__(self, config: dict, train: bool = True, batch_size=None, load_from_checkpoint: bool = True, seed=0):
        """
        Please see help(GridTrainer) for more information.
        """
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.model, self.weight_decay, self.lr_boundarys, self.detach_interval = initiator.initiate_model(self.config)
        # self.lr_boundarys = (self.lr_boundarys[0]*0.1,self.lr_boundarys[1]*0.1)
        self.model = self.model.to(self.device)
        self.criterion = initiator.initiate_criterion(self.config)
        # self.logger = initiator.initiate_logger(self.lr_boundarys[0])
        self.logger = defaultdict(list)
        self.metric_logger = defaultdict(list)

        self.batch_size = self.config["batch_size"] if batch_size is None else batch_size
        self.time_logger = time_logger.TimeLogger(restart_time=60 * 60 * 1.19)  # 60 * 60 * 1.2
        self.dataset = YT_Greenscreen(train=train, start_index=0,
                                      batch_size=self.batch_size, seed=self.seed)
        self.test = not train
        self.loader = DataLoader(dataset=self.dataset, shuffle=False,
                                 batch_size=self.batch_size)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.lr_boundarys[0],
                                    weight_decay=self.weight_decay)

        self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer,
                                                     base_lr=self.lr_boundarys[0], max_lr=self.lr_boundarys[1],
                                                     cycle_momentum=False,
                                                     mode="triangular2",
                                                     step_size_up=7 * int(len(self.loader)))  # 6 * len(self.loader)
        # self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.lr_boundarys[1],
        #                                              steps_per_epoch=len(self.loader), epochs=self.config["num_epochs"])
        self._RESTART = False
        self.cur_idx = self.dataset.start_index
        self.set_seeds(self.seed)
        if load_from_checkpoint:
            self.load_after_restart()

    def set_seeds(self, seed):
        """
        Ensures reproducibility
        :param seed: int: value that should be used as seed
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.dataset.seed = seed
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def load_from_checkpoint(self, checkpoint):
        """
        Assigns saved status from given checkpoint to relevant attribute variables
        :param checkpoint: dict: previously saved checkpoint that should be loaded
        """
        print("=> Loading checkpoint at epoch {} with index {}".format(checkpoint["epochs"][-1],
                                                                       checkpoint["batch_index"]))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optim_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.logger = checkpoint
        self.dataset.start_index = self.logger["batch_index"]
        self.cur_idx = self.logger["batch_index"]
        self.set_seeds(self.logger["seed"])
        self._RESTART = True
        sys.stderr.write(
            "\n--Loading previous checkpoint--\n"
            "ID: {}\tEpoch: {}\tBatch_idx: {}"
            "\n".format(self.config["track_ID"], self.logger["epochs"][-1], self.logger["batch_index"]))

    def load_after_restart(self, name="checkpoint.pth.tar"):
        try:
            checkpoint = torch.load(self.config["save_files_path"] + "/" + name, map_location=self.device)
            self.load_from_checkpoint(checkpoint)
        except IOError:
            sys.stderr.write("\nNo previous Checkpoint was found, new checkpoints will be saved at: {}".format(
                self.config["save_files_path"] + "/checkpoint.pth.tar"))
        try:
            self.metric_logger = torch.load(self.config["save_files_path"] + "/metrics.pth.tar",
                                            map_location=self.device)
        except IOError:
            sys.stderr.write("\nNo Metric log was found. Evaluation has not been done")

    def save_checkpoint(self):
        """
        captures the current training progress (model-, optimizer-, scheduler state dict, seed and current batch index)
        and saves it as "checkpoint.pth.tar" such that it the load_after_restart() method is able to reload the current
        training progress.
        """
        self.logger["state_dict"] = self.model.state_dict()
        self.logger["optim_state_dict"] = self.optimizer.state_dict()
        self.logger["batch_index"] = self.dataset.cur_idx  # self.cur_idx[-1] + 1 if self.cur_idx[-1] != 0 else 0
        self.logger["scheduler"] = self.scheduler.state_dict()
        self.logger["seed"] = self.dataset.seed
        torch.save(self.logger, self.config["save_files_path"] + "/checkpoint.pth.tar")
        # save checkpoint every 10 epochs
        if self.logger["epochs"][-1] % 10 == 0:
            torch.save(self.logger,
                       self.config["save_files_path"] + "/checkpoint_{}.pth.tar".format(self.logger["epochs"][-1]))

    def save_metric_logger(self, path=None):
        """
        captures the current evaluation status and saves it in metrics.pth.tar. 
        It will be reloaded by load_after_restart().
        """
        torch.save(self.metric_logger, path)

    def restart_script(self):
        """
        saves the current training progress and calls a script that will restart the training process from where
        it was stopped.
        """

        from subprocess import call
        self.save_checkpoint()
        sys.stderr.write("\n--Restarting script--\n"
                         "ID: {}\tEpoch: {}\tBatch_idx: {}"
                         "\n".format(self.config["track_ID"], self.logger["epochs"][-1], self.logger["batch_index"]))

        job_name = "id" + str(self.config["track_ID"]).zfill(2) + "e" + str(self.logger["epochs"][-1])
        VRAM = "3.8G"
        if "V6" in self.config["model"] or "V7" in self.config["model"] or "V3" in self.config["model"]:
            VRAM = "7.8G"
        if "V4" in self.config["model"]:
            VRAM = "10G"
        recall_parameter = 'qsub -N ' + "id" + str(self.config["track_ID"]) + "e" + str(self.logger["epochs"][-1]) + \
                           str(self.config["model"]) + ' -l nv_mem_free=' + VRAM + " -o " \
                           + str(self.config["save_files_path"]) + "/log_files/" + job_name + ".o$JOB_ID" + " -e " \
                           + str(self.config["save_files_path"]) + "/log_files/" + job_name + ".e$JOB_ID" + ' -v CFG=' \
                           + str(self.config["save_files_path"]) + "/train_config.json" + ' src/train.sge'

        if torch.cuda.is_available():
            sys.stderr.write(f"\nRecall Parameter:\n{recall_parameter}")
            call(recall_parameter, shell=True)
        else:
            print("Script would have been called:\n" + recall_parameter)

    def intermediate_eval(self, num_eval_steps=-1, random_start=True, final=False):
        """
        Calls script that will initiate an intermediate (or final) evaluation of the model at the current epoch.

        :param num_eval_steps: int: how many batches of the dataset should be taken for evaluation
        :param random_start: boolean: If True, a random position in the dataset will be used as start for the evaluation
        :param final: boolean: Indicates if the evaluation is the last one or just an intermediate one
        """
        from subprocess import call
        job_name = "IE" + str(self.config["track_ID"]).zfill(2) + "e" + str(self.logger["epochs"][-1])
        VRAM = "3.8G"
        # if "V4" in self.config["model"]:
        #     VRAM = "5G"
        # if "gruV3" in self.config["model"]:
        #     VRAM = "5G"
        option = ' -l nv_mem_free=' + str(VRAM)
        recall_parameter = "qsub -N " + job_name + self.config["model"] + option \
                           + " -l h_rt=01:29:00" \
                           + " -o " + str(self.config["save_files_path"]) + "/log_files/" + job_name + ".o$JOB_ID" \
                           + " -e " + str(self.config["save_files_path"]) + "/log_files/" + job_name + ".e$JOB_ID" \
                           + " -v STPS=" + str(num_eval_steps) \
                           + " -v RDM=" + str(int(random_start)) \
                           + " -v FNL=" + str(int(final)) \
                           + " -v PTH=" + str(self.config["save_files_path"]) + " src/eval_model.sge"

        if torch.cuda.is_available():
            sys.stderr.write(f"\nRecall Parameter:\n{recall_parameter}")
            call(recall_parameter, shell=True)
        else:
            print("Script would have been called:\n" + recall_parameter)

    def get_starting_parameters(self, what=""):
        """
        Only called at the beginning of the training if this script is a restart.
        It ensures that the running loss and the starting epoch are set correctly.

        :param what: either "running_loss" or "epochs"
        :return: 0 if this training run is not a restarted one
        :return: the running loss value that was present before the script was restarted if what == "running_loss"
        :return: the last epoch number that was trained before the script was restarted if what == "epochs"
        """
        if not self._RESTART:
            return 0
        else:
            if what == "running_loss":
                return self.logger["running_loss"]
            elif what == "epoch":
                return self.logger["epochs"][-1]
            elif what == "miou":
                return self.logger["miou"]

    def train(self):
        """
        The main training loop.
        Keeps track if the maximum runtime is exceded and restarts the script if necessary.
        """
        for epoch in tqdm(range(self.get_starting_parameters(what="epoch"), self.config["num_epochs"])):
            sys.stderr.write(f"\nStarting new epoch: {epoch}\n")
            if not self._RESTART:
                self.logger["epochs"].append(epoch)
                self.logger["lrs"].append(self.optimizer.state_dict()["param_groups"][0]["lr"])
            self.logger["running_loss"] = self.get_starting_parameters(what="running_loss")
            self.logger["miou"] = self.get_starting_parameters(what="miou")
            self._RESTART = False
            for i, batch in enumerate(self.loader):

                # ensure that current batch can be finished within the max runtime
                if self.time_logger.check_for_restart():
                    self.restart_script()
                    return  # End the script

                idx, video_start, (images, labels) = batch
                self.cur_idx = idx
                sys.stderr.write(f"\nCurrent Index: {idx}; dataset idx {self.cur_idx}")

                # sent tensores to gpu if available
                images, labels = (images.to(self.device), labels.to(self.device))

                # check if a new 4 sec clip has started, if so make sure the hidden and cell state are reset and no
                # wrong information is used
                if torch.any(video_start):
                    self.model.reset()

                # manually check if end of batch is reached. Dataset will return idx=0 if length is overshot
                if len(idx) == self.batch_size:
                    if torch.sum(idx == 0) > 1:
                        sys.stderr.write(f"\nEnd reached of batch at index {idx}\n")
                        self.dataset.start_index = 0  # reset start index for the next batch
                        break

                pred = self.model(images)
                loss = self.criterion(pred, labels)

                # keep track of memory usage
                # memory = get_gpu_memory_map()[0] if torch.cuda.is_available() else 0
                # max_mem = max_mem if max_mem > memory else memory

                self.optimizer.zero_grad()
                # keep the retain graph for certain intervals (only used in LSTMV6)
                if self.detach_interval == 1:
                    loss.backward(retain_graph=False)
                    self.model.detach()
                elif i > 0 and i % self.detach_interval == 0:
                    loss.backward(retain_graph=False)  # <---------------------------------------------------------
                    self.model.detach()
                else:
                    loss.backward(retain_graph=True)
                self.optimizer.step()
                self.scheduler.step()
                self.logger["running_loss"] += loss.item() * images.size(0)
                print("Loss: {}, running_loss: {}".format(loss, self.logger["running_loss"]))
                with torch.no_grad():
                    labels = labels.type(torch.uint8)
                    outputs = outputs.type(torch.uint8)
                    set_out = torch.max(outputs.int())  # can only be in range (0-1)
                    set_lbl = torch.max(labels.int())
                    num_classes = max(set_out, set_lbl) + 1
                    hist = fast_hist(outputs.to("cpu"), labels.to("cpu"), num_classes=num_classes)
                    self.logger["miou"] += jaccard_index(hist)

            # with open(str(self.config["save_files_path"] + "/memory.txt"), "w") as txt_file:
            #
            #     txt_file.write(f"Max cuda memory used in epoch {epoch}: {max_mem}\n")
            self.logger["mious"].append(self.logger["miou"] / len(self.dataset))
            self.logger["loss"].append(self.logger["running_loss"] / len(self.dataset))
            visualize_logger(self.logger, self.config["save_files_path"])
            self.save_checkpoint()
            if epoch == self.config["num_epochs"] - 1:
                print("final")
                self.intermediate_eval(random_start=False, final=True)
            elif epoch % self.config["evaluation_steps"] == 0 and epoch > 0:
                print("intermediate")
                self.intermediate_eval(num_eval_steps=29 * 4 * 5, random_start=False, final=False)

    def eval(self, random_start=True, eval_length=29 * 4, save_file_path=None, load_most_recent=True,
             checkpoint="checkpoint.pth.tar", final=False):
        """
        Evaluation loop. Will usually be called by intermediate_eval() through a different script.
        Stores and saves the evaluation results.

        :param random_start:     determines if evaluation should start from a random position in the dataset.
        :param eval_length:      how much of the dataset should be processed for evaluation
        :param save_file_path:   config["save_file_path"] + /intermediate_results/ or /final_results/,
                                 where results are saved
        :param load_most_recent: Should the most recent checlpoint be loaded? (usefull for debugging)
        :param checkpoint: the checkpoint that should be loaded
        :param final: is it a final evaluation or an intermediate
        """
        import time
        self.set_seeds(seed=0)
        video_freq = 20
        if load_most_recent:
            self.load_after_restart(name=checkpoint)  # load the most recent log data
        else:
            self.logger["epochs"] = [-1]
        self.dataset.set_start_index(0)
        running_loss = 0
        with torch.no_grad():
            sys.stderr.write("\nEvaluating\n")
            self.model.eval()
            self.model.start_eval()
            metrics = defaultdict(AverageMeter)
            to_PIL = T.ToPILImage()
            if random_start:
                start_index = np.random.choice(range(len(self.dataset) - eval_length))
                self.dataset.set_start_index(int(start_index))
            loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)
            out_folder = Path(save_file_path)
            out_folder.mkdir(parents=True, exist_ok=True)
            mode = "train" if self.dataset.train else "val"
            if self.logger["epochs"][-1] % video_freq == 0 or self.logger["epochs"][-1] == self.config[
                "num_epochs"] - 1 or final:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_vid = cv2.VideoWriter(
                    str(out_folder) + "/eval_{}_ep{}.mp4".format(mode, self.logger["epochs"][-1]), fourcc, 29,
                    (1536, 270))

            flickering_sum = 0
            flickering_img_size = 0
            fp = 0
            flickering_sum2 = 0
            fip = 0
            flickeringv2 = 0
            flickering2v2 = 0
            flickering_img_sizev2 = 0
            fpv2 = 0
            fipv2 = 0

            '''
            Evaluation loop:
            - meassures time the model takes to process 1 batch
            - tracks several metric values (FP, FIP, MIoU, Pixel Accuracy, Per Class Accuracy, Dice)
            '''
            for i, batch in enumerate(loader):
                print(self.time_logger.get_status())
                sys.stderr.write("\n" + self.time_logger.get_status() + "\n")
                start = time.time()
                idx, video_start, (images, labels) = batch
                if torch.sum(idx == 0) > 1:
                    sys.stderr.write(f"\nlen: {len(self.dataset)}; eval_length: {eval_length}; idx: {idx}\n")
                images, labels = (images.to(self.device), labels.to(self.device))
                if torch.any(video_start.bool()):
                    self.model.reset()
                pred = self.model(images)
                outputs = torch.argmax(pred, dim=1).float()
                end = time.time() - start
                loss = self.criterion(pred, labels)
                # Conversion for metric evaluations
                labels = labels.type(torch.uint8)
                outputs = outputs.type(torch.uint8)
                set_out = torch.max(outputs.int())  # can only be in range (0-1)
                set_lbl = torch.max(labels.int())
                num_classes = max(set_out, set_lbl) + 1
                sys.stderr.write(f"\nNum Classes: {num_classes} and sets: ({set_out, set_lbl})")
                overall_acc, avg_per_class_acc, avg_jacc, avg_dice = eval_metrics(outputs.to("cpu"),
                                                                                  labels.to("cpu"),
                                                                                  num_classes=num_classes)

                running_loss += loss.item() * images.size(0)
                diff_img = (outputs != labels) * (labels + 1)
                if i > 0:
                    flickering = torch.sum(diff_img != old_out)
                    flickering_sum += flickering
                    flickering_img_size += outputs.shape[1] * outputs.shape[2]
                    fp = float(flickering_sum) / float(flickering_img_size)
                    flickering2 = torch.sum(outputs != old_out2)
                    flickering_sum2 += flickering2
                    fip = float(flickering_sum2) / float(flickering_img_size)

                diff_imgv2 = abs(outputs - labels)
                if i > 0:
                    flickeringv2 += torch.sum(abs(diff_imgv2 - old_outv2))
                    flickering_img_sizev2 += outputs.shape[1] * outputs.shape[2]
                    fpv2 = float(flickeringv2) / float(flickering_img_sizev2)
                    flickering2v2 += torch.sum(abs(outputs - old_out2))
                    fipv2 = float(flickering2v2) / float(flickering_img_sizev2)
                print(fp, fip, "and: ", fpv2, fipv2)
                print(f"current idx: {idx}\t len(dataset): {len(self.dataset)}")
                sys.stderr.write(f"\ncurrent idx: {idx}\t len(dataset): {len(self.dataset)}\n")
                old_outv2 = diff_imgv2
                old_out = diff_img
                old_out2 = outputs
                metrics["FPv2"].update(fpv2)
                metrics["FIPv2"].update(fipv2)
                metrics["FP"].update(fp)
                metrics["FIP"].update(fip)
                metrics["Time_taken"].update(end)
                metrics["Mean IoU"].update(avg_jacc)
                metrics["Pixel Accuracy"].update(overall_acc)
                metrics["Per Class Accuracy"].update(avg_per_class_acc)
                metrics["Dice"].update(avg_dice)

                # conversions since hstack expects PIL image or np array and cv2 np array with channel at last position
                if self.logger["epochs"][-1] % video_freq == 0 or self.logger["epochs"][-1] == self.config[
                    "num_epochs"] - 1 or final:
                    for j in range(self.batch_size):  # if batchsize > 1 assert that the video writing works
                        out = outputs[j, :, :].unsqueeze(0)
                        lbl = labels[j, :, :].unsqueeze(0)
                        img = images[j, :, :, :].unsqueeze(0)
                        tmp_prd = to_PIL(out[0].cpu().float())
                        tmp_inp = to_PIL(img.squeeze(0).cpu())
                        tmp_inp = Image.fromarray(cv2.cvtColor(np.asarray(tmp_inp), cv2.COLOR_RGB2BGR))
                        tmp_lbl = to_PIL(lbl.cpu().float())
                        out_vid.write(np.array(stack.hstack([tmp_inp, tmp_lbl, tmp_prd])))
                        # break after certain amount of frames (remove for final (last) evaluation)
                if i == eval_length:
                    break
            # save status of evaluation
            metrics["eval_loss"].update(running_loss / len(self.dataset))
            metrics["curr_epoch"] = self.logger["epochs"][-1]
            metrics["num_params"].update(sum([param.nelement() for param in self.model.parameters()]))

            if self.logger["epochs"][-1] % video_freq == 0 or self.logger["epochs"][-1] == self.config[
                "num_epochs"] - 1 or final:
                out_vid.release()
            self.model.train()
            self.model.end_eval()
            self.metric_logger[mode].append(metrics)
            path = self.config[
                       "save_files_path"] + "/metrics.pth.tar" if not final else save_file_path + "/metrics.pth.tar"
            self.save_metric_logger(path=path)

    def time_and_image_eval(self, checkpoint="checkpoint.pth.tar", batch_size=1):
        """
        This method is used to meassure the average time a model takes to evaluate a single image. In addition to that
        several image are selected and saves as png files.
        :param self:
        :param checkpoint: the checkpoint that should be loaded
        :param batch_size: what batch size should be used
        :return: average time_taken

        """
        from statistics import mean
        import time
        self.set_seeds(seed=0)
        print("dataset seed: ", self.dataset.seed)
        self.load_after_restart(name=checkpoint)
        self.dataset.set_start_index(0)
        durations = []
        with torch.no_grad():
            sys.stderr.write("\nEvaluating\n")
            self.model.eval()
            self.model.start_eval()
            to_PIL = T.ToPILImage()
            loader = DataLoader(dataset=self.dataset, batch_size=batch_size, shuffle=False)
            out_folder = Path(self.config["save_files_path"]) / "example_results"
            out_folder.mkdir(parents=True, exist_ok=True)
            mode = "train" if not self.test else "val"
            for i, batch in enumerate(loader):
                if self.test:
                    if i not in [58, 174, 290, 406]:
                        continue
                print("i", i)
                start = time.time()
                idx, video_start, (images, labels) = batch
                print("index: ", idx)
                images, labels = (images.to(self.device), labels.to(self.device))
                if torch.any(video_start.bool()):
                    self.model.reset()
                pred = self.model(images)
                outputs = torch.argmax(pred, dim=1).float()
                end = time.time() - start
                durations.append(end * 1000)
                if i in [58, 174, 290, 406]:  #
                    labels = labels.type(torch.uint8)
                    outputs = outputs.type(torch.uint8)
                    for j in range(batch_size):  # if batchsize > 1 assert that the video writing works
                        out = outputs[j, :, :].unsqueeze(0)
                        lbl = labels[j, :, :].unsqueeze(0)
                        img = images[j, :, :, :].unsqueeze(0)
                        tmp_prd = to_PIL(out[0].cpu().float())
                        tmp_inp = to_PIL(img.squeeze(0).cpu())
                        tmp_inp = Image.fromarray(cv2.cvtColor(np.asarray(tmp_inp), cv2.COLOR_RGB2BGR))
                        tmp_lbl = to_PIL(lbl.cpu().float())
                        image = (np.array(stack.hstack([tmp_inp, tmp_lbl, tmp_prd])))
                        print("saving: ", str(out_folder))
                        cv2.imwrite(str(out_folder) + "/{}_{}_{}.png".format(self.config["model"], mode, i), image)
                if i == 500:
                    break
        return mean(durations)


if __name__ == "__main__":
    model = "Deep+_mobile"
    batch_size = 6
    track_id = 00
    num_epochs = 10
    eval_steps = 2

    unique_name = "ID" + str(track_id) + model + "_bs" + str(batch_size) + "num_ep" \
                  + str(num_epochs) + "ev" + str(eval_steps)
    config = {
        "save_folder_path": "src/models/trained_models/testing2",
        "save_files_path": "src/models/trained_models/testing2/" + unique_name,
        "model": model,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "evaluation_steps": eval_steps,
        "loss": "SoftDice",
        "track_ID": track_id
    }
    Path(config["save_files_path"]).mkdir(parents=True, exist_ok=True)
    with open(str(Path(config["save_files_path"]) / "train_config.json"), "w") as js:  # save learn config
        json.dump(config, js)
    trainer = GridTrainer(config, load_from_checkpoint=False)
    trainer.eval(save_file_path=str(config["save_files_path"] + "/intermediate_results"), load_most_recent=False)
    # trainer.train()

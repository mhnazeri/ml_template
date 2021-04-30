import gc
from pathlib import Path
from datetime import datetime
import sys
try:
    sys.path.append(str(Path(".").resolve()))
except:
    raise RuntimeError("Can't append root directory of the project the path")

import comet_ml
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR

from model.net import CustomModel
from model.data_loader import CustomDataset, CustomDatasetVal
from utils.nn import check_grad_norm, init_weights_normal, EarlyStopping, op_counter
from utils.io import save_checkpoint, load_checkpoint,
from utils.utility import get_conf, timeit


class Learner:
    def __init__(self, cfg_dir: str):
        self.cfg = get_conf(cfg_dir)
        self.logger = self.init_logger(self.cfg.logger)
        self.dataset = CustomDataset(**self.cfg.dataset)
        self.data = DataLoader(self.dataset, **self.cfg.dataloader)
        self.val_dataset = CustomDatasetVal(**self.cfg.val_dataset)
        self.val_data = DataLoader(self.val_dataset, **self.cfg.dataloader)
        self.logger.log_parameters({"tr_len": len(self.dataset),
                                    "val_len": len(self.val_dataset)})
        self.model = CustomModel(**self.cfg.model)
        self.model.apply(init_weights_normal)
        self.device = self.cfg.train_params.device
        self.model = self.model.to(device=self.device)
        if self.cfg.train_params.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), **self.cfg.adam)
        elif self.cfg.train_params.optimizer.lower() == "rmsprop":
            self.optimizer = optim.RMSprop(self.model.parameters(), **self.cfg.rmsprop)
        else:
            raise ValueError(f"Unknown optimizer {self.cfg.train_params.optimizer}")

        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.criterion = None

        if self.cfg.logger.resume:
            # load checkpoint
            print("Loading checkpoint")
            save_dir = self.cfg.directory.load
            checkpoint = load_checkpoint(save_dir, self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.epoch = checkpoint["epoch"]
            self.e_loss = checkpoint["e_loss"]
            self.best = checkpoint["best"]
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} "
                f"Loading checkpoint was successful, start from epoch {self.epoch}"
                f" and loss {self.best}"
            )
        else:
            self.epoch = 1
            self.best = np.inf
            self.e_loss = []

        # initialize the early_stopping object
        self.early_stopping = EarlyStopping(
            patience=self.cfg.train_params.patience,
            verbose=True,
            path=self.cfg.directory.load,
            delta=self.cfg.train_params.early_stopping_delta,
        )

        # stochastic weight averaging
        self.swa_model = AveragedModel(self.model)
        self.swa_scheduler = SWALR(self.optimizer, **self.cfg.SWA)

    def train(self):
        while self.epoch <= self.cfg.train_params.epochs:
            running_loss = []
            self.model.train()
            
            bar = tqdm(enumerate(self.data), desc=f"Epoch {self.epoch}/{self.cfg.train_params.epochs}")
            for idx, (x, y) in bar:
                self.optimizer.zero_grad()
                # move data to device
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                # forward, backward
                out = self.model(x)
                loss = self.criterion(out, y)
                loss.backward()
                # check grad norm for debugging
                grad_norm = check_grad_norm(self.model)
                # update
                self.optimizer.step()

                running_loss.append(loss.item())

                bar.set_postfix(
                    loss=loss.item(),
                    Grad_Norm=grad_norm
                )

                self.logger.log_metrics(
                    {
                        "epoch": self.epoch,
                        "batch": idx,
                        "loss": loss.item(),
                        "GradNorm": grad_norm,
                    }
                )

            bar.close()
            if self.epoch > self.cfg.train_params.swa_start:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
            else:
                self.lr_scheduler.step()

            # validate on val set
            val_loss, t = self.validate()
            t /= len(self.val_dataset)

            # average loss for an epoch
            self.e_loss.append(np.mean(running_loss))  # epoch loss
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} Epoch {self.epoch} summary: train Loss: {self.e_loss[-1]:.2f} \t| Val loss: {val_loss:.2f}"
                f"\t| time: {t:.3f} seconds"
            )

            self.logger.log_metrics(
                {
                    "epoch": self.epoch,
                    "epoch_loss": self.e_loss[-1],
                    "val_loss": val_loss,
                    "time": t,
                }
            )

            # early_stopping needs the validation loss to check if it has decreased,
            # and if it has, it will make a checkpoint of the current model
            self.early_stopping(val_loss, self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                self.save()
                break

            if self.epoch % self.cfg.train_params.save_every == 0:
                self.save()

            gc.collect()
            self.epoch += 1

        # Update bn statistics for the swa_model at the end
        if self.epoch >= self.cfg.train_params.swa_start:
            torch.optim.swa_utils.update_bn(self.data, self.swa_model)
            self.save(name=self.cfg.directory.model_name + "-final" + str(self.epoch) + "-swa")

        macs, params = op_counter(self.model, sample=x)
        print(macs, params)
        self.logger.log_metrics({"GFLOPS": macs[:-1], "#Params": params[:-1]})
        print("Training Finished!")

    @timeit
    @torch.no_grad()
    def validate(self):

        self.model.eval()

        running_loss = []

        for idx, (x, y) in tqdm(enumerate(self.val_data), desc="Validation"):
            # move data to device
            x = x.to(device=self.device)
            y = y.to(device=self.device)

            # forward, backward
            if self.epoch > self.cfg.train_params.swa_start:
                # Update bn statistics for the swa_model
                torch.optim.swa_utils.update_bn(self.data, self.swa_model)
                out = self.swa_model(x)
            else:
                out = self.model(x)

            loss = self.criterion(out, y)
            running_loss.append(loss.item())

        # average loss
        loss = np.mean(running_loss)

        return loss

    def init_logger(self, cfg):
        logger = None
        # Check to see if there is a key in environment:
        EXPERIMENT_KEY = cfg.experiment_key

        # First, let's see if we continue or start fresh:
        CONTINUE_RUN = cfg.resume
        if (EXPERIMENT_KEY is not None):
            # There is one, but the experiment might not exist yet:
            api = comet_ml.API()  # Assumes API key is set in config/env
            try:
                api_experiment = api.get_experiment_by_id(EXPERIMENT_KEY)
            except Exception:
                api_experiment = None
            if api_experiment is not None:
                CONTINUE_RUN = True
                # We can get the last details logged here, if logged:
                # step = int(api_experiment.get_parameters_summary("batch")["valueCurrent"])
                # epoch = int(api_experiment.get_parameters_summary("epochs")["valueCurrent"])

        if CONTINUE_RUN:
            # 1. Recreate the state of ML system before creating experiment
            # otherwise it could try to log params, graph, etc. again
            # ...
            # 2. Setup the existing experiment to carry on:
            logger = comet_ml.ExistingExperiment(
                previous_experiment=EXPERIMENT_KEY,
                log_env_details=True,  # to continue env logging
                log_env_gpu=True,  # to continue GPU logging
                log_env_cpu=True,  # to continue CPU logging
                auto_histogram_weight_logging=True,
                auto_histogram_gradient_logging=True,
                auto_histogram_activation_logging=True
            )
            # Retrieved from above APIExperiment
            # self.logger.set_epoch(epoch)

        else:
            # 1. Create the experiment first
            #    This will use the COMET_EXPERIMENT_KEY if defined in env.
            #    Otherwise, you could manually set it here. If you don't
            #    set COMET_EXPERIMENT_KEY, the experiment will get a
            #    random key!
            logger = comet_ml.Experiment(
                disabled=cfg.disabled,
                project_name=cfg.project,
                auto_histogram_weight_logging=True,
                auto_histogram_gradient_logging=True,
                auto_histogram_activation_logging=True
            )
            logger.add_tags(cfg.tags.split())
            logger.log_parameters(self.cfg)

        return logger

    def save(self, name=None):
        checkpoint = {"epoch": self.epoch,
                      "model": self.model.state_dict(),
                      "optimizer": self.optimizer.state_dict(),
                      "lr_scheduler": self.lr_scheduler.state_dict(),
                      "best": self.best,
                      "e_loss": self.e_loss
                      }

        if name is None and self.epoch >= self.cfg.train_params.swa_start:
            save_name = self.cfg.directory.model_name + str(self.epoch) + "-swa"
            checkpoint['model-swa'] = self.swa_model.state_dict()

        elif name is None:
            save_name = self.cfg.directory.model_name + str(self.epoch)
            
        else:
            save_name = name

        if self.e_loss[-1] < self.best:
            self.best = self.e_loss[-1]
            checkpoint["best"] = self.best
            save_checkpoint(
                checkpoint, True, self.cfg.directory.save, save_name
            )
        else:
            save_checkpoint(
                checkpoint, False, self.cfg.directory.save, save_name
            )


if __name__ == "__main__":
    cfg_path = "./conf/config"
    learner = Learner(cfg_path)
    learner.train()

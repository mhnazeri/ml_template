import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from thop import profile, clever_format

from model.net import CustomModel
from model.data_loader import CustomDataset
from logger import Logger
from utils import (
    get_conf,
    check_grad_norm,
    save_checkpoint,
    load_checkpoint,
    timeit,
    init_weights_normal,
    EarlyStopping,
)


class Learner:
    def __init__(self, cfg_dir: str, **kwargs):
        self.cfg = get_conf(cfg_dir)
        self.logger = Logger(config=self.cfg, **cfg.logger)
        dataset = CustomDataset(**cfg.dataset)
        self.data = DataLoader(dataset, **cfg.dataloader)
        val_dataset = CustomDatasetVal(**cfg.val_dataset)
        self.val_data = DataLoader(val_dataset, **cfg.dataloader)
        self.model = CustomModel(**cfg.model)
        self.model.apply(init_weights_normal)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)
        if cfg.train_params.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), **cfg.adam)
        elif cfg.train_params.optimizer.lower() == "rmsprop":
            self.optimizer = optim.RMSprop(self.model.parameters(), **cfg.rmsprop)
        else:
            raise ValueError(f"Unknown optimizer {cfg.train_params.optimizer}")

        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        self.criterion = None

        if self.cfg.logger.resume:
            # load checkpoint
            print("Loading checkpoint")
            save_dir = cfg.directory.load
            checkpoint = load_checkpoint(save_dir, device)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.epoch = checkpoint["epoch"]
            self.best = checkpoint["best"]
            print(
                f"Loading checkpoint was successful, start from epoch {epoch}"
                f" and loss {best}"
            )
        else:
            self.epoch = 1
            self.best = np.inf

        # initialize the early_stopping object
        self.early_stopping = EarlyStopping(
            patience=cfg.train_params.patience,
            verbose=True,
            path=cfg.directory.load,
            delta=cfg.train_params.early_stopping_delta,
        )

        # stochastic weight averaging
        self.swa_model = AveragedModel(model)
        self.swa_scheduler = SWALR(optimizer, swa_lr=0.05)

        self.logger.log_model(self.model)

    def train(self):
        while self.epoch <= self.cfg.train_params.epochs:
            running_loss = []
            print(f"Epoch {self.epoch}:\n")
            for idx, (x, y) in tqdm(enumerate(self.data)):
                self.model.train()
                self.optimizer.zero_grad()
                # move data to device
                x = x.to(self.device)
                y = y.to(self.device)

                # forward, backward
                out = self.model(x)
                loss = self.criterion(out, y)
                loss.backward()
                # check grad norm for debugging
                grad_norm = check_grad_norm(self.model)
                # update
                self.optimizer.step()

                running_loss.append(loss.item())
                print(
                    f"Batch {idx}, train loss: {loss.item():.2f}"
                    f"\t Grad_Norm: {grad_norm:.2f}"
                )
                self.logger.log_metric(
                    {
                        "epoch": self.epoch,
                        "batch": idx,
                        "loss": loss.item(),
                        "GradNorm": grad_norm,
                    }
                )

            if self.epoch > self.cfg.train_params.swa_start:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
            else:
                self.lr_scheduler.step()
            # validate on val set
            val_loss, t = self.validate()
            # average loss for an epoch
            e_loss = append(np.mean(running_loss))  # epoch loss
            print(
                f"Epoch {self.epoch}, train Loss: {e_loss:.2f} \t Val loss: {val_loss:.2f}"
                f"\t time: {t:.3f} seconds"
            )
            self.logger.log_metric(
                {
                    "epoch": self.epoch,
                    "epoch_loss": e_loss,
                    "val_loss": val_loss,
                    "time": t,
                }
            )

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            self.early_stopping(val_loss, self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break

            if self.epoch % self.cfg.train_params.save_every == 0:
                checkpoint = {}
                checkpoint["epoch"] = self.epoch
                checkpoint["unet"] = (
                    self.swa_model.state_dict()
                    if self.epoch == self.cfg.train_params.epochs
                    else self.model.state_dict()
                )
                checkpoint["optimizer"] = self.optimizer.state_dict()
                checkpoint["lr_scheduler"] = self.lr_scheduler.state_dict()

                if val_loss < best:
                    best = val_loss
                    checkpoint["best"] = best
                    save_checkpoint(
                        checkpoint, True, self.cfg.directory.save, str(self.epoch)
                    )
                else:
                    save_checkpoint(
                        checkpoint, False, self.cfg.directory.save, str(self.epoch)
                    )

        # Update bn statistics for the swa_model at the end
        # torch.optim.swa_utils.update_bn(self.data, self.swa_model)
        gc.collect()
        self.epoch += 1

        macs, params = op_counter(self.model)
        print(macs, params)
        self.logger.log_metric({"GFLOPS": macs[:-1], "#Params": params[:-1]})
        print("ALL Finished!")

    @torch.no_grad()
    @timeit
    def validate(self):

        self.model.eval()

        running_loss = []
        for idx, (x, y) in tqdm(enumerate(self.val_data)):
            # move data to device
            x = x.to(self.device)
            y = y.to(self.device)

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

    def op_counter(self):
        self.model.eval()
        _input = torch.randn(1, 3, 256, 256).to(self.device)  # generate a random input
        macs, params = profile(self.model, inputs=(_input,))
        macs, params = clever_format([macs, params], "%.3f")
        return macs, params


if __name__ == "__main__":
    cfg_path = Path("./conf/config")
    learner = Learner(cfg_path)
    learner.train()

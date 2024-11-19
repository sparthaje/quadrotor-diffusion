import os
import copy
import time
from datetime import datetime
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tqdm

import quadrotor_diffusion.utils.nn.ema as ema
from quadrotor_diffusion.utils.nn.args import TrainerArgs
from quadrotor_diffusion.utils.dataset.normalizer import Normalizer
from quadrotor_diffusion.utils.logging import (
    dataclass_to_table,
    iprint as print
)


class Trainer:
    def __init__(
        self,
        args: TrainerArgs,
        model: nn.Module,
        dataset: Dataset,
    ):
        """"
        Generalized trainer for any nn.Module that takes a list of dataclasses as arguments for initialization

        Parameters:
        - args: training hyperparameters
        - model: nn.Module that takes a Tuple[dataclass] for initialization and has property .args
        - dataset: dataset that has a .normalizer property which returns a normalizer
        """

        self.args = args
        if args.num_gpus == -1:
            args.num_gpus = torch.cuda.device_count()

        self.multi_gpu = args.num_gpus > 1
        if self.multi_gpu:
            self.model = nn.DataParallel(model)
            # TODO(shreepa): Fix this
            raise NotImplementedError("There is some problem where multiple gpus isn't working")
        else:
            self.model = model

        self.model.to(args.device)
        self.ema_model = None

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.batches_seen = 0

        batch_size = args.batch_size_per_gpu * args.num_gpus
        self.train_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.normalizer: Normalizer = dataset.normalizer
        self.create_log_dir()

    @torch.no_grad()
    def test_forward_pass(self):
        start = time.time()
        first_item: torch.Tensor = next(iter(self.train_data_loader)).to(self.args.device)
        if self.multi_gpu:
            loss = self.model.module.compute_loss(first_item)
        else:
            loss = self.model.compute_loss(first_item)
        duration = time.time() - start
        print(f"Forward pass succeeded in {duration:.2f}s")
        return loss

    def train(self):
        epoch = 0
        while self.args.max_epochs is None or epoch < self.args.max_epochs:
            epoch_losses = {}

            start = time.time()
            for batch in tqdm.tqdm(self.train_data_loader, desc=f"[ training ] Epoch {epoch+1}"):
                batch = batch.to(self.args.device)
                self.batches_seen += 1

                loss_dict = self.model.module.compute_loss(batch) if isinstance(
                    self.model, nn.DataParallel) else self.model.compute_loss(batch)

                loss = loss_dict["loss"]
                loss /= self.args.batches_per_backward
                loss.backward()

                if self.batches_seen % self.args.batches_per_backward == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if self.ema_model is None and self.batches_seen > self.args.num_batches_no_ema:
                    self.ema_model = copy.deepcopy(self.model.module if isinstance(
                        self.model, nn.DataParallel) else self.model)

                if self.ema_model is not None and self.batches_seen % self.args.num_batches_per_ema == 0:
                    ema.update_model_average(
                        self.ema_model,
                        self.model.module if self.multi_gpu else self.model,
                        self.args.ema_decay
                    )

                detached_losses = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
                detached_losses["loss"] *= self.args.batches_per_backward
                for k, v in detached_losses.items():
                    if k not in epoch_losses:
                        epoch_losses[k] = 0.0
                    epoch_losses[k] += v

            epoch_losses = {k: v / len(self.train_data_loader.dataset)
                            for k, v in epoch_losses.items()}
            if (epoch + 1) % self.args.save_freq == 0:
                self.save(epoch, epoch_losses["loss"])

            log_stmnt = f"Epoch {epoch + 1}: Avg Loss per Sample {epoch_losses['loss']:.1e} {time.time() - start:.2f}s"
            print(log_stmnt, "\n")

            log_parts = [
                str(epoch),
                f"{time.time() - start:.2f}"
            ]

            loss_components = sorted(epoch_losses.keys())
            log_parts.extend([f"{epoch_losses[k]}" for k in loss_components])

            log_stmnt = ",".join(log_parts)
            with open(os.path.join(self.args.log_dir, "logs.csv"), "a") as f:
                f.write(log_stmnt + "\n")

            epoch += 1

    def create_log_dir(self):
        now = datetime.now()
        date_str = now.strftime("%b.%d_%I:%M_%p")

        dirs = [d for d in os.listdir(self.args.log_dir) if os.path.isdir(os.path.join(self.args.log_dir, d))]
        max_num = max([int(d.split('.')[0]) for d in dirs if d.split('.')[0].isdigit()], default=0)

        model_type: str = str(type(self.model.module) if type(self.model) == nn.DataParallel else type(self.model))
        model_type = model_type.split('.')[-1][:-2]
        new_dir_name = f"{max_num + 1}.{model_type}.{date_str}"
        new_dir_path = os.path.join(self.args.log_dir, new_dir_name)
        os.makedirs(new_dir_path)

        epoch_path = os.path.join(new_dir_path, "checkpoints")
        os.makedirs(epoch_path)

        print(f"Save directory created: {new_dir_path}")
        self.args.log_dir = new_dir_path

        losses = self.test_forward_pass()
        loss_components = sorted(losses.keys())
        with open(os.path.join(self.args.log_dir, "logs.csv"), "w") as f:
            f.write("epoch,time," + ','.join(loss_components) + "\n")

        print(f"Dataset: {type(self.train_data_loader.dataset)}")
        print(f"Normalization: {self.train_data_loader.dataset.normalizer}")

        def args_to_str(model):
            return "\n".join([dataclass_to_table(arg) for arg in model.args])

        with open(os.path.join(self.args.log_dir, "overview.txt"), "w") as f:
            f.write(dataclass_to_table(self.args) + "\n")
            f.write(args_to_str(self.model.module) if self.multi_gpu else args_to_str(self.model))
            f.write(f"\nDataset: {type(self.train_data_loader.dataset)}\n")
            f.write(f"Normalization: {self.train_data_loader.dataset.normalizer}\n\n")

    def save(self, epoch: int, loss: float):
        filename = os.path.join(self.args.log_dir, f"checkpoints/epoch_{epoch}_loss_{loss:.4f}")
        torch.save(
            {
                "model_type": type(self.model.module) if self.multi_gpu else type(self.model),
                "model_state": self.model.module.state_dict() if self.multi_gpu else self.model.state_dict(),
                "ema_model_state": self.ema_model.state_dict() if self.ema_model is not None else dict(),
                "normalizer": self.train_data_loader.dataset.normalizer,
                "args": self.model.module.args if self.multi_gpu else self.model.args,
                "trainer_args": self.args,
            }, filename
        )
        print(f"Saved to {filename}")

    @staticmethod
    def load(filepath) -> Tuple[nn.Module, nn.Module, Normalizer, TrainerArgs]:
        """
        Loads from a checkpoint

        Returns:
        - model: The model that was optimized
        - ema: A exponential moving average of the model during training
        - nomralizer: Normalizer used in training for the dataset
        - trainer_args: Args used for training
        """
        state_dict = torch.load(filepath)
        model_type: nn.Module = state_dict["model_type"]

        model = model_type(state_dict["args"])
        model.load_state_dict(state_dict["model_state"])

        ema_model = model_type(state_dict["args"])
        ema_model.load_state_dict(state_dict["ema_model_state"])

        return (
            model,
            ema_model,
            state_dict["normalizer"],
            state_dict["trainer_args"],
        )

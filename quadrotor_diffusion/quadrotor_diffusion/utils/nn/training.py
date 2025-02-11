import os
import copy
import time
from datetime import datetime
from typing import Tuple
import warnings

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

# TODO(shreepa): Probably should fix this at some point
# Suppress FutureWarning for this specific issue
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only=False.*")


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
        self.epoch = 0

    @torch.no_grad()
    def test_forward_pass(self):
        start = time.time()
        first_item: torch.Tensor | dict[str, torch.Tensor] = next(iter(self.train_data_loader))

        # If dataset uses a dict, move each tensor to gpu
        if type(first_item) == dict:
            first_item = {k: v.to(self.args.device) for k, v in first_item.items()}
        else:
            first_item = first_item.to(self.args.device)

        if self.multi_gpu:
            loss = self.model.module.compute_loss(first_item, debug=True)
        else:
            loss = self.model.compute_loss(first_item, debug=True)
        duration = time.time() - start
        print(f"Forward pass succeeded in {duration:.2f}s")
        return loss

    def train(self):
        while self.args.max_epochs is None or self.epoch < self.args.max_epochs:
            epoch_losses = {}

            start = time.time()
            samples_seen = 0
            for batch in tqdm.tqdm(self.train_data_loader, desc=f"[ training ] Epoch {self.epoch+1}"):
                # If dataset uses a dict, move each tensor to gpu
                if type(batch) == dict:
                    batch = {k: v.to(self.args.device) for k, v in batch.items()}
                else:
                    batch = batch.to(self.args.device)

                self.batches_seen += 1

                loss_dict = self.model.module.compute_loss(batch, epoch=self.epoch) if isinstance(
                    self.model, nn.DataParallel) else self.model.compute_loss(batch, epoch=self.epoch)

                # Average loss per sample in batch
                loss = loss_dict["loss"]
                (loss / self.args.batches_per_backward).backward()

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
                batch_size = len(batch) if type(batch) != dict else len(next(iter(batch.values())))
                for k, v in detached_losses.items():
                    if k not in epoch_losses:
                        epoch_losses[k] = 0.0

                    # This converts it to total loss across all samples in batch
                    epoch_losses[k] += v * batch_size
                samples_seen += batch_size

            epoch_losses = {k: v / samples_seen for k, v in epoch_losses.items()}
            if (self.epoch + 1) % self.args.save_freq == 0:
                self.save(self.epoch, epoch_losses["loss"])

            log_stmnt = f"Epoch {self.epoch + 1}: Avg Loss per Sample {epoch_losses['loss']:.3e} {time.time() - start:.2f}s"
            print(log_stmnt, "\n")

            log_parts = [
                str(self.epoch),
                f"{time.time() - start:.2f}"
            ]

            loss_components = sorted(epoch_losses.keys())
            log_parts.extend([f"{epoch_losses[k]}" for k in loss_components])

            log_stmnt = ",".join(log_parts)
            with open(os.path.join(self.args.log_dir, "logs.csv"), "a") as f:
                f.write(log_stmnt + "\n")

            self.epoch += 1

    def create_log_dir(self):
        losses = self.test_forward_pass()

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

        loss_components = sorted(losses.keys())
        with open(os.path.join(self.args.log_dir, "logs.csv"), "w") as f:
            f.write("epoch,time," + ','.join(loss_components) + "\n")

        print(f"Dataset: {str(self.train_data_loader.dataset)}")
        print(f"Normalization: {self.train_data_loader.dataset.normalizer}")

        def args_to_str(model):
            return "\n".join([dataclass_to_table(arg) for arg in model.args])

        with open(os.path.join(self.args.log_dir, "overview.txt"), "w") as f:
            f.write(dataclass_to_table(self.args) + "\n")
            f.write(args_to_str(self.model.module) if self.multi_gpu else args_to_str(self.model))
            f.write(f"\nDataset: {str(self.train_data_loader.dataset)}\n")
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
    def load(filepath, get_ema=True) -> Tuple[nn.Module, nn.Module, Normalizer, TrainerArgs]:
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
        if get_ema:
            ema_model.load_state_dict(state_dict["ema_model_state"])

        return (
            model,
            ema_model,
            state_dict["normalizer"],
            state_dict["trainer_args"],
        )

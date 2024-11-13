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
from quadrotor_diffusion.models.diffusion import DiffusionWrapper
from quadrotor_diffusion.utils.dataset.normalizer import Normalizer
from quadrotor_diffusion.utils.logging import (
    dataclass_to_table,
    iprint as print
)


class Trainer:
    def __init__(
        self,
        args: TrainerArgs,
        model: DiffusionWrapper,
        dataset: Dataset,
    ):
        self.args = args
        if args.num_gpus == -1:
            args.num_gpus = torch.cuda.device_count()

        if args.num_gpus > 1:
            self.model = nn.DataParallel(model)
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

    def test_forward_pass(self):
        start = time.time()
        first_item: torch.Tensor = next(iter(self.train_data_loader)).to(self.args.device)
        if isinstance(self.model, nn.DataParallel):
            self.model.module.compute_loss(first_item)
        else:
            self.model.compute_loss(first_item)
        duration = time.time() - start
        print(f"Forward pass succeeded in {duration:.2f}")

    def train(self):
        epoch = 0
        while self.args.max_epochs is None or epoch < self.args.max_epochs:
            epoch_loss = 0.0

            start = time.time()
            for batch in tqdm.tqdm(self.train_data_loader, desc=f"Epoch {epoch+1}"):
                batch = batch.to(self.args.device)
                self.batches_seen += 1

                loss = self.model.module.compute_loss(batch) if isinstance(
                    self.model, nn.DataParallel) else self.model.compute_loss(batch)
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
                        self.model.module if isinstance(self.model, nn.DataParallel) else self.model,
                        self.args.ema_decay
                    )

                loss = loss.detach().cpu().numpy()
                loss *= self.args.batches_per_backward
                epoch_loss += loss

            epoch_loss /= len(self.train_data_loader.dataset)
            if (epoch + 1) % self.args.save_freq == 0:
                self.save(epoch, epoch_loss)

            log_stmnt = f"Epoch {epoch + 1}: Avg Loss per Sample {epoch_loss:.1e} {time.time() - start:.2f}s"
            print(log_stmnt, "\n")

            log_stmnt = f"{epoch},{epoch_loss},{time.time() - start:.2f}"
            with open(os.path.join(self.args.log_dir, "logs.csv"), "a") as f:
                f.write(log_stmnt + "\n")

            epoch += 1

    def create_log_dir(self):
        now = datetime.now()
        date_str = now.strftime("%b.%d_%I:%M_%p")

        dirs = [d for d in os.listdir(self.args.log_dir) if os.path.isdir(os.path.join(self.args.log_dir, d))]
        max_num = max([int(d.split('.')[0]) for d in dirs if d.split('.')[0].isdigit()], default=0)

        new_dir_name = f"{max_num + 1}.{date_str}"
        new_dir_path = os.path.join(self.args.log_dir, new_dir_name)

        os.makedirs(new_dir_path)
        print(f"Save directory created: {new_dir_path}")
        self.args.log_dir = new_dir_path

        with open(os.path.join(self.args.log_dir, "logs.csv"), "w") as f:
            f.write("epoch,loss,time\n")

        print(f"Dataset: {type(self.train_data_loader.dataset)}")
        print(f"Normalization: {self.train_data_loader.dataset.normalizer}\n")

        with open(os.path.join(self.args.log_dir, "overview.txt"), "w") as f:
            f.write(dataclass_to_table(self.args) + "\n")
            f.write(dataclass_to_table(self.model.module.diffusion_args if isinstance(
                self.model, nn.DataParallel) else self.model.diffusion_args) + "\n")
            f.write(dataclass_to_table(self.model.module.unet_args if isinstance(
                self.model, nn.DataParallel) else self.model.unet_args) + "\n")
            f.write(f"\nDataset: {type(self.train_data_loader.dataset)}\n")
            f.write(f"Normalization: {self.train_data_loader.dataset.normalizer}\n\n")

    def save(self, epoch: int, loss: float):
        filename = os.path.join(self.args.log_dir, f"epoch_{epoch}_loss_{loss:.4f}")
        torch.save(
            {
                "model_state": self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(),
                "ema_model_state": self.ema_model.state_dict() if self.ema_model is not None else dict(),
                "normalizer": self.normalizer,
                "diffusion_args": self.model.module.diffusion_args if isinstance(self.model, nn.DataParallel) else self.model.diffusion_args,
                "unet_args": self.model.module.unet_args if isinstance(self.model, nn.DataParallel) else self.model.unet_args,
                "trainer_args": self.args,
            }, filename
        )
        print(f"Saved to {filename}")

    @staticmethod
    def load(filepath) -> Tuple[DiffusionWrapper, DiffusionWrapper, Normalizer, TrainerArgs]:
        state_dict = torch.load(filepath)

        model = DiffusionWrapper(state_dict["diffusion_args"], state_dict["unet_args"])
        model.load_state_dict(state_dict["model_state"])

        ema_model = DiffusionWrapper(state_dict["diffusion_args"], state_dict["unet_args"])
        ema_model.load_state_dict(state_dict["ema_model_state"])

        return (
            model,
            ema_model,
            state_dict["normalizer"],
            state_dict["trainer_args"],
        )

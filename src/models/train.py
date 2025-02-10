"""Module for training with pines flowers images"""

from datetime import datetime
import enum
import os
from types import SimpleNamespace

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np

import dataset
from engine import train_one_epoch, evaluate
import models
from utils import LOGGER, files_manipulator

LOSS_TYPES = [
    "loss_classifier",
    "loss_objectness",
    "loss_rpn_box_reg",
    "loss_box_reg",
    "loss_sum",
]


class Metric(enum.Enum):
    LOSS = 1
    AP = -1


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation value is less than the previous least less, then save the
    model state.
    """

    def __init__(
        self, best_valid_value: float, metric: Metric, output_dir: str
    ) -> None:
        self.best_valid_value: float = best_valid_value
        self.metric: Metric = metric
        self.output_dir: str = output_dir

    def __call__(
        self,
        current_valid_value: float,
        epoch: int,
        model_state_dict,
        optimizer_state_dict,
    ):
        LOGGER.info(
            f"model_saver {self.metric.name} : curr {current_valid_value}, best {self.best_valid_value}"
        )
        if (
            self.metric.value * current_valid_value
            < self.metric.value * self.best_valid_value
        ):
            self.best_valid_value = current_valid_value
            LOGGER.info(
                f"Saving best validation {self.metric.name}: {self.best_valid_value} for epoch {epoch}"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_state_dict,
                    "optimizer_state_dict": optimizer_state_dict,
                },
                f"{self.output_dir}/best_model_{self.metric.name}.pth",
            )

    @staticmethod
    def save_last(epoch, model_state_dict, optimizer_state_dict, output_dir):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": optimizer_state_dict,
            },
            f"{output_dir}/last_model.pth",
        )


def load_hyperparameters(hyperparameters_file_path: str) -> SimpleNamespace:
    return SimpleNamespace(
        **files_manipulator.load_json(hyperparameters_file_path)
    )


def get_sets(scale):
    from dataset import PinesDataset
    from torch.utils.data import random_split

    pines_dataset = PinesDataset("/mnt/sda1/1-Pins/DATASET/", scale=scale)
    train_set, val_set, _ = random_split(
        dataset=pines_dataset,
        lengths=[0.8, 0.1, 0.1],
        generator=torch.Generator().manual_seed(42),
    )
    # Method 2
    # train_set = torch.load(
    #     os.path.join(file_cwd, "../../datasets/DATASET_train_set.pt")
    # )
    # val_set = torch.load(
    #     os.path.join(file_cwd, "../../datasets/DATASET_validation_set.pt")
    # )
    # LOGGER.info("Subsets train and val loaded")
    return train_set, val_set


def main():
    now = datetime.now().strftime("%Y%m%d_%H:%M:%S")
    file_cwd = os.path.dirname(__file__)

    hyp = load_hyperparameters(
        os.path.join(file_cwd, "cfg/hyperparameters.json")
    )
    # paths = files.load_json("cfg/dataset_settings.json")
    writer = SummaryWriter(os.path.join(file_cwd, "../../runs/train-" + now))
    LOGGER.info("SummaryWriter is setup in runs/train-" + now)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("%s is available" % device)

    train_set, val_set = get_sets(scale=hyp.scale)
    LOGGER.info("Subsets train and val loaded.")

    train_dataloader = DataLoader(
        train_set, batch_size=hyp.batch_size, collate_fn=dataset.collate_fn
    )
    validation_dataloader = DataLoader(
        val_set, batch_size=hyp.batch_size, collate_fn=dataset.collate_fn
    )

    model = models.get_fasterrcnn_mobilenet_v3().to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=hyp.learning_rate,
        momentum=hyp.momentum,
        weight_decay=hyp.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=hyp.mode,
        factor=hyp.factor,
        patience=hyp.patience,
        verbose=True,
    )

    output_dir_now = os.path.join(file_cwd, "../../outputs/", now)
    os.makedirs(output_dir_now, exist_ok=True)
    LOGGER.info(f"output dir created at {output_dir_now}")

    best_model_loss_saver = SaveBestModel(
        float("inf"), Metric.LOSS, output_dir_now
    )
    best_model_ap_saver = SaveBestModel(0.0, Metric.AP, output_dir_now)
    for epoch in range(hyp.num_epochs):
        train_metric_logger = train_one_epoch(
            model, optimizer, train_dataloader, device, epoch, print_freq=1
        )
        evaluator, eval_metric_logger = evaluate(
            model, validation_dataloader, device, epoch, print_freq=1
        )
        s = evaluator.coco_eval.eval["precision"][:, :, :, 0, 2]
        current_valid_ap = np.mean(s[s > -1])
        current_valid_loss = eval_metric_logger.loss_sum.avg
        # Assuming `train_metric_logger` and `eval_metric_logger` collect various losses
        # Here, we're assuming these loggers have attributes for different types of losses
        # For example, `loss_classifier`, `loss_box_reg`, etc.

        for loss_type in LOSS_TYPES:
            writer.add_scalar(
                f"Train/{loss_type}",
                getattr(train_metric_logger, loss_type).avg,
                epoch,
            )
            writer.add_scalar(
                f"Eval/{loss_type}",
                getattr(eval_metric_logger, loss_type).avg,
                epoch,
            )

        writer.add_scalars(
            "Eval/times",
            {
                "model": eval_metric_logger.model_time.avg,
                "eval": eval_metric_logger.evaluator_time.avg,
            },
            epoch,
        )
        writer.add_scalar("AP", current_valid_ap, epoch)

        # Log learning rate
        for i, param_group in enumerate(optimizer.param_groups):
            writer.add_scalar(
                f"LearningRate/Group_{i}", param_group["lr"], epoch
            )

        best_model_ap_saver(
            current_valid_ap,
            epoch,
            model.state_dict(),
            optimizer.state_dict(),
        )
        best_model_loss_saver(
            current_valid_loss,
            epoch,
            model.state_dict(),
            optimizer.state_dict(),
        )
        if epoch % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                f"{output_dir_now}/checkpoint-epoch-{epoch}.pth",
            )
        scheduler.step(current_valid_loss)

    writer.close()
    SaveBestModel.save_last(
        hyp.num_epochs - 1,
        model.state_dict(),
        optimizer.state_dict(),
        output_dir_now,
    )

    LOGGER.info("Finished Training")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

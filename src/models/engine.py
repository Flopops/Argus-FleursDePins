import time

import torch
from torch.utils.data import DataLoader

import dataset
import coco_eval
import tracking


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]],
    device: str | torch.device,
):
    images, targets = dataset.prepare_batch(batch, device)
    optimizer.zero_grad()
    loss_dict = model(images, targets)
    loss: torch.Tensor = sum(loss_dict.values())
    loss.backward()
    optimizer.step()
    loss_dict["loss_sum"] = loss
    return loss_dict


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: str | torch.device,
    epoch: int,
    print_freq: int,
):
    """
    Inspired by https://github.com/pytorch/vision/blob/main/references/detection/engine.py
    """
    model.train()
    metric_logger = tracking.MetricLogger()
    for batch in metric_logger.log_every(
        data_loader, print_freq, header=f"Train: [{epoch + 1}]"
    ):
        loss_dict = train_step(model, optimizer, batch, device)
        metric_logger.update(**loss_dict)
    return metric_logger


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    print_freq: int,
):
    """
    Inspired by https://github.com/pytorch/vision/blob/main/references/detection/engine.py
    """
    metric_logger = tracking.MetricLogger()
    evaluator = coco_eval.CocoEvaluator(ds=data_loader.dataset)
    for batch in metric_logger.log_every(
        data_loader, print_freq, f"Test: [{epoch+1}]"
    ):
        images, targets = dataset.prepare_batch(batch, device)
        model.eval()
        model_time = time.time()
        detections = model(images)
        model_time = time.time() - model_time
        model.train()
        model_train_time = time.time()
        losses = model(images, targets)
        model_time = time.time() - model_train_time
        losses["loss_sum"] = sum(losses.values())

        predictions = {
            t["image_id"].item(): d for t, d in zip(targets, detections)
        }
        evaluator_time = time.time()
        evaluator.update(predictions)
        evaluator_time = time.time() - evaluator_time

        metric_logger.update(
            model_time=model_time,
            evaluator_time=evaluator_time,
            **losses,
        )
    evaluator.accumulate()
    evaluator.summarize()
    return evaluator, metric_logger

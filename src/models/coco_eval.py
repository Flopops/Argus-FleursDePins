import contextlib
import io
import copy
import abc

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch.distributed as dist
import torchvision
from torch.utils.data import Subset

import numpy as np
import torch


class AbstractEvaluator(abc.ABC):
    def __init__(self):
        self.img_ids = []
        self.eval_imgs = []

    @abc.abstractmethod
    def update(self, predictions):
        ...

    @abc.abstractmethod
    def synchronize_between_processes(self):
        ...

    @abc.abstractmethod
    def prepare(self, predictions):
        ...

    @abc.abstractmethod
    def accumulate(self):
        ...

    @abc.abstractmethod
    def summarize(self):
        ...


class CocoEvaluator(AbstractEvaluator):
    def __init__(self, ds):
        super().__init__()
        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt = get_coco_api_from_dataset(ds)
        self.coco_gt = copy.deepcopy(coco_gt)
        self.coco_eval = COCOeval(coco_gt, iouType="bbox")

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)
        results = self.prepare(predictions)
        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = (
                COCO.loadRes(self.coco_gt, results) if results else COCO()
            )
        self.coco_eval.cocoDt = coco_dt
        self.coco_eval.params.imgIds = list(img_ids)
        self.eval_imgs.append(evaluate_images(self.coco_eval)[1])

    def synchronize_between_processes(self):
        self.eval_imgs = np.concatenate(self.eval_imgs, 2)
        create_common_coco_eval(
            self.coco_eval,
            self.img_ids,
            self.eval_imgs,
        )

    def prepare(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue
            boxes = convert_to_xywh(prediction["boxes"]).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": label,
                        "bbox": box,
                        "score": score,
                    }
                    for label, box, score in zip(labels, boxes, scores)
                ]
            )
        return coco_results

    def accumulate(self):
        self.coco_eval.accumulate()

    def summarize(self):
        self.coco_eval.summarize()


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def merge(img_ids, eval_imgs):
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs) -> None:
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    setattr(coco_eval, "_paramsEval", copy.deepcopy(coco_eval.params))
    # coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


def evaluate_images(imgs):
    with contextlib.redirect_stdout(io.StringIO()):
        imgs.evaluate()
    return imgs.params.imgIds, np.asarray(imgs.evalImgs).reshape(
        -1, len(imgs.params.areaRng), len(imgs.params.imgIds)
    )


def convert_to_coco_api(ds):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    for img_idx, _ in enumerate(ds):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]
        image_id = targets["image_id"].item()
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2]
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"].clone()
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            dataset["annotations"].append(
                {
                    "image_id": image_id,
                    "bbox": bboxes[i],
                    "category_id": labels[i],
                    "area": areas[i],
                    "iscrowd": iscrowd[i],
                    "id": ann_id,
                }
            )
            categories.add(labels[i])
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    return convert_to_coco_api(dataset)

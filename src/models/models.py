"""Models module"""

import torch
from torchvision.models import detection
from sahi.utils.torch import to_float_tensor
from sahi.predict import ObjectPrediction
from sahi.models.base import DetectionModel
import numpy as np

from utils import LOGGER


NUM_CLASSES = 2  # 1 class (flower) + background


def get_fasterrcnn_mobilenet_v3(
    weights_path: str | None = None,
    box_score_thresh=0.9,
) -> detection.FasterRCNN:
    model = detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
        box_score_thresh=box_score_thresh,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(
        in_features, NUM_CLASSES
    )
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        LOGGER.info("FasterRCNN loaded with %s", weights_path)
    else:
        LOGGER.info("FasterRCNN loaded with no pretrained weights")
    return model


class PinesDetectionModel(DetectionModel):
    def set_model(self, model):
        model.eval()
        self.model = model.to(self.device)
        self.category_mapping = {"0": "__background__", "1": "flower"}

    def perform_inference(
        self, image: np.ndarray, image_size: int = 320
    ) -> None:
        # arrange model input size
        if self.image_size is not None:
            # get min and max of image height and width
            min_shape, max_shape = min(image.shape[:2]), max(image.shape[:2])
            # torchvision resize transform scales the shorter dimension to the target size
            # we want to scale the longer dimension to the target size
            image_size: float = self.image_size * min_shape / max_shape
            self.model.transform.min_size = (image_size,)  # default is (800,)
            self.model.transform.max_size = image_size  # default is 1333

        # Same as image_torch_to_np
        image: torch.Tensor = to_float_tensor(image)
        image = image.to(self.device)
        prediction_result = self.model([image])
        self._original_predictions = prediction_result

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        # return 2
        return len(self.category_mapping)

    @property
    def has_mask(self):
        return False

    @property
    def category_names(self):
        return list(self.category_mapping.values())

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: list[list[int]] = [[0, 0]],
        full_shape_list: list[list[int]] | None = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions

        # compatilibty for sahi v0.8.20
        if isinstance(shift_amount_list[0], int):
            shift_amount_list = [shift_amount_list]
        if full_shape_list is not None and isinstance(full_shape_list[0], int):
            full_shape_list = [full_shape_list]

        object_prediction_list_per_image = []
        for image_predictions in original_predictions:
            # get indices of boxes with score > confidence_threshold
            scores = image_predictions["scores"].cpu().detach().numpy()
            selected_indices = np.where(scores > self.confidence_threshold)[0]

            # parse boxes, masks, scores, category_ids from predictions
            category_ids = list(
                image_predictions["labels"][selected_indices]
                .cpu()
                .detach()
                .numpy()
            )
            boxes = list(
                image_predictions["boxes"][selected_indices]
                .cpu()
                .detach()
                .numpy()
            )
            scores = scores[selected_indices]
            shift_amount = shift_amount_list[0]
            full_shape = (
                None if full_shape_list is None else full_shape_list[0]
            )
            object_prediction_list_per_image.append(
                [
                    ObjectPrediction(
                        bbox=b,
                        bool_mask=None,
                        category_id=int(category_ids[i]),
                        category_name=self.category_mapping[
                            str(int(category_ids[i]))
                        ],
                        shift_amount=shift_amount,
                        score=scores[i],
                        full_shape=full_shape,
                    )
                    for i, b in enumerate(boxes)
                ]
            )

        self._object_prediction_list_per_image = (
            object_prediction_list_per_image
        )

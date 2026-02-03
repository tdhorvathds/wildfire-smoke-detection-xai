from dataclasses import dataclass
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

@dataclass
class XaiTrainConfig:
    pretrained: bool = True
    num_classes: int = 2
    label_smoothing: float = 0.1

def get_fasterrcnn_model_xai(cfg: XaiTrainConfig) -> nn.Module:
    model = fasterrcnn_resnet50_fpn_v2(
        weights="DEFAULT" if cfg.pretrained else None,
        weights_backbone=None,
    )

    anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios,
    )

    model.rpn.fg_iou_thresh = 0.4
    model.rpn.bg_iou_thresh = 0.1
    model.rpn.pre_nms_top_n_train = 4000
    model.rpn.post_nms_top_n_train = 2000
    model.rpn.pre_nms_top_n_test  = 2000
    model.rpn.post_nms_top_n_test = 1000

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, cfg.num_classes)
    model.roi_heads.detections_per_img = 200

    if cfg.label_smoothing > 0:
        model.roi_heads.fastrcnn_loss_func = torch.nn.CrossEntropyLoss(
            label_smoothing=cfg.label_smoothing
        )
    else:
        model.roi_heads.fastrcnn_loss_func = torch.nn.CrossEntropyLoss()

    return model

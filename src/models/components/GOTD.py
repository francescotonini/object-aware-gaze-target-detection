import math
import torch
import torch.nn as nn
from torch.utils import model_zoo
from torch_intermediate_layer_getter import IntermediateLayerGetter

from libs.detr.models import build_model as build_detr
from src import utils
from src.models.components.GazeTransformer import (
    GazeTransformerLayer,
    GazeTransformer,
)
from src.models.components.MLP import MLP
from src.utils.AttributeDict import AttributeDict
from src.utils.gaze_ops import get_gaze_cone
from src.utils.misc import load_pretrained, get_annotation_id

log = utils.get_pylogger(__name__)


class GOTD(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_queries: int,
        num_gaze_queries: int,
        gaze_heatmap_size: int,
        num_gaze_decoder_layers: int,
        gaze_vector_type: str = "2d",
        gaze_cone_angle: int = 120,
    ):
        super().__init__()

        self.dim_feedforward = 2048
        self.hidden_dim = 256
        self.nhead = nhead = 8
        self.dropout = dropout = 0.1

        # Setup backbone
        self.backbone, _, _ = build_detr(
            AttributeDict(
                {
                    "dataset_file": "coco",
                    "device": "cuda",  # TODO: override this using hydra config
                    "num_queries": num_queries,
                    "aux_loss": False,
                    "masks": False,
                    "eos_coef": 0,
                    "hidden_dim": self.hidden_dim,
                    "position_embedding": "sine",
                    "lr_backbone": 1e-5,
                    "backbone": "resnet50",
                    "dilation": False,
                    "nheads": nhead,
                    "dim_feedforward": self.dim_feedforward,
                    "enc_layers": 6,
                    "dec_layers": 6,
                    "dropout": dropout,
                    "pre_norm": False,
                    "set_cost_class": 1,
                    "set_cost_bbox": 5,
                    "set_cost_giou": 2,
                    "mask_loss_coef": 1,
                    "bbox_loss_coef": 5,
                    "giou_loss_coef": 2,
                }
            )
        )
        log.info("Loading DETR pretrained weights")
        load_pretrained(
            self.backbone,
            model_zoo.load_url(
                "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"
            ),
        )
        # Extract features from the object detection backbone
        self.backbone_getter = IntermediateLayerGetter(
            self.backbone, return_layers={"transformer": "hs"}
        )

        # Setup object detector MLPs
        self.class_embed = MLP(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=num_classes + 1,
            num_layers=1,
        )
        self.bbox_embed = self.head_bbox_embed = MLP(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=4,
            num_layers=3,
        )

        # Setup gaze vector MLP
        self.gaze_vector_type = gaze_vector_type
        vector_output_dim = 2 if gaze_vector_type == "2d" else 3
        self.gaze_cone_angle = gaze_cone_angle
        self.gaze_vector_embed = MLP(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=vector_output_dim,
            num_layers=2,
        )

        # Setup gaze query embeddings
        self.num_gaze_queries = num_gaze_queries
        self.gaze_embed = self.gaze_query_embed = nn.Embedding(
            self.num_gaze_queries, self.hidden_dim
        )

        # Setup gaze transformer
        self.gaze_transformer = GazeTransformer(
            GazeTransformerLayer(
                self.hidden_dim,
                nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=dropout,
                activation="relu",
            ),
            num_gaze_decoder_layers,
            norm=nn.LayerNorm(self.hidden_dim),
        )

        # Setup heatmap and watch outside MLPs
        self.gaze_watch_outside_embed = MLP(self.hidden_dim, self.hidden_dim, 1, 1)
        self.gaze_heatmap_obj_embed = MLP(
            self.hidden_dim,
            self.hidden_dim,
            gaze_heatmap_size**2,
            5,
        )
        self.gaze_heatmap_no_obj_embed = MLP(
            self.hidden_dim,
            self.hidden_dim,
            gaze_heatmap_size**2,
            5,
        )

    def forward(self, samples, depths, img_sizes):
        backbone_intermediate_layers, _ = self.backbone_getter(samples)
        object_detection_decoder_features = backbone_intermediate_layers["hs"][0]
        object_detection_decoder_embed = self.backbone.query_embed.weight.repeat(
            object_detection_decoder_features.shape[0],
            object_detection_decoder_features.shape[1],
            1,
            1,
        )

        outputs_logits = self.class_embed(object_detection_decoder_features)
        outputs_bbox = self.bbox_embed(object_detection_decoder_features).sigmoid()
        outputs_labels = outputs_logits.argmax(-1)

        sort_idx = outputs_labels.argsort(dim=-1, descending=False)
        outputs_logits = outputs_logits.gather(
            2,
            sort_idx.unsqueeze(-1).expand(-1, -1, -1, outputs_logits.shape[-1]),
        )
        outputs_bbox = outputs_bbox.gather(
            2, sort_idx.unsqueeze(-1).expand(-1, -1, -1, outputs_bbox.shape[-1])
        )
        object_detection_decoder_features = object_detection_decoder_features.gather(
            2,
            sort_idx.unsqueeze(-1).expand(
                -1, -1, -1, object_detection_decoder_features.shape[-1]
            ),
        )
        object_detection_decoder_embed = object_detection_decoder_embed.gather(
            2,
            sort_idx.unsqueeze(-1).expand(
                -1, -1, -1, object_detection_decoder_embed.shape[-1]
            ),
        )

        # There are 6 layers in the decoder, we only want the last one
        object_detection_decoder_features = object_detection_decoder_features[-1:]
        object_detection_decoder_embed = object_detection_decoder_embed[-1:]
        outputs_bbox = outputs_bbox[-1:]
        outputs_logits = outputs_logits[-1:]
        # outputs_labels = outputs_labels[-1:]

        # Keep only the first max_objects objects
        object_detection_decoder_features = object_detection_decoder_features[
            :, :, : self.num_gaze_queries
        ]
        object_detection_decoder_embed = object_detection_decoder_embed[
            :, :, : self.num_gaze_queries
        ]
        outputs_bbox = outputs_bbox[:, :, : self.num_gaze_queries]
        outputs_logits = outputs_logits[:, :, : self.num_gaze_queries]
        # outputs_labels = outputs_labels[:, :, : self.num_gaze_queries]

        # This mask is true for all the objects that are faces
        face_presence = torch.logical_and(
            outputs_logits.argmax(dim=-1) == get_annotation_id("face"),
            outputs_logits.max(dim=-1).values > 0.5,
        )

        # This mask is true for all the objects that are not background and confidence > 0.5
        # NOTE: it includes faces
        objects_presence = torch.logical_and(
            outputs_logits.argmax(dim=-1) != get_annotation_id("no-object"),
            outputs_logits.max(dim=-1).values > 0.5,
        )

        gaze_vectors = self.gaze_vector_embed(
            object_detection_decoder_features
        ).sigmoid()

        # For each batch size, get the depth value at the center of the bounding box
        if self.gaze_vector_type == "3d":
            bbox_center = outputs_bbox[..., 0:2]

            # [b, 1, w, h]
            bbox_center_orig_size = bbox_center.clone()
            bbox_center_orig_size[..., 0] *= img_sizes[:, 1].unsqueeze(1)
            bbox_center_orig_size[..., 1] *= img_sizes[:, 0].unsqueeze(1)

            # Get z-axis value at the center of the bounding box
            bbox_center_z = depths.tensors[
                torch.arange(bbox_center.shape[1]).unsqueeze(1),
                0,
                bbox_center_orig_size[0, :, :, 1].long(),
                bbox_center_orig_size[0, :, :, 0].long(),
            ].unsqueeze(0)

            # [l, b, q, 2] -> [lbq, 3]
            flatten_bbox_center = torch.cat(
                [
                    outputs_bbox[..., 0:2].flatten(0, 2),
                    bbox_center_z.flatten().unsqueeze(-1),
                ],
                dim=-1,
            )
        else:
            # [l, b, q, 2] -> [lbq, 2]
            flatten_bbox_center = outputs_bbox[..., 0:2].flatten(0, 2)

        # [l, b, q, 2] -> [lbq, 2]
        flatten_gaze_vectors = gaze_vectors.flatten(0, 2)

        if self.gaze_vector_type == "3d":
            # Convert to dx, dy, dz
            # phi, theta, and magnitude are in range [0, 1]
            phi = flatten_gaze_vectors[..., 0]
            theta = flatten_gaze_vectors[..., 1]
            magnitude = flatten_gaze_vectors[..., 2]

            # Rescale magnitude
            magnitude = magnitude * math.sqrt(3)

            # Rescale phi and theta
            phi = (phi * (2 * math.pi)) - math.pi
            theta = (theta * math.pi) - (math.pi / 2)

            flatten_gaze_vectors = torch.stack(
                [
                    magnitude * torch.cos(phi) * torch.cos(theta),
                    magnitude * torch.sin(phi) * torch.cos(theta),
                    magnitude * torch.sin(theta),
                ],
                dim=-1,
            )
        elif self.gaze_vector_type == "2d":
            # Convert to dx, dy
            phi = flatten_gaze_vectors[..., 0]
            magnitude = flatten_gaze_vectors[..., 1]

            # Rescale magnitude
            magnitude = magnitude * math.sqrt(2)

            # Rescale phi
            phi = (phi * (2 * math.pi)) - math.pi

            flatten_gaze_vectors = torch.stack(
                [
                    magnitude * torch.cos(phi),
                    magnitude * torch.sin(phi),
                ],
                dim=-1,
            )

        # [lbq, 2] -> [lbq, 64, 64]
        gaze_cones = get_gaze_cone(
            flatten_bbox_center,
            flatten_gaze_vectors,
            out_size=(64, 64, 64) if self.gaze_vector_type == "3d" else (64, 64),
            cone_angle=self.gaze_cone_angle,
        )

        # [lbq, 2] -> [l, b, q, 64, 64]
        if self.gaze_vector_type == "3d":
            gaze_cones = gaze_cones.reshape(*outputs_bbox.shape[:-1], 64, 64, 64)
            center_bbox = flatten_bbox_center.reshape(*outputs_bbox.shape[:-1], 3)
        else:
            gaze_cones = gaze_cones.reshape(*outputs_bbox.shape[:-1], 64, 64)
            center_bbox = outputs_bbox[..., 0:2]

        cone_shape = gaze_cones.shape
        l, b, q = cone_shape[0], cone_shape[1], cone_shape[2]

        # Rescale the center of the bbox to the gaze cone size
        center_bbox = (center_bbox * (64 - 1)).long()

        L_tensor = torch.arange(l, dtype=torch.long, device=gaze_cones.device)
        B_tensor = torch.arange(b, dtype=torch.long, device=gaze_cones.device)
        Q_tensor = torch.arange(q, dtype=torch.long, device=gaze_cones.device)
        I, J, K, M = torch.meshgrid(L_tensor, B_tensor, Q_tensor, Q_tensor)

        # NOTE: flipped coords
        if self.gaze_vector_type == "3d":
            objects_scores = gaze_cones[
                I,
                J,
                K,
                center_bbox[I, J, M, 0],
                center_bbox[I, J, M, 1],
                center_bbox[I, J, M, 2],
            ]
        else:
            objects_scores = gaze_cones[
                I, J, K, center_bbox[I, J, M, 0], center_bbox[I, J, M, 1]
            ]

        # Zero out scores of not objects nor faces
        objects_scores = objects_scores * (
            objects_presence.unsqueeze(-2) * face_presence.unsqueeze(-1)
        )
        # Zero out diagonal of objects_scores
        objects_scores = objects_scores * (
            1 - torch.eye(q, device=objects_scores.device)
        )

        object_detection_decoder_features = object_detection_decoder_features[
            -1
        ].permute(1, 0, 2)
        object_detection_decoder_embed = object_detection_decoder_embed[-1].permute(
            1, 0, 2
        )
        num_queries, bs, feat_dim = object_detection_decoder_features.shape

        gaze_query_embed = self.gaze_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        gaze_decoder_tgt = torch.zeros_like(gaze_query_embed)

        faces_without_objects_presence = torch.logical_and(
            face_presence, objects_scores.sum(dim=-1) == 0
        )
        faces_with_objects_presence = torch.logical_and(
            face_presence, ~faces_without_objects_presence
        )

        objects_hs, sa_attn_weights, ca_attn_weights = self.gaze_transformer(
            sa=gaze_decoder_tgt,
            ca=object_detection_decoder_features,
            sa_mask=face_presence.permute(2, 1, 0),
            ca_mask=objects_presence.permute(2, 1, 0),
            attn_booster=objects_scores[-1],
            sa_pos=gaze_query_embed,
            ca_pos=object_detection_decoder_embed,
        )

        objects_hs = objects_hs[-1:] + object_detection_decoder_features

        objects_hs_masked = objects_hs.masked_fill(
            ~faces_with_objects_presence.permute(0, 2, 1).unsqueeze(-1), 0
        ).transpose(1, 2)
        hs_masked = (
            object_detection_decoder_features.unsqueeze(0)
            .masked_fill(
                ~faces_without_objects_presence.permute(0, 2, 1).unsqueeze(-1),
                0,
            )
            .transpose(1, 2)
        )

        outputs_gaze_heatmap_obj = self.gaze_heatmap_obj_embed(objects_hs_masked)
        outputs_gaze_heatmap_no_obj = self.gaze_heatmap_no_obj_embed(hs_masked)
        outputs_gaze_heatmap = outputs_gaze_heatmap_obj + outputs_gaze_heatmap_no_obj

        outputs_watch_outside = self.gaze_watch_outside_embed(
            objects_hs_masked + hs_masked
        ).sigmoid()

        out = {
            "pred_logits": outputs_logits[-1],
            "pred_boxes": outputs_bbox[-1],
            "pred_gaze_vectors": gaze_vectors[-1],
            "pred_gaze_heatmap": outputs_gaze_heatmap[-1],
            "pred_gaze_watch_outside": outputs_watch_outside[-1],
            "pred_gaze_cone": gaze_cones[-1],
            "sa_attn_weights": sa_attn_weights[-1],
            "ca_attn_weights": ca_attn_weights[-1],
            "objects_scores": objects_scores[-1],
        }

        return out

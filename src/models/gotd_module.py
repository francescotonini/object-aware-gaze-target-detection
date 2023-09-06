import math
from typing import Any, List

import pytorch_lightning.loggers
import seaborn as sns
import torch
import torchvision.transforms.functional as TF
import wandb
from matplotlib import pyplot as plt, patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch_lightning import LightningModule
from torch.utils import model_zoo
from torchmetrics import MeanMetric

from src.utils.box_ops import box_cxcywh_to_xyxy
from src.utils.misc import (
    get_annotations,
    load_pretrained,
    get_annotation_id,
    fig2img,
    unnorm,
)
from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class GOTDLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        criterion: torch.nn.Module,
        evaluation: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        net_pretraining: str = None,
        gaze_vector_type: str = "2d",
        n_of_images_to_log: int = 8,
    ):
        super().__init__()

        # This line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "net",
                "criterion",
                "evaluation",
            ],
        )

        self.net = net
        self.criterion = criterion
        self.evaluation = evaluation
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        if net_pretraining is not None:
            log.info(f"Loading pretrained model from {net_pretraining}")

            # Check whether net_pretraining is path or url
            if net_pretraining.startswith("http"):
                load_pretrained(self.net, model_zoo.load_url(net_pretraining))
            else:
                # Assuming we are loading a pytorch lightning checkpoint
                load_pretrained(
                    self.net,
                    torch.load(net_pretraining, map_location="cpu"),
                    drop_prefix="net.",
                )

    def on_start(self):
        # Clean cache
        torch.cuda.empty_cache()
        self.evaluation.reset()

    def on_train_start(self):
        self.on_start()

    def on_validation_start(self):
        self.on_start()

    def step(self, batch: Any, do_eval: bool = False):
        samples, depths, img_sizes, targets = batch
        outputs = self.net(samples, depths, img_sizes)
        losses = self.criterion(outputs, targets)

        if do_eval:
            self.evaluation(outputs, targets)

        return sum(losses.values()), losses, outputs

    def training_step(self, batch: Any, batch_idx: int):
        total_loss, losses, _ = self.step(batch)
        bs = batch[0].tensors.shape[0]

        self.train_loss(total_loss)
        self.log(
            "train/loss",
            self.train_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=bs,
        )

        # Log lr
        for idx, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            self.log(
                f"train/lr_{idx}",
                param_group["lr"],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                batch_size=bs,
            )

        # Log each loss
        for loss_name, loss in losses.items():
            # If loss is nan crash
            if torch.isnan(loss):
                raise ValueError(f"Loss {loss_name} is nan")

            self.log(
                f"train/{loss_name}",
                loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                batch_size=bs,
            )

        return {"loss": total_loss}

    def validation_step(self, batch: Any, batch_idx: int):
        total_loss, losses, outputs = self.step(batch, do_eval=True)
        bs = batch[0].tensors.shape[0]

        self.val_loss(total_loss)
        self.log(
            "val/loss",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=bs,
        )

        # Log each loss
        for loss_name, loss in losses.items():
            self.log(
                f"val/{loss_name}",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=bs,
            )

        # Estimate how many images to log from this batch and batch_size
        n_of_images_logged_so_far = bs * batch_idx
        n_of_images_to_log = self.hparams.n_of_images_to_log
        n_of_images_still_to_log = n_of_images_to_log - n_of_images_logged_so_far
        log_images = n_of_images_still_to_log > 0

        return {
            "loss": total_loss,
            "batch": batch if log_images else None,
            "outputs": outputs if log_images else None,
            "n_of_images_still_to_log": n_of_images_still_to_log,
        }

    def validation_step_end(self, outputs: List[Any]):
        if outputs["batch"] is None:
            return

        n_of_images_still_to_log = outputs["n_of_images_still_to_log"]

        # Get "batch" from the first element of outputs
        batch = outputs["batch"]
        samples, _, _, targets = batch
        outputs = outputs["outputs"]

        # Stop here if logger is not WandbLogger
        if not isinstance(self.logger, pytorch_lightning.loggers.WandbLogger):
            log.warning("Logger is not WandbLogger, skipping logging to wandb")

            return

        # Log images with bounding boxes on wandb
        wandb_od_images = []
        if self.hparams.n_of_images_to_log > 0:
            for idx in range(min(len(samples.tensors), n_of_images_still_to_log)):
                sample = samples.tensors[idx]
                mask = samples.mask[idx]

                # Remove mask from sample
                sample = sample[:, ~mask[:, 0], :]
                sample = sample[:, :, ~mask[0, :]]

                target_box_data = []
                target_boxes = targets[idx]["boxes"]
                target_labels = targets[idx]["labels"]
                for box, label in zip(target_boxes, target_labels):
                    box = box_cxcywh_to_xyxy(box)
                    target_box_data.append(
                        {
                            "position": {
                                "minX": box[0].item(),
                                "minY": box[1].item(),
                                "maxX": box[2].item(),
                                "maxY": box[3].item(),
                            },
                            "class_id": label.argmax().item(),
                            "scores": {"prob": label.max().item()},
                        }
                    )

                pred_box_data = []
                pred_boxes = outputs["pred_boxes"][idx]
                pred_labels = outputs["pred_logits"][idx]
                for box, label in zip(pred_boxes, pred_labels):
                    box = box_cxcywh_to_xyxy(box)
                    pred_box_data.append(
                        {
                            "position": {
                                "minX": box[0].item(),
                                "minY": box[1].item(),
                                "maxX": box[2].item(),
                                "maxY": box[3].item(),
                            },
                            "class_id": label.argmax().item(),
                            "scores": {"prob": label.softmax(-1).max().item()},
                        }
                    )

                img = wandb.Image(
                    sample,
                    boxes={
                        "predictions": {
                            "box_data": pred_box_data,
                            "class_labels": get_annotations(),
                        },
                        "targets": {
                            "box_data": target_box_data,
                            "class_labels": get_annotations(),
                        },
                    },
                )

                wandb_od_images.append(img)

        wandb_gaze_heatmap_images = []
        wandb_attention_images = []
        if self.hparams.n_of_images_to_log > 0:
            # Log gaze heatmaps and attention maps
            for idx in range(min(len(samples.tensors), n_of_images_still_to_log)):
                sample = samples.tensors[idx]
                mask = samples.mask[idx]

                # Remove mask from sample
                sample = sample[:, ~mask[:, 0], :]
                sample = sample[:, :, ~mask[0, :]]

                pred_labels = outputs["pred_logits"][idx].argmax(-1)
                face_idxs = (
                    (pred_labels == get_annotation_id("face"))
                    .nonzero()
                    .flatten()
                    .tolist()
                )

                for face_idx in face_idxs:
                    head_box = (
                        box_cxcywh_to_xyxy(outputs["pred_boxes"][idx][face_idx]).cpu()
                        * torch.tensor([sample.shape[2], sample.shape[1]] * 2)
                    ).numpy()
                    gaze_heatmap = (
                        outputs["pred_gaze_heatmap"][idx][face_idx]
                        .reshape(64, 64)
                        .cpu()
                    )

                    gaze_cone_shape = (64, 64)
                    if self.hparams.gaze_vector_type == "3d":
                        gaze_cone_shape = (64, 64, 64)

                    gaze_cone = (
                        outputs["pred_gaze_cone"][idx][face_idx]
                        .reshape(gaze_cone_shape)
                        .cpu()
                    )

                    if self.hparams.gaze_vector_type == "3d":
                        gaze_cone = gaze_cone.sum(-1)

                    if self.hparams.gaze_vector_type == "2d":
                        gaze_vectors = outputs["pred_gaze_vectors"][idx][face_idx]

                        phi = gaze_vectors[0]
                        magnitude = gaze_vectors[1]

                        # Rescale magnitude
                        magnitude = magnitude * math.sqrt(2)

                        # Rescale phi
                        phi = (phi * (2 * math.pi)) - math.pi

                        gaze_vector_x = magnitude * torch.cos(phi)
                        gaze_vector_y = magnitude * torch.sin(phi)

                        # Multiply by sample size
                        gaze_vector_x *= sample.shape[2]
                        gaze_vector_y *= sample.shape[1]

                        gaze_vectors = (
                            torch.stack([gaze_vector_x, gaze_vector_y], dim=0)
                            .cpu()
                            .numpy()
                        )
                    elif self.hparams.gaze_vector_type == "3d":
                        gaze_vectors = outputs["pred_gaze_vectors"][idx][face_idx]
                        phi = gaze_vectors[0]
                        theta = gaze_vectors[1]
                        magnitude = gaze_vectors[2]

                        magnitude = magnitude * math.sqrt(3)

                        # Rescale phi and theta
                        phi = (phi * (2 * math.pi)) - math.pi
                        theta = (theta * math.pi) - (math.pi / 2)

                        gaze_vector_x = magnitude * torch.cos(phi) * torch.cos(theta)
                        gaze_vector_y = magnitude * torch.sin(phi) * torch.cos(theta)

                        # Multiply by sample size
                        gaze_vector_x *= sample.shape[2]
                        gaze_vector_y *= sample.shape[1]

                        gaze_vectors = (
                            torch.stack([gaze_vector_x, gaze_vector_y], dim=0)
                            .cpu()
                            .numpy()
                        )

                    gaze_heatmap = (
                        TF.resize(
                            gaze_heatmap.unsqueeze(0),  # Add channel dim
                            (sample.shape[1], sample.shape[2]),  # [h, w]
                        ).squeeze(0)
                    ).numpy()

                    gaze_cone = (
                        TF.resize(
                            gaze_cone.permute(1, 0).unsqueeze(0),  # Add channel dim
                            (sample.shape[1], sample.shape[2]),  # [h, w]
                        ).squeeze(0)
                    ).numpy()

                    fig, (
                        ax_bbox,
                        ax_cone,
                        ax_heatmap,
                        ax_heatmap_unscaled,
                    ) = plt.subplots(
                        1,
                        4,
                        figsize=((sample.shape[2] * 4) / 96, sample.shape[1] / 96),
                        dpi=96,
                    )

                    ax_bbox.axis("off")
                    ax_bbox.imshow(
                        unnorm(sample).permute(1, 2, 0).cpu().numpy(), vmin=0, vmax=1
                    )

                    # Head bbox
                    ax_bbox.add_patch(
                        patches.Rectangle(
                            (head_box[0], head_box[1]),
                            head_box[2] - head_box[0],
                            head_box[3] - head_box[1],
                            linewidth=2,
                            edgecolor=(1, 0, 0),
                            facecolor="none",
                        )
                    )

                    # Arrow from center head bbox to gaze_vector
                    gaze_vector_x, gaze_vector_y = gaze_vectors
                    final_gaze_x = (
                        head_box[0] + (head_box[2] - head_box[0]) / 2 + gaze_vector_x
                    )
                    final_gaze_y = (
                        head_box[1] + (head_box[3] - head_box[1]) / 2 + gaze_vector_y
                    )
                    # If the coordinates overflow the sample size, we clip them
                    final_gaze_x = min(final_gaze_x, sample.shape[2])
                    final_gaze_y = min(final_gaze_y, sample.shape[1])
                    final_gaze_x = max(final_gaze_x, 0)
                    final_gaze_y = max(final_gaze_y, 0)
                    # Get the vector back
                    gaze_vector_x = (
                        final_gaze_x
                        - (head_box[0] + (head_box[2] - head_box[0]) / 2)
                        - 10
                    )
                    gaze_vector_y = (
                        final_gaze_y
                        - (head_box[1] + (head_box[3] - head_box[1]) / 2)
                        - 10
                    )
                    ax_bbox.arrow(
                        head_box[0] + (head_box[2] - head_box[0]) / 2,
                        head_box[1] + (head_box[3] - head_box[1]) / 2,
                        gaze_vector_x,
                        gaze_vector_y,
                        color="r",
                        width=0.1,
                        head_width=10,
                        head_length=10,
                    )
                    ax_bbox.set_title("Predicted head bbox and gaze vector")

                    # Plot gaze cone
                    ax_cone.axis("off")
                    ax_cone.imshow(
                        unnorm(sample).permute(1, 2, 0).cpu().numpy(),
                        vmin=0,
                        vmax=1,
                    )
                    im = ax_cone.imshow(gaze_cone, cmap="jet", alpha=0.5)
                    divider = make_axes_locatable(ax_cone)
                    cax = divider.append_axes("right", size="3%", pad=0.05)
                    plt.colorbar(im, cax=cax)
                    ax_cone.set_title("Gaze cone")

                    # Plot gaze heatmap normalized
                    ax_heatmap.axis("off")
                    ax_heatmap.imshow(
                        unnorm(sample).permute(1, 2, 0).cpu().numpy(), vmin=0, vmax=1
                    )
                    im = ax_heatmap.imshow(
                        gaze_heatmap, cmap="jet", alpha=0.3, vmin=0, vmax=1
                    )
                    divider = make_axes_locatable(ax_heatmap)
                    cax = divider.append_axes("right", size="3%", pad=0.05)
                    plt.colorbar(im, cax=cax)
                    ax_heatmap.set_title("Gaze heatmap normalized [0, 1]")

                    # Plot heatmap unnormed
                    ax_heatmap_unscaled.axis("off")
                    ax_heatmap_unscaled.imshow(
                        unnorm(sample).permute(1, 2, 0).cpu().numpy(), vmin=0, vmax=1
                    )
                    im = ax_heatmap_unscaled.imshow(gaze_heatmap, cmap="jet", alpha=0.3)
                    divider = make_axes_locatable(ax_heatmap_unscaled)
                    cax = divider.append_axes("right", size="3%", pad=0.05)
                    plt.colorbar(im, cax=cax)
                    ax_heatmap_unscaled.set_title("Gaze heatmap unscaled")

                    plt.suptitle(f"Sample {idx} of {len(samples.tensors)}")
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    fig = plt.gcf()

                    final_image = fig2img(fig)

                    # Cleanup matplotlib
                    plt.close(fig)
                    plt.close("all")
                    del fig

                    wandb_gaze_heatmap_images.append(wandb.Image(final_image))

            # Log attention weights
            for idx in range(len(samples.tensors)):
                if (
                    outputs["sa_attn_weights"] is None
                    or outputs["ca_attn_weights"] is None
                    or outputs["objects_scores"] is None
                ):
                    break

                sa_attn_weights = outputs["sa_attn_weights"][idx].cpu()
                ca_attn_weights = outputs["ca_attn_weights"][idx].cpu()
                objects_scores = outputs["objects_scores"][idx].cpu()

                class_idxs = outputs["pred_logits"][idx].argmax(-1).cpu().numpy()

                # Use get_annotations to get the labels
                labels = [get_annotations()[class_idx] for class_idx in class_idxs]

                # Plot attention weights
                fig, (ax_sa, ax_ca, ax_objects) = plt.subplots(
                    1,
                    3,
                    figsize=((1536 * 3) / 96, 1536 / 96),
                    dpi=96,
                )

                ax_sa.set_title("Self-attention weights (last layer)")
                sns.heatmap(
                    sa_attn_weights,
                    ax=ax_sa,
                    cmap="RdYlGn",
                    linecolor="black",
                    linewidths=0.1,
                )
                # Set the labels on the y-axis and x-axis
                ax_sa.set_xticks(torch.arange(len(labels)) + 0.5)
                ax_sa.set_yticks(torch.arange(len(labels)) + 0.5)
                ax_sa.set_yticklabels(labels)
                ax_sa.set_xticklabels(labels)

                # Rotate the tick labels and set their alignment.
                plt.setp(
                    ax_sa.get_xticklabels(),
                    rotation=45,
                    ha="right",
                    rotation_mode="anchor",
                )

                # Rotate the tick labels and set their alignment.
                plt.setp(
                    ax_sa.get_yticklabels(),
                    rotation=45,
                    ha="right",
                    rotation_mode="anchor",
                )

                sns.heatmap(
                    ca_attn_weights,
                    ax=ax_ca,
                    cmap="RdYlGn",
                    # vmin=0,
                    # vmax=1,
                    linecolor="black",
                    linewidths=0.1,
                )
                ax_ca.set_title("Cross-attention weights (last layer)")
                # Set the labels on the y-axis and x-axis
                ax_ca.set_xticks(torch.arange(len(labels)) + 0.5)
                ax_ca.set_yticks(torch.arange(len(labels)) + 0.5)
                ax_ca.set_yticklabels(labels)
                ax_ca.set_xticklabels(labels)

                # Rotate the tick labels and set their alignment.
                plt.setp(
                    ax_ca.get_xticklabels(),
                    rotation=45,
                    ha="right",
                    rotation_mode="anchor",
                )

                plt.setp(
                    ax_ca.get_yticklabels(),
                    rotation=45,
                    ha="right",
                    rotation_mode="anchor",
                )

                sns.heatmap(
                    objects_scores,
                    ax=ax_objects,
                    cmap="RdYlGn",
                    vmin=0,
                    vmax=1,
                    linecolor="black",
                    linewidths=0.1,
                )
                ax_objects.set_title("Objects scores")
                # Set the labels on the y-axis and x-axis
                ax_objects.set_xticks(torch.arange(len(labels)) + 0.5)
                ax_objects.set_yticks(torch.arange(len(labels)) + 0.5)
                ax_objects.set_yticklabels(labels)
                ax_objects.set_xticklabels(labels)

                # Rotate the tick labels and set their alignment. Add a bit of top padding to the labels
                plt.setp(
                    ax_objects.get_xticklabels(),
                    rotation=45,
                    ha="right",
                    rotation_mode="anchor",
                )

                plt.setp(
                    ax_objects.get_yticklabels(),
                    rotation=45,
                    ha="right",
                    rotation_mode="anchor",
                )

                # Set plot title
                plt.suptitle(
                    f"Attention weights and objects scores for image {idx + 1} of {len(samples.tensors)}"
                )

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                fig = plt.gcf()

                final_image = fig2img(fig)

                # Cleanup matplotlib
                plt.close(fig)
                plt.close("all")
                del fig

                wandb_attention_images.append(wandb.Image(final_image))

        self.logger.experiment.log(
            {
                "val/images": wandb_od_images,
                "val/attention_images": wandb_attention_images,
                "val/gaze_heatmap_images": wandb_gaze_heatmap_images,
                "trainer/global_step": self.trainer.global_step,
            }
        )

    def on_validation_epoch_end(self):
        # Log each eval
        for eval_name, eval_value in self.evaluation.get_metrics().items():
            self.log(
                f"val/{eval_name}",
                eval_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()

    def configure_optimizers(self):
        params = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone.backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone.backbone" in n and p.requires_grad
                ],
                "lr": self.hparams.optimizer.keywords["lr"] / 10,
            },
        ]

        optimizer = self.hparams.optimizer(params=params)

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)

            return [optimizer], [scheduler]

        return optimizer


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "gotd.yaml")
    _ = hydra.utils.instantiate(cfg)

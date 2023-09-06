"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references and DETR's official implementation.
"""
import numpy as np
import torch
from PIL import Image
from matplotlib.pyplot import cm

from src import utils
from .NestedTensor import NestedTensor

log = utils.get_pylogger(__name__)


def collate_fn(batch):
    batch = list(zip(*batch))
    # img
    batch[0] = NestedTensor.nested_tensor_from_tensor_list(batch[0])
    # depth
    batch[1] = NestedTensor.nested_tensor_from_tensor_list(batch[1])
    # sample size
    batch[2] = torch.stack(batch[2])

    return tuple(batch)


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != "numpy":
        raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray


def unnorm(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    std = torch.tensor(std, device=img.device).reshape(3, 1, 1)
    mean = torch.tensor(mean, device=img.device).reshape(3, 1, 1)
    return img * std + mean


def load_pretrained(model, checkpoint, drop_prefix=None):
    model_dict = model.state_dict()
    model_weights = (
        checkpoint["model"] if "model" in checkpoint else checkpoint["state_dict"]
    )

    # Check if shapes between model_dict and checkpoint match, otherwise add to new_state_dict
    new_state_dict = {}
    for k, v in model_weights.items():
        # Remove prefix if needed
        if drop_prefix is not None and k.startswith(drop_prefix):
            k = k[len(drop_prefix) :]

        if k in model_dict and model_dict[k].shape == v.shape:
            new_state_dict[k] = v
        elif k in model_dict and model_dict[k].shape != v.shape:
            log.warning(
                f"Skipping {k} from pretrained weights: shape mismatch ({v.shape} vs {model_dict[k].shape})"
            )
        else:
            log.warning(f"Skipping {k} from pretrained weights: not found in model")

    log.info(f"Total weights from file: {len(model_weights)}")
    log.info(f"Total weights loaded: {len(new_state_dict)}")

    model_dict.update(new_state_dict)
    log.info(model.load_state_dict(model_dict, strict=False))


# https://stackoverflow.com/questions/4971269/how-to-pick-a-new-color-for-each-plotted-line-within-a-figure-in-matplotlib
def get_bbox_colors(num_classes=81):
    color = iter(cm.prism(np.linspace(0, 1, num_classes)))

    return [next(color)[:3] for _ in range(num_classes)]


def get_annotation_id(annotation_name):
    return list(get_annotations().keys())[
        list(get_annotations().values()).index(annotation_name)
    ]


def get_annotations():
    return {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "airplane",
        5: "bus",
        6: "train",
        7: "truck",
        8: "boat",
        9: "traffic light",
        10: "fire hydrant",
        11: "stop sign",
        12: "parking meter",
        13: "bench",
        14: "bird",
        15: "cat",
        16: "dog",
        17: "horse",
        18: "sheep",
        19: "cow",
        20: "elephant",
        21: "bear",
        22: "zebra",
        23: "giraffe",
        24: "backpack",
        25: "umbrella",
        26: "handbag",
        27: "tie",
        28: "suitcase",
        29: "frisbee",
        30: "skis",
        31: "snowboard",
        32: "sports ball",
        33: "kite",
        34: "baseball bat",
        35: "baseball glove",
        36: "skateboard",
        37: "surfboard",
        38: "tennis racket",
        39: "bottle",
        40: "wine glass",
        41: "cup",
        42: "fork",
        43: "knife",
        44: "spoon",
        45: "bowl",
        46: "banana",
        47: "apple",
        48: "sandwich",
        49: "orange",
        50: "broccoli",
        51: "carrot",
        52: "hot dog",
        53: "pizza",
        54: "donut",
        55: "cake",
        56: "chair",
        57: "couch",
        58: "potted plant",
        59: "bed",
        60: "dining table",
        61: "toilet",
        62: "tv",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
        67: "cell phone",
        68: "microwave",
        69: "oven",
        70: "toaster",
        71: "sink",
        72: "refrigerator",
        73: "book",
        74: "clock",
        75: "vase",
        76: "scissors",
        77: "teddy bear",
        78: "hair drier",
        79: "toothbrush",
        80: "face",  # <-- max_object_id
        81: "no-object",
    }


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

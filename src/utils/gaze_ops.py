import math
import numpy as np
import torch

from src.utils.misc import to_numpy, to_torch


def get_label_map(img, pt, sigma, pdf="Gaussian"):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    img = to_numpy(img)

    # Check that any part of the gaussian is in-bounds
    ul = [
        pt[0].round().int().item() - 3 * sigma,
        pt[1].round().int().item() - 3 * sigma,
    ]
    br = [
        pt[0].round().int().item() + 3 * sigma + 1,
        pt[1].round().int().item() + 3 * sigma + 1,
    ]
    if ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0:
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if pdf == "Gaussian":
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
    elif pdf == "Cauchy":
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma**2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0] : img_y[1], img_x[0] : img_x[1]] += g[g_y[0] : g_y[1], g_x[0] : g_x[1]]
    img = img / np.max(img)  # normalize heatmap so it has max value of 1

    return to_torch(img)


def get_multi_hot_map(gaze_pts, out_res, device=torch.device("cuda")):
    h, w = out_res
    target_map = torch.zeros((h, w), device=device).long()
    for p in gaze_pts:
        if p[0] >= 0:
            x, y = map(int, [p[0] * float(w), p[1] * float(h)])
            x = min(x, w - 1)
            y = min(y, h - 1)
            target_map[y, x] = 1

    return target_map


def get_heatmap_peak_coords(heatmap):
    np_heatmap = to_numpy(heatmap)
    idx = np.unravel_index(np_heatmap.argmax(), np_heatmap.shape)
    pred_y, pred_x = map(float, idx)

    return pred_x, pred_y


def get_l2_dist(p1, p2):
    return torch.sqrt((p1[:, 0] - p2[:, 0]) ** 2 + (p1[:, 1] - p2[:, 1]) ** 2)


def rescale_heatmap(heatmap):
    heatmap -= heatmap.min(1, keepdim=True)[0]
    heatmap /= heatmap.max(1, keepdim=True)[0]

    return heatmap


def get_gaze_cone(
    head_center_point,
    gaze_vector,
    out_size=(64, 64),
    cone_angle=120,
):
    return get_continuous_gaze_cone(
        head_center_point, gaze_vector, out_size, cone_angle
    )


def get_continuous_gaze_cone(
    head_center_point,
    gaze_vector,
    out_size=(64, 64),
    cone_angle=120,
):
    coords_dim = head_center_point.shape[1]
    if coords_dim == 3:
        width, height, depth = out_size

        eye_coords = (
            (
                head_center_point
                * torch.tensor([width, height, depth], device=head_center_point.device)
            )
            .unsqueeze(1)
            .unsqueeze(1)
            .unsqueeze(1)
        )
        gaze_coords = (
            (
                (head_center_point + gaze_vector)
                * torch.tensor([width, height, depth], device=head_center_point.device)
            )
            .unsqueeze(1)
            .unsqueeze(1)
            .unsqueeze(1)
        )

        pixel_mat = (
            torch.stack(
                torch.meshgrid(
                    [
                        torch.arange(1, width + 1),
                        torch.arange(1, height + 1),
                        torch.arange(1, depth + 1),
                    ]
                ),
                dim=-1,
            )
            .unsqueeze(0)
            .repeat(head_center_point.shape[0], 1, 1, 1, 1)
            .to(head_center_point.device)
        )
    else:
        width, height = out_size

        eye_coords = (
            (
                head_center_point
                * torch.tensor([width, height], device=head_center_point.device)
            )
            .unsqueeze(1)
            .unsqueeze(1)
        )
        gaze_coords = (
            (
                (head_center_point + gaze_vector)
                * torch.tensor([width, height], device=head_center_point.device)
            )
            .unsqueeze(1)
            .unsqueeze(1)
        )

        pixel_mat = (
            torch.stack(
                torch.meshgrid(
                    [torch.arange(1, width + 1), torch.arange(1, height + 1)]
                ),
                dim=-1,
            )
            .unsqueeze(0)
            .repeat(head_center_point.shape[0], 1, 1, 1)
            .to(head_center_point.device)
        )

    dot_prod = torch.sum((pixel_mat - eye_coords) * (gaze_coords - eye_coords), dim=-1)
    gaze_vector_norm = torch.sqrt(torch.sum((gaze_coords - eye_coords) ** 2, dim=-1))
    pixel_mat_norm = torch.sqrt(torch.sum((pixel_mat - eye_coords) ** 2, dim=-1))

    theta = cone_angle * (torch.pi / 180)
    beta = torch.acos(dot_prod / (gaze_vector_norm * pixel_mat_norm))

    # Create mask where true if beta is less than theta/2
    pixel_mat_presence = beta < (theta / 2)

    gaze_cones = dot_prod / (gaze_vector_norm * pixel_mat_norm)

    # Zero out values outside the gaze cone
    gaze_cones[~pixel_mat_presence] = 0
    gaze_cones = torch.clamp(gaze_cones, 0, None)

    return gaze_cones


def get_angle_magnitude(p1, p2, dimension=2):
    # Add first dimension if it doesn't exist
    if len(p1.shape) == 1:
        p1 = p1.unsqueeze(0)
    if len(p2.shape) == 1:
        p2 = p2.unsqueeze(0)

    vx, vy = p2[:, 0] - p1[:, 0], p2[:, 1] - p1[:, 1]

    if dimension == 3:
        vz = p2[:, 2] - p1[:, 2]

        magnitude = torch.sqrt(vx**2 + vy**2 + vz**2)
        # phi range is [-pi, pi]
        phi = torch.atan2(vy, vx)
        # theta range is [-pi/2, pi/2]
        theta = torch.arcsin(vz / magnitude)

        # Scale magnitude to be between 0 and 1
        magnitude = magnitude / math.sqrt(3)

        # Scale phi to be between 0 and 1
        phi = (phi + torch.pi) / (2 * torch.pi)

        # Scale theta to be between 0 and 1
        theta = (theta + (torch.pi / 2)) / torch.pi

        return torch.cat(
            [phi.unsqueeze(1), theta.unsqueeze(1), magnitude.unsqueeze(1)], dim=1
        )
    else:
        magnitude = torch.sqrt(vx**2 + vy**2)
        # phi range is [-pi, pi]
        phi = torch.atan2(vy, vx)

        # Scale magnitude to be between 0 and 1
        magnitude = magnitude / math.sqrt(2)

        # Scale phi to be between 0 and 1
        phi = (phi + torch.pi) / (2 * torch.pi)

        return torch.cat([phi.unsqueeze(1), magnitude.unsqueeze(1)], dim=1)

"""Common preprocessing and photon-parameter helpers."""

import numpy as np
from skimage.transform import resize


# =========================================================
# Common processing
# =========================================================
def resize_rgb_depth(rgb, depth, res):
    rgb_resized = resize(rgb, (res, res, 3), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
    depth_resized = resize(depth, (res, res), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
    return rgb_resized, depth_resized


def rgb_to_albedo_and_intensity(rgb):
    rgb01 = np.clip(rgb / 255.0, 0.0, 1.0).astype(np.float32)
    intensity = (
        0.2989 * rgb01[..., 0]
        + 0.5870 * rgb01[..., 1]
        + 0.1140 * rgb01[..., 2]
    ).astype(np.float32)
    albedo = rgb01[..., 2].astype(np.float32)
    return np.maximum(albedo, 0.0), np.maximum(intensity, 0.0)


def get_simulation_parameters(param_idx: int):
    simulation_params = np.array(
        [
            [10, 2],
            [5, 2],
            [2, 2],
            [10, 10],
            [5, 10],
            [2, 10],
            [10, 50],
            [5, 50],
            [2, 50],
        ],
        dtype=np.float32,
    )

    if 1 <= param_idx <= 9:
        mean_signal_photons = float(simulation_params[param_idx - 1, 0])
        mean_background_photons = float(simulation_params[param_idx - 1, 1])
        sbr = mean_signal_photons / mean_background_photons
    elif param_idx == 10:
        mean_signal_photons = float(np.random.choice([2, 5, 10, 20]))
        sbr = float(np.random.choice([0.03, 0.04, 0.1, 0.2]))
        mean_background_photons = mean_signal_photons / sbr
    else:
        raise ValueError("param_idx must be in [1, 10].")

    return mean_signal_photons, mean_background_photons, sbr

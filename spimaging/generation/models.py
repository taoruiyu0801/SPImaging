"""SPAD measurement models."""

import numpy as np


# =========================================================
# Original single-surface model
# =========================================================
def build_deepinverse_signal(depth, albedo, intensity, bins, bin_size, mean_signal_photons, sbr):
    c = 3e8

    depth = np.maximum(depth.astype(np.float32), 1e-6)
    albedo = np.maximum(albedo.astype(np.float32), 0.0)
    intensity = np.maximum(intensity.astype(np.float32), 0.0)

    tof = 2.0 * depth / c
    depth_bins = tof / bin_size
    depth_bins = np.clip(depth_bins, 1.0, float(bins)).astype(np.float32)

    signal = albedo / np.maximum(depth**2, 1e-6)
    signal_mean = float(signal.mean()) if float(signal.mean()) > 0 else 1.0
    signal = signal / signal_mean * mean_signal_photons

    background = intensity.copy()
    background_mean = float(background.mean()) if float(background.mean()) > 0 else 1.0
    background = background / background_mean * (mean_signal_photons / sbr / bins)

    x = np.stack([depth_bins, signal, background], axis=0).astype(np.float32)
    return x


def simulate_with_deepinverse(x_np, bins, irf_sigma, device=None):
    import torch

    from spimaging.generation.deepinverse import import_deepinv
    from spimaging.training_common.device import get_torch_device

    dinv = import_deepinv()
    if device is None:
        device = get_torch_device()
    physics = dinv.physics.SinglePhotonLidar(bins=bins, sigma=irf_sigma, device=device)

    x = torch.from_numpy(x_np).unsqueeze(0).to(device)
    y = physics(x)
    xhat = physics.A_dagger(y)

    y_np = y.squeeze(0).detach().cpu().numpy().astype(np.float32)
    xhat_np = xhat.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return y_np, xhat_np


# =========================================================
# Shared helpers for multi-surface models
# =========================================================
def gaussian_kernel_2d(kernel_size=5, sigma=1.0):
    if kernel_size % 2 == 0:
        raise ValueError("mix_kernel_size must be odd.")
    ax = np.arange(kernel_size, dtype=np.float32) - kernel_size // 2
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    k = k / np.sum(k)
    return k.astype(np.float32)


def gaussian_pulse_1d(sigma_bins=2.0, radius=None):
    if radius is None:
        radius = max(3, int(np.ceil(4 * sigma_bins)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    g = np.exp(-(x**2) / (2 * sigma_bins**2))
    g = g / np.sum(g)
    return g.astype(np.float32)


def shift_add_pulse(signal_vec, center_bin_float, amplitude, pulse):
    T = signal_vec.shape[0]
    p_len = pulse.shape[0]
    radius = p_len // 2
    center_int = int(np.round(center_bin_float))
    for i in range(p_len):
        t = center_int + (i - radius)
        if 0 <= t < T:
            signal_vec[t] += amplitude * pulse[i]


def compute_base_signal_and_background(depth, albedo, intensity, bins, mean_signal_photons, sbr):
    signal_base = albedo / np.maximum(depth**2, 1e-6)
    signal_mean = float(signal_base.mean()) if float(signal_base.mean()) > 0 else 1.0
    signal_base = signal_base / signal_mean * mean_signal_photons

    background = intensity.copy()
    background_mean = float(background.mean()) if float(background.mean()) > 0 else 1.0
    background = background / background_mean * (mean_signal_photons / sbr / bins)

    return signal_base.astype(np.float32), background.astype(np.float32)


# =========================================================
# Neighborhood-mixing multi-surface model
# =========================================================
def build_neighborhood_mixed_measurement(
    depth, albedo, intensity, bins, bin_size, mean_signal_photons, sbr,
    mix_kernel_size=5, mix_sigma_xy=1.0, mix_time_sigma_bins=2.0,
):
    c = 3e8
    H, W = depth.shape

    depth = np.maximum(depth.astype(np.float32), 1e-6)
    albedo = np.maximum(albedo.astype(np.float32), 0.0)
    intensity = np.maximum(intensity.astype(np.float32), 0.0)

    spatial_kernel = gaussian_kernel_2d(mix_kernel_size, mix_sigma_xy)
    pulse = gaussian_pulse_1d(mix_time_sigma_bins)

    signal_base, background = compute_base_signal_and_background(depth, albedo, intensity, bins, mean_signal_photons, sbr)

    transient_clean = np.zeros((bins, H, W), dtype=np.float32)
    mixed_depth_bin = np.zeros((H, W), dtype=np.float32)
    mixed_signal = np.zeros((H, W), dtype=np.float32)

    r = mix_kernel_size // 2

    for i in range(H):
        for j in range(W):
            local_hist = np.zeros((bins,), dtype=np.float32)
            depth_acc = 0.0
            weight_acc = 0.0
            signal_acc = 0.0

            for du in range(-r, r + 1):
                for dv in range(-r, r + 1):
                    ii = i + du
                    jj = j + dv
                    if ii < 0 or ii >= H or jj < 0 or jj >= W:
                        continue

                    w = spatial_kernel[du + r, dv + r]
                    tof = 2.0 * depth[ii, jj] / c
                    depth_bin = np.clip(tof / bin_size, 1.0, float(bins - 1))
                    amp = w * signal_base[ii, jj]

                    shift_add_pulse(local_hist, depth_bin, amp, pulse)
                    depth_acc += w * depth_bin
                    weight_acc += w
                    signal_acc += amp

            local_hist += background[i, j]
            transient_clean[:, i, j] = local_hist
            mixed_depth_bin[i, j] = depth_acc / max(weight_acc, 1e-8)
            mixed_signal[i, j] = signal_acc

    counts = np.random.poisson(np.maximum(transient_clean, 0.0)).astype(np.float32)
    peak_bin = np.argmax(counts, axis=0).astype(np.float32)
    peak_signal = np.max(counts, axis=0).astype(np.float32)
    xhat = np.stack([peak_bin, peak_signal, background], axis=0).astype(np.float32)
    x_proxy = np.stack([mixed_depth_bin, mixed_signal, background], axis=0).astype(np.float32)

    return counts, xhat, x_proxy, transient_clean


# =========================================================
# Translucent-layer model
# =========================================================
def build_translucent_front_depth(H, W, front_type="flat", base_depth=1.0, x_slope=0.0, y_slope=0.0, amplitude=0.1):
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, H, dtype=np.float32),
        np.linspace(-1.0, 1.0, W, dtype=np.float32),
        indexing="ij",
    )

    if front_type == "flat":
        front = np.full((H, W), base_depth, dtype=np.float32)
    elif front_type == "sloped":
        front = base_depth + x_slope * xx + y_slope * yy
    elif front_type == "sinusoidal":
        front = base_depth + amplitude * np.sin(np.pi * xx) * np.cos(np.pi * yy)
    else:
        raise ValueError(f"Unknown translucent_front_type: {front_type}")

    return np.maximum(front, 1e-3).astype(np.float32)


def build_translucent_layer_measurement(
    depth, albedo, intensity, bins, bin_size, mean_signal_photons, sbr,
    front_type="flat", front_depth=1.0, front_depth_x_slope=0.0, front_depth_y_slope=0.0,
    front_depth_amplitude=0.1, front_signal_ratio=0.25, transmission=0.6, time_sigma_bins=2.0,
):
    c = 3e8
    H, W = depth.shape

    depth = np.maximum(depth.astype(np.float32), 1e-6)
    albedo = np.maximum(albedo.astype(np.float32), 0.0)
    intensity = np.maximum(intensity.astype(np.float32), 0.0)

    pulse = gaussian_pulse_1d(time_sigma_bins)
    signal_base, background = compute_base_signal_and_background(depth, albedo, intensity, bins, mean_signal_photons, sbr)

    front_depth_map = build_translucent_front_depth(
        H=H,
        W=W,
        front_type=front_type,
        base_depth=front_depth,
        x_slope=front_depth_x_slope,
        y_slope=front_depth_y_slope,
        amplitude=front_depth_amplitude,
    )
    front_depth_map = np.minimum(front_depth_map, depth - 1e-3)
    front_depth_map = np.maximum(front_depth_map, 1e-3)

    front_tof_bin = np.clip((2.0 * front_depth_map / c) / bin_size, 1.0, float(bins - 1)).astype(np.float32)
    back_tof_bin = np.clip((2.0 * depth / c) / bin_size, 1.0, float(bins - 1)).astype(np.float32)

    front_signal = front_signal_ratio * mean_signal_photons * np.ones((H, W), dtype=np.float32)
    back_signal = transmission * signal_base

    transient_clean = np.zeros((bins, H, W), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            local_hist = np.zeros((bins,), dtype=np.float32)
            shift_add_pulse(local_hist, front_tof_bin[i, j], front_signal[i, j], pulse)
            shift_add_pulse(local_hist, back_tof_bin[i, j], back_signal[i, j], pulse)
            local_hist += background[i, j]
            transient_clean[:, i, j] = local_hist

    counts = np.random.poisson(np.maximum(transient_clean, 0.0)).astype(np.float32)
    peak_bin = np.argmax(counts, axis=0).astype(np.float32)
    peak_signal = np.max(counts, axis=0).astype(np.float32)
    xhat = np.stack([peak_bin, peak_signal, background], axis=0).astype(np.float32)

    depth_proxy = (front_signal * front_tof_bin + back_signal * back_tof_bin) / np.maximum(front_signal + back_signal, 1e-8)
    total_signal_proxy = front_signal + back_signal
    x_proxy = np.stack([depth_proxy, total_signal_proxy, background], axis=0).astype(np.float32)

    extra = {
        "front_depth_m": front_depth_map.astype(np.float32),
        "front_signal": front_signal.astype(np.float32),
        "back_signal_after_transmission": back_signal.astype(np.float32),
        "front_tof_bin": front_tof_bin.astype(np.float32),
        "back_tof_bin": back_tof_bin.astype(np.float32),
    }

    return counts, xhat, x_proxy, transient_clean, extra


# =========================================================
# Volume-scattering model
# =========================================================
def get_volume_medium_defaults(medium_type):
    if medium_type == "fog":
        return {
            "extinction_coeff": 0.35,
            "backscatter_ratio": 0.20,
            "front_boost": 1.0,
        }
    elif medium_type == "water":
        return {
            "extinction_coeff": 0.8,
            "backscatter_ratio": 0.35,
            "front_boost": 1.5,
        }
    else:
        raise ValueError(f"Unknown volume_medium_type: {medium_type}")


def build_volume_scattering_measurement(
    depth, albedo, intensity, bins, bin_size, mean_signal_photons, sbr,
    medium_type="fog",
    extinction_coeff=None,
    backscatter_ratio=None,
    scatter_depth_fraction=0.9,
    num_steps=64,
    time_sigma_bins=2.0,
    range_weight_power=1.0,
    water_front_boost=1.5,
    fog_front_boost=1.0,
):
    """
    Model:
    - direct surface return attenuated by Beer-Lambert transmission
    - integrated path backscatter accumulated from sensor to a fraction of target depth
    """
    c = 3e8
    H, W = depth.shape

    depth = np.maximum(depth.astype(np.float32), 1e-6)
    albedo = np.maximum(albedo.astype(np.float32), 0.0)
    intensity = np.maximum(intensity.astype(np.float32), 0.0)

    defaults = get_volume_medium_defaults(medium_type)
    if extinction_coeff is None:
        extinction_coeff = defaults["extinction_coeff"]
    if backscatter_ratio is None:
        backscatter_ratio = defaults["backscatter_ratio"]

    front_boost = water_front_boost if medium_type == "water" else fog_front_boost

    pulse = gaussian_pulse_1d(time_sigma_bins)
    signal_base, background = compute_base_signal_and_background(depth, albedo, intensity, bins, mean_signal_photons, sbr)

    transient_clean = np.zeros((bins, H, W), dtype=np.float32)
    scatter_integral_signal = np.zeros((H, W), dtype=np.float32)
    surface_signal_after_medium = np.zeros((H, W), dtype=np.float32)
    surface_tof_bin = np.zeros((H, W), dtype=np.float32)
    volume_depth_limit_m = np.zeros((H, W), dtype=np.float32)
    volume_scatter_tof_map = np.zeros((H, W), dtype=np.float32)

    eps = 1e-8

    for i in range(H):
        for j in range(W):
            z = depth[i, j]
            local_hist = np.zeros((bins,), dtype=np.float32)

            # direct surface return after medium attenuation
            direct_transmission = np.exp(-2.0 * extinction_coeff * z)
            surface_amp = signal_base[i, j] * direct_transmission
            surface_bin = np.clip((2.0 * z / c) / bin_size, 1.0, float(bins - 1))
            shift_add_pulse(local_hist, surface_bin, surface_amp, pulse)

            # path backscatter
            z_max = scatter_depth_fraction * z
            z_max = max(z_max, 1e-3)
            z_samples = np.linspace(1e-3, z_max, num_steps, dtype=np.float32)

            # nominal total backscatter amount, relative to mean-signal level
            total_scatter_amp = backscatter_ratio * mean_signal_photons

            # depth-dependent weighting
            z_norm = z_samples / max(z_max, eps)
            weights = z_norm ** range_weight_power

            # front-boost for water / fog styling
            # smaller z => stronger near-camera scatter when front_boost > 1
            near_boost = 1.0 + (front_boost - 1.0) * (1.0 - z_norm)

            # two-way attenuation for scattered photons from depth z_s back to detector
            atten = np.exp(-2.0 * extinction_coeff * z_samples)

            scatter_profile = weights * near_boost * atten
            if np.sum(scatter_profile) > 0:
                scatter_profile = scatter_profile / np.sum(scatter_profile)
            scatter_profile = scatter_profile * total_scatter_amp

            for k, z_s in enumerate(z_samples):
                scatter_bin = np.clip((2.0 * z_s / c) / bin_size, 1.0, float(bins - 1))
                shift_add_pulse(local_hist, scatter_bin, scatter_profile[k], pulse)

            local_hist += background[i, j]
            transient_clean[:, i, j] = local_hist

            scatter_integral_signal[i, j] = float(np.sum(scatter_profile))
            surface_signal_after_medium[i, j] = surface_amp
            surface_tof_bin[i, j] = surface_bin
            volume_depth_limit_m[i, j] = z_max

            if np.sum(scatter_profile) > 0:
                weighted_scatter_depth = np.sum(z_samples * scatter_profile) / np.sum(scatter_profile)
            else:
                weighted_scatter_depth = 0.0
            volume_scatter_tof_map[i, j] = np.clip((2.0 * weighted_scatter_depth / c) / bin_size, 1.0, float(bins - 1))

    counts = np.random.poisson(np.maximum(transient_clean, 0.0)).astype(np.float32)

    peak_bin = np.argmax(counts, axis=0).astype(np.float32)
    peak_signal = np.max(counts, axis=0).astype(np.float32)
    xhat = np.stack([peak_bin, peak_signal, background], axis=0).astype(np.float32)

    total_signal_proxy = scatter_integral_signal + surface_signal_after_medium
    depth_proxy = (
        scatter_integral_signal * volume_scatter_tof_map + surface_signal_after_medium * surface_tof_bin
    ) / np.maximum(total_signal_proxy, 1e-8)

    x_proxy = np.stack([depth_proxy, total_signal_proxy, background], axis=0).astype(np.float32)

    extra = {
        "volume_depth_limit_m": volume_depth_limit_m.astype(np.float32),
        "volume_scatter_signal": scatter_integral_signal.astype(np.float32),
        "surface_signal_after_medium": surface_signal_after_medium.astype(np.float32),
        "volume_scatter_tof_map": volume_scatter_tof_map.astype(np.float32),
        "surface_tof_bin": surface_tof_bin.astype(np.float32),
    }

    return counts, xhat, x_proxy, transient_clean, extra

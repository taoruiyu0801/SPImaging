import argparse
from pathlib import Path
import pickle
import random
import textwrap
import zipfile

import numpy as np

from spimaging.cli import (
    ArgumentParser,
    HelpFormatter,
    create_output_parent,
    nonnegative_int,
    positive_int,
    require_directory,
    validate_npz_archive,
    validate_output_directory,
    validate_output_file,
)

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


def build_parser():
    parser = ArgumentParser(
        prog="spad-browse",
        description="Interactively browse generated SPAD samples with OpenCV.",
        formatter_class=HelpFormatter,
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory containing generated .npz samples.",
    )
    parser.add_argument(
        "--start_index",
        type=nonnegative_int,
        default=0,
        metavar="INDEX",
        help="Zero-based initial sample index (must exist).",
    )
    parser.add_argument(
        "--random_start",
        action="store_true",
        help="Start from a random sample instead of --start_index.",
    )
    parser.add_argument(
        "--window_name",
        type=str,
        default="SPAD Dataset Browser",
        help="OpenCV window name.",
    )
    parser.add_argument(
        "--cell_size",
        type=positive_int,
        default=320,
        metavar="PIXELS",
        help="Width and height in pixels of each image panel.",
    )
    parser.add_argument(
        "--output_dir",
        "--save_dir",
        dest="output_dir",
        type=str,
        default=None,
        help=(
            "Directory for canvases saved with the G key; by default uses "
            "<dataset_dir>/browse_exports. --save_dir is a compatibility alias."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow a previously saved canvas file to be replaced.",
    )
    parser.add_argument(
        "--browse_mode",
        type=str,
        default="auto",
        choices=["auto", "single", "neighborhood_mix", "translucent_layer", "volume_scattering"],
        help="Display mode; auto reads surface_model from each sample when available.",
    )
    parser.add_argument(
        "--pixel_source",
        type=str,
        default="auto",
        choices=["auto", "counts", "transient_clean"],
        help="Source used for the three per-pixel histograms.",
    )
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def list_npz_files(dataset_dir: Path):
    files = sorted(dataset_dir.glob("sample_*.npz"))
    if len(files) == 0:
        files = sorted(dataset_dir.glob("*.npz"))
    return files


def normalize_to_uint8(arr):
    arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    vmin = arr.min()
    vmax = arr.max()
    if vmax <= vmin:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = (arr - vmin) / (vmax - vmin)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def require_cv2():
    if cv2 is None:
        raise ImportError("OpenCV is required for the dataset browser. Install it with: pip install opencv-python")


def colorize_gray(arr, cmap=None):
    require_cv2()
    if cmap is None:
        cmap = cv2.COLORMAP_VIRIDIS
    gray = normalize_to_uint8(arr)
    return cv2.applyColorMap(gray, cmap)


def rgb_to_bgr_display(rgb):
    require_cv2()
    rgb = rgb.astype(np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = (rgb * 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def prepare_xhat_views(xhat):
    views = {}
    if xhat.ndim == 3:
        c, _, _ = xhat.shape
        if c == 3:
            views["xhat channel 0"] = ("viridis", xhat[0])
            views["xhat channel 1"] = ("magma", xhat[1])
            views["xhat channel 2"] = ("magma", xhat[2])
        else:
            views["xhat sum over time"] = ("magma", xhat.sum(axis=0))
            views["xhat peak time-bin index"] = ("viridis", np.argmax(xhat, axis=0).astype(np.float32))
    elif xhat.ndim == 2:
        views["xhat 2D"] = ("viridis", xhat)
    else:
        flat = np.squeeze(xhat)
        if flat.ndim == 2:
            views["xhat squeezed"] = ("viridis", flat)
    return views


def cmap_name_to_cv2(cmap_name):
    if cmap_name == "gray":
        return None
    if cmap_name == "magma":
        return cv2.COLORMAP_MAGMA
    return cv2.COLORMAP_VIRIDIS


def render_panel(title, image_bgr, cell_size=320, title_height=32):
    canvas = np.full((cell_size + title_height, cell_size, 3), 245, dtype=np.uint8)
    resized = cv2.resize(image_bgr, (cell_size, cell_size), interpolation=cv2.INTER_NEAREST)
    canvas[title_height:title_height + cell_size] = resized

    cv2.rectangle(canvas, (0, 0), (cell_size - 1, title_height - 1), (230, 230, 230), -1)
    cv2.putText(canvas, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (cell_size - 1, cell_size + title_height - 1), (180, 180, 180), 1)
    return canvas


def render_text_panel(lines, width, height, title="Metadata"):
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (width - 1, 34), (230, 230, 230), -1)
    cv2.putText(canvas, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 1, cv2.LINE_AA)

    y = 55
    for line in lines:
        wrapped = textwrap.wrap(str(line), width=54) if len(str(line)) > 54 else [str(line)]
        for w in wrapped:
            if y > height - 10:
                break
            cv2.putText(canvas, w, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
            y += 22
        if y > height - 10:
            break

    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (180, 180, 180), 1)
    return canvas


def render_histogram_panel(hist, width, height, title="Global photon histogram", subtitle=""):
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (width - 1, 34), (230, 230, 230), -1)
    cv2.putText(canvas, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle, (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (60, 60, 60), 1, cv2.LINE_AA)

    if hist is None or len(hist) == 0:
        cv2.putText(canvas, "No histogram available", (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 1)
        cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (180, 180, 180), 1)
        return canvas

    hist = np.asarray(hist).astype(np.float32)
    hist = np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)

    left = 50
    top = 70 if subtitle else 50
    right = width - 20
    bottom = height - 35

    cv2.rectangle(canvas, (left, top), (right, bottom), (240, 240, 240), -1)
    cv2.rectangle(canvas, (left, top), (right, bottom), (180, 180, 180), 1)

    hnorm = hist / hist.max() if hist.max() > 0 else hist
    n = len(hist)
    pts = []
    for i, v in enumerate(hnorm):
        x = int(left + i * (right - left) / max(n - 1, 1))
        y = int(bottom - v * (bottom - top))
        pts.append((x, y))

    for i in range(1, len(pts)):
        cv2.line(canvas, pts[i - 1], pts[i], (40, 90, 220), 1, cv2.LINE_AA)

    cv2.putText(canvas, "time-bin", (width // 2 - 30, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
    cv2.putText(canvas, "counts", (8, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (180, 180, 180), 1)
    return canvas


def choose_three_pixels(h, w):
    return [(h // 4, w // 4), (h // 2, w // 2), (3 * h // 4, 3 * w // 4)]


def draw_points_on_map(image_bgr, points, original_hw):
    out = image_bgr.copy()
    H, W = original_hw
    h2, w2 = out.shape[:2]
    colors = [(0, 0, 255), (0, 180, 0), (255, 0, 0)]
    labels = ["P1", "P2", "P3"]
    for i, (r, c) in enumerate(points):
        x = int(c * w2 / max(W, 1))
        y = int(r * h2 / max(H, 1))
        cv2.circle(out, (x, y), 6, colors[i], -1, cv2.LINE_AA)
        cv2.putText(out, labels[i], (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[i], 2, cv2.LINE_AA)
    return out


def render_pixel_histogram_panel(hist, width, height, title="Pixel histogram", color=(40, 90, 220), subtitle=""):
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (width - 1, 34), (230, 230, 230), -1)
    cv2.putText(canvas, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle, (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (60, 60, 60), 1, cv2.LINE_AA)

    if hist is None or len(hist) == 0:
        cv2.putText(canvas, "No counts available", (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 1)
        cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (180, 180, 180), 1)
        return canvas

    hist = np.asarray(hist).astype(np.float32)
    hist = np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)

    left, top, right, bottom = 50, 70, width - 20, height - 35
    cv2.rectangle(canvas, (left, top), (right, bottom), (240, 240, 240), -1)
    cv2.rectangle(canvas, (left, top), (right, bottom), (180, 180, 180), 1)

    hnorm = hist / hist.max() if hist.max() > 0 else hist
    n = len(hist)
    pts = []
    for i, v in enumerate(hnorm):
        x = int(left + i * (right - left) / max(n - 1, 1))
        y = int(bottom - v * (bottom - top))
        pts.append((x, y))
    for i in range(1, len(pts)):
        cv2.line(canvas, pts[i - 1], pts[i], color, 1, cv2.LINE_AA)

    cv2.putText(canvas, "time-bin", (width // 2 - 30, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
    cv2.putText(canvas, "counts", (8, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (180, 180, 180), 1)
    return canvas


def resolve_browse_mode(data, browse_mode):
    if browse_mode != "auto":
        return browse_mode
    if "surface_model" in data:
        val = data["surface_model"]
        if isinstance(val, np.ndarray) and val.ndim == 0:
            val = val.item()
        return str(val)
    return "single"


def build_hist_source(data, pixel_source, resolved_mode):
    if pixel_source == "counts":
        if "counts" in data and data["counts"].ndim == 3:
            return data["counts"], "counts"
        return None, "none"

    if pixel_source == "transient_clean":
        if "transient_clean" in data and data["transient_clean"].ndim == 3:
            return data["transient_clean"], "transient_clean"
        return None, "none"

    # auto
    if resolved_mode in ["neighborhood_mix", "translucent_layer", "volume_scattering"]:
        if "transient_clean" in data and data["transient_clean"].ndim == 3:
            return data["transient_clean"], "transient_clean"
    if "counts" in data and data["counts"].ndim == 3:
        return data["counts"], "counts"
    return None, "none"


def build_common_items(data, pixel_points):
    items = []
    count_hist = None

    if "rgb" in data:
        items.append(("RGB image", rgb_to_bgr_display(data["rgb"])))
    if "depth_m" in data:
        items.append(("Depth map (meters)", colorize_gray(data["depth_m"], cv2.COLORMAP_VIRIDIS)))
    if "albedo" in data:
        items.append(("Albedo surrogate", colorize_gray(data["albedo"], cv2.COLORMAP_BONE)))
    if "intensity" in data:
        items.append(("Intensity surrogate", colorize_gray(data["intensity"], cv2.COLORMAP_BONE)))

    if "counts" in data:
        counts = data["counts"]
        if counts.ndim == 3:
            count_map = counts.sum(axis=0)
            peak_map = np.argmax(counts, axis=0).astype(np.float32)
            count_hist = counts.sum(axis=(1, 2))
            count_map_img = colorize_gray(count_map, cv2.COLORMAP_MAGMA)
            if pixel_points:
                count_map_img = draw_points_on_map(count_map_img, pixel_points, (count_map.shape[0], count_map.shape[1]))
            items.append(("Photon count map", count_map_img))
            items.append(("Photon peak time-bin index", colorize_gray(peak_map, cv2.COLORMAP_VIRIDIS)))
        elif counts.ndim == 2:
            items.append(("Photon count map", colorize_gray(counts, cv2.COLORMAP_MAGMA)))
            count_hist = counts.sum(axis=0)

    if "xhat" in data:
        xhat_views = prepare_xhat_views(data["xhat"])
        for title, (cmap_name, arr) in xhat_views.items():
            cmap = cmap_name_to_cv2(cmap_name)
            img = cv2.cvtColor(normalize_to_uint8(arr), cv2.COLOR_GRAY2BGR) if cmap is None else colorize_gray(arr, cmap)
            items.append((title, img))

    if "x" in data:
        x = data["x"]
        if x.ndim == 3 and x.shape[0] == 3:
            items.append(("Saved x channel 0", colorize_gray(x[0], cv2.COLORMAP_VIRIDIS)))
            items.append(("Saved x channel 1", colorize_gray(x[1], cv2.COLORMAP_MAGMA)))
            items.append(("Saved x channel 2", colorize_gray(x[2], cv2.COLORMAP_MAGMA)))

    return items, count_hist


def build_mode_specific_items(data, resolved_mode):
    items = []

    if resolved_mode == "neighborhood_mix":
        if "transient_clean" in data and data["transient_clean"].ndim == 3:
            transient_clean = data["transient_clean"]
            items.append(("Clean transient sum over time", colorize_gray(transient_clean.sum(axis=0), cv2.COLORMAP_MAGMA)))
            items.append(("Clean transient peak time-bin", colorize_gray(np.argmax(transient_clean, axis=0).astype(np.float32), cv2.COLORMAP_VIRIDIS)))

    elif resolved_mode == "translucent_layer":
        if "transient_clean" in data and data["transient_clean"].ndim == 3:
            transient_clean = data["transient_clean"]
            items.append(("Clean transient sum over time", colorize_gray(transient_clean.sum(axis=0), cv2.COLORMAP_MAGMA)))
            items.append(("Clean transient peak time-bin", colorize_gray(np.argmax(transient_clean, axis=0).astype(np.float32), cv2.COLORMAP_VIRIDIS)))
        if "front_depth_m" in data:
            items.append(("Front layer depth (meters)", colorize_gray(data["front_depth_m"], cv2.COLORMAP_VIRIDIS)))
        if "front_signal" in data:
            items.append(("Front layer signal", colorize_gray(data["front_signal"], cv2.COLORMAP_MAGMA)))
        if "back_signal_after_transmission" in data:
            items.append(("Back signal after transmission", colorize_gray(data["back_signal_after_transmission"], cv2.COLORMAP_MAGMA)))
        if "front_tof_bin" in data:
            items.append(("Front layer TOF bin", colorize_gray(data["front_tof_bin"], cv2.COLORMAP_VIRIDIS)))
        if "back_tof_bin" in data:
            items.append(("Back scene TOF bin", colorize_gray(data["back_tof_bin"], cv2.COLORMAP_VIRIDIS)))

    elif resolved_mode == "volume_scattering":
        if "transient_clean" in data and data["transient_clean"].ndim == 3:
            transient_clean = data["transient_clean"]
            items.append(("Clean transient sum over time", colorize_gray(transient_clean.sum(axis=0), cv2.COLORMAP_MAGMA)))
            items.append(("Clean transient peak time-bin", colorize_gray(np.argmax(transient_clean, axis=0).astype(np.float32), cv2.COLORMAP_VIRIDIS)))
        if "volume_depth_limit_m" in data:
            items.append(("Volume scatter depth limit (m)", colorize_gray(data["volume_depth_limit_m"], cv2.COLORMAP_VIRIDIS)))
        if "volume_scatter_signal" in data:
            items.append(("Volume scatter signal", colorize_gray(data["volume_scatter_signal"], cv2.COLORMAP_MAGMA)))
        if "surface_signal_after_medium" in data:
            items.append(("Surface signal after medium", colorize_gray(data["surface_signal_after_medium"], cv2.COLORMAP_MAGMA)))
        if "volume_scatter_tof_map" in data:
            items.append(("Volume scatter TOF map", colorize_gray(data["volume_scatter_tof_map"], cv2.COLORMAP_VIRIDIS)))
        if "surface_tof_bin" in data:
            items.append(("Surface TOF bin", colorize_gray(data["surface_tof_bin"], cv2.COLORMAP_VIRIDIS)))

    return items


def build_display_items(data, browse_mode="auto", pixel_source="auto"):
    resolved_mode = resolve_browse_mode(data, browse_mode)

    pixel_hists = []
    pixel_points = []
    hist_source_name = "none"

    hist_volume, hist_source_name = build_hist_source(data, pixel_source, resolved_mode)
    if hist_volume is not None:
        _, H, W = hist_volume.shape
        pixel_points = choose_three_pixels(H, W)
        for (r, c) in pixel_points:
            pixel_hists.append(hist_volume[:, r, c])

    common_items, count_hist = build_common_items(data, pixel_points)
    mode_items = build_mode_specific_items(data, resolved_mode)

    items = common_items + mode_items
    return items, count_hist, pixel_hists, pixel_points, hist_source_name, resolved_mode


def make_metadata_lines(data, sample_path, index, total, pixel_source_name, resolved_mode, browse_mode_arg):
    lines = [
        f"sample: {sample_path.name}",
        f"index: {index + 1}/{total}",
        f"browse_mode: {browse_mode_arg}",
        f"resolved_mode: {resolved_mode}",
        f"pixel_hist_source: {pixel_source_name}",
    ]
    for key in [
        "source_mode",
        "scene",
        "surface_model",
        "rgb_file",
        "depth_file",
        "time_diff",
        "mean_signal_photons",
        "mean_background_photons",
        "sbr",
        "bins",
        "bin_size",
        "irf_sigma",
    ]:
        if key in data:
            val = data[key]
            if isinstance(val, np.ndarray) and val.ndim == 0:
                val = val.item()
            lines.append(f"{key}: {val}")
    lines.extend([
        "",
        "Keys:",
        "A/D or <-/-> : prev / next",
        "R            : random",
        "Home/End     : first / last",
        "G            : save current canvas",
        "I            : print sample info",
        "Q or Esc     : quit",
    ])
    return lines


def pad_to_same_height(images):
    hmax = max(img.shape[0] for img in images)
    padded = []
    for img in images:
        h, w = img.shape[:2]
        if h == hmax:
            padded.append(img)
        else:
            pad = np.full((hmax - h, w, 3), 255, dtype=np.uint8)
            padded.append(np.vstack([img, pad]))
    return padded


def assemble_canvas(items, hist, pixel_hists, pixel_points, hist_source_name, meta_lines, cell_size=320, cols=3):
    title_h = 32
    panels = [render_panel(title, img, cell_size=cell_size, title_height=title_h) for title, img in items]
    if len(panels) == 0:
        panels = [np.full((cell_size + title_h, cell_size, 3), 255, dtype=np.uint8)]

    rows = []
    for i in range(0, len(panels), cols):
        row = panels[i:i + cols]
        if len(row) < cols:
            blank = np.full_like(panels[0], 255)
            while len(row) < cols:
                row.append(blank.copy())
        rows.append(np.hstack(row))
    main_grid = np.vstack(rows)

    side_w = max(cell_size + 80, 420)
    hist_panel = render_histogram_panel(hist, side_w, main_grid.shape[0] // 2, title="Global photon histogram", subtitle="from counts")
    meta_panel = render_text_panel(meta_lines, side_w, main_grid.shape[0] - hist_panel.shape[0], title="Metadata")
    side_col = np.vstack([hist_panel, meta_panel])

    pixel_col_w = max(cell_size + 40, 400)
    pixel_col_h = main_grid.shape[0]
    each_h = pixel_col_h // 3
    colors = [(0, 0, 255), (0, 180, 0), (255, 0, 0)]
    labels = ["P1", "P2", "P3"]
    pixel_panels = []
    for i in range(3):
        height_i = each_h if i < 2 else pixel_col_h - each_h * 2
        if i < len(pixel_hists):
            r, c = pixel_points[i]
            subtitle = f"{hist_source_name}: pixel ({r}, {c})"
            panel = render_pixel_histogram_panel(
                pixel_hists[i],
                pixel_col_w,
                height_i,
                title=f"{labels[i]} photon histogram",
                color=colors[i],
                subtitle=subtitle,
            )
        else:
            panel = render_pixel_histogram_panel(
                None,
                pixel_col_w,
                height_i,
                title=f"{labels[i]} photon histogram",
                color=colors[i],
                subtitle="N/A",
            )
        pixel_panels.append(panel)
    pixel_col = np.vstack(pixel_panels)

    main_grid, side_col, pixel_col = pad_to_same_height([main_grid, side_col, pixel_col])
    canvas = np.hstack([main_grid, side_col, pixel_col])

    footer_h = 34
    footer = np.full((footer_h, canvas.shape[1], 3), 245, dtype=np.uint8)
    msg = "A/D or <-/-> prev/next | R random | Home/End first/last | G save | I info | Q/Esc quit"
    cv2.putText(footer, msg, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    canvas = np.vstack([canvas, footer])
    return canvas


class SPADBrowser:
    def __init__(
        self,
        files,
        start_index=0,
        save_dir=None,
        window_name="SPAD Dataset Browser",
        cell_size=320,
        browse_mode="auto",
        pixel_source="auto",
        overwrite=False,
        parser=None,
    ):
        self.files = files
        self.index = max(0, min(start_index, len(files) - 1))
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.window_name = window_name
        self.cell_size = cell_size
        self.browse_mode = browse_mode
        self.pixel_source = pixel_source
        self.overwrite = overwrite
        self.parser = parser

    def load_current(self):
        path = self.files[self.index]
        try:
            with np.load(path, allow_pickle=True) as archive:
                data = {key: archive[key] for key in archive.files}
        except (
            EOFError,
            OSError,
            TypeError,
            ValueError,
            pickle.UnpicklingError,
            zipfile.BadZipFile,
        ) as exc:
            if self.parser is not None:
                self.parser.error(f"cannot read sample {path}: {exc}")
            raise ValueError(f"Cannot read sample {path}: {exc}") from exc
        return path, data

    def print_info(self):
        path, data = self.load_current()
        print("=" * 80)
        print(f"Sample file: {path.name}")
        print(f"Full path   : {path}")
        print("-" * 80)
        for key in data:
            arr = data[key]
            if isinstance(arr, np.ndarray):
                if arr.ndim == 0:
                    print(f"{key:>24s} | scalar={arr.item()} dtype={arr.dtype}")
                else:
                    print(f"{key:>24s} | shape={arr.shape}, dtype={arr.dtype}")
            else:
                print(f"{key:>24s} | value={arr}")
        print("=" * 80)

    def save_canvas(self, canvas):
        out_dir = self.files[self.index].parent / "browse_exports" if self.save_dir is None else self.save_dir
        out_path = out_dir / f"{self.files[self.index].stem}_browse.png"
        if self.parser is not None:
            out_path = validate_output_file(
                self.parser,
                str(out_path),
                overwrite=self.overwrite,
                option="saved canvas",
                suffixes=(".png",),
            )
            create_output_parent(self.parser, out_path, option="saved canvas")
        else:
            if out_path.exists() and not self.overwrite:
                raise FileExistsError(
                    f"Saved canvas already exists: {out_path}; enable overwrite to replace it"
                )
            out_dir.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_path), canvas):
            if self.parser is not None:
                self.parser.error(f"failed to save canvas: {out_path}")
            raise OSError(f"Failed to save canvas: {out_path}")
        print(f"Saved figure to: {out_path}")

    def build_canvas(self):
        path, data = self.load_current()
        items, hist, pixel_hists, pixel_points, hist_source_name, resolved_mode = build_display_items(
            data,
            browse_mode=self.browse_mode,
            pixel_source=self.pixel_source,
        )
        meta_lines = make_metadata_lines(
            data,
            path,
            self.index,
            len(self.files),
            hist_source_name,
            resolved_mode,
            self.browse_mode,
        )
        return assemble_canvas(
            items,
            hist,
            pixel_hists,
            pixel_points,
            hist_source_name,
            meta_lines,
            cell_size=self.cell_size,
            cols=3,
        )

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 2000, 1000)
        while True:
            canvas = self.build_canvas()
            cv2.imshow(self.window_name, canvas)
            key = cv2.waitKeyEx(0)
            if key in [27, ord("q"), ord("Q")]:
                break
            elif key in [2424832, ord("a"), ord("A")]:
                self.index = (self.index - 1) % len(self.files)
            elif key in [2555904, ord("d"), ord("D")]:
                self.index = (self.index + 1) % len(self.files)
            elif key in [ord("r"), ord("R")]:
                self.index = random.randrange(len(self.files))
            elif key == 2359296:
                self.index = 0
            elif key == 2293760:
                self.index = len(self.files) - 1
            elif key in [ord("g"), ord("G")]:
                self.save_canvas(canvas)
            elif key in [ord("i"), ord("I")]:
                self.print_info()
        cv2.destroyAllWindows()


def main():
    parser = build_parser()
    args = parser.parse_args()
    dataset_dir = require_directory(parser, args.dataset_dir, "--dataset_dir")

    files = list_npz_files(dataset_dir)
    if not files:
        parser.error(f"--dataset_dir contains no .npz samples: {dataset_dir}")
    for sample_path in files:
        validate_npz_archive(parser, sample_path, "dataset sample")
    if not args.random_start and args.start_index >= len(files):
        parser.error(
            f"--start_index {args.start_index} is out of range for {len(files)} sample(s); "
            f"expected 0 to {len(files) - 1}"
        )

    output_dir = None
    if args.output_dir is not None:
        output_dir = validate_output_directory(
            parser,
            args.output_dir,
            overwrite=args.overwrite,
            option="--output_dir",
            require_empty=False,
        )

    if cv2 is None:
        parser.error(
            "OpenCV is required for the dataset browser; install it with "
            "'pip install opencv-python'"
        )

    start_index = random.randrange(len(files)) if args.random_start else args.start_index

    browser = SPADBrowser(
        files=files,
        start_index=start_index,
        save_dir=output_dir,
        window_name=args.window_name,
        cell_size=args.cell_size,
        browse_mode=args.browse_mode,
        pixel_source=args.pixel_source,
        overwrite=args.overwrite,
        parser=parser,
    )
    browser.run()


if __name__ == "__main__":
    main()

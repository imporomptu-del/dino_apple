"""
Optical flow motion detection — NvOF 2.0 + dual-path detection (RTX 4070 optimised).

Pipeline order per frame:
  1. Read frame (full res)
  2. Apply mask        ← kills unwanted pixels immediately
  3. Rescale           ← process_scale if < 1.0
  4. Grayscale
  5. Path A: Brightness threshold + persistence filter  ← slow/stationary bright objects
     Path B: MOG2 background subtraction + NvOF flow   ← moving objects
  6. OR both masks → morphology → contours → CSV / annotate
     CSV column detection_type: "bright", "motion", or "both"

Backend priority:
  1. NVIDIA OFA hardware (NvidiaOpticalFlow_2_0) — RTX 2000+, ~2 ms/frame
  2. GPU RAFT (torchvision)                       — any CUDA GPU, ~30 ms/frame
  3. CPU Farneback                                — no CUDA, ~80 ms/frame
"""

import argparse
import csv
import gc
import os
import subprocess
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import (
    Raft_Large_Weights, Raft_Small_Weights, raft_large, raft_small,
)

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

_REF_WIDTH   = 2592
_REF_HEIGHT  = 1944
_REF_VRAM_GB = 23.0

# NvOF: resize masked frame to this fraction before running OFA, upsample after.
# 0.5 → 1296×972 for 2592×1944 — sweet spot for RTX 4070.
NVOF_SCALE = 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    torch.backends.cudnn.benchmark = True
    return torch.device("cuda")


def _scaled_flow_size(width: int, height: int, scale: float) -> tuple[int, int]:
    flow_w = max(8, int(width  * scale) // 8 * 8)
    flow_h = max(8, int(height * scale) // 8 * 8)
    return flow_w, flow_h


def _estimate_vram_gb(width: int, height: int, flow_w: int, flow_h: int) -> float:
    pixel_ratio = (width * height) / (_REF_WIDTH * _REF_HEIGHT)
    scale_ratio = (flow_w * flow_h) / (width * height)
    return _REF_VRAM_GB * pixel_ratio * (scale_ratio ** 2)


def pick_flow_size(width: int, height: int, device: torch.device,
                   safety: float = 0.65, reserve_gb: float = 0.0) -> tuple[int, int]:
    if device.type != "cuda":
        return width, height
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    free_gb  = free_bytes  / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)
    budget_gb = max(0.5, free_gb - reserve_gb)
    print(f"GPU memory free: {free_gb:.1f} GB / {total_gb:.1f} GB"
          + (f" (budget {budget_gb:.1f} GB after {reserve_gb:.1f} GB reserve)" if reserve_gb else ""))
    if budget_gb < 2.0:
        print("Warning: little GPU memory free — close other apps before running.")
    for scale in (1.0, 0.75, 0.5, 0.375, 0.25):
        flow_w, flow_h = _scaled_flow_size(width, height, scale)
        needed_gb = _estimate_vram_gb(width, height, flow_w, flow_h)
        if needed_gb <= budget_gb * safety:
            print(f"RAFT flow size: {flow_w}x{flow_h} "
                  f"(~{flow_w / width:.0%} of frame, est. VRAM {needed_gb:.1f} GB)")
            return flow_w, flow_h
    flow_w, flow_h = _scaled_flow_size(width, height, 0.25)
    print(f"RAFT flow size (minimum): {flow_w}x{flow_h}")
    return flow_w, flow_h


def cpu_farneback_flow(prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0,
    )
    return np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)


# ──────────────────────────────────────────────────────────────────────────────
# Persistence tracker — for brightness path
# ──────────────────────────────────────────────────────────────────────────────

class PersistenceTracker:
    """
    Tracks how many consecutive frames each bright region has appeared.
    Regions that appear for >= min_frames are passed through.
    Regions that appear for >= max_frames are suppressed (stars — always bright).

    Works by tracking centroid positions across frames with a simple
    nearest-neighbour match.
    """

    def __init__(self, min_frames: int = 3, max_frames: int = 0,
                 match_dist: int = 20):
        self.min_frames  = min_frames
        self.max_frames  = max_frames   # 0 = no upper limit
        self.match_dist  = match_dist
        # dict: centroid_key → consecutive frame count
        self._counts: dict[tuple, int] = {}

    def _centroid(self, cnt) -> tuple[int, int]:
        x, y, w, h = cv2.boundingRect(cnt)
        return (x + w // 2, y + h // 2)

    def update(self, contours: list,
               gray: np.ndarray) -> tuple[list, list]:
        """
        Returns (passed_contours, detection_types).
        passed_contours: contours that passed persistence filter.
        """
        new_counts: dict[tuple, int] = {}
        passed = []

        for cnt in contours:
            cx, cy = self._centroid(cnt)

            # find closest existing tracked centroid
            best_key = None
            best_dist = self.match_dist
            for key in self._counts:
                d = abs(key[0] - cx) + abs(key[1] - cy)
                if d < best_dist:
                    best_dist = d
                    best_key = key

            new_key = (cx, cy)
            if best_key is not None:
                new_counts[new_key] = self._counts[best_key] + 1
            else:
                new_counts[new_key] = 1

            count = new_counts[new_key]

            # suppress if below min or above max
            if count < self.min_frames:
                continue
            if self.max_frames > 0 and count >= self.max_frames:
                continue

            passed.append(cnt)

        self._counts = new_counts
        return passed


# ──────────────────────────────────────────────────────────────────────────────
# Resource logger
# ──────────────────────────────────────────────────────────────────────────────

class ResourceLogger:
    def __init__(self, log_dir: str, run_name: str, cuda_device_index: int = 0):
        os.makedirs(log_dir, exist_ok=True)
        self.per_frame_path = os.path.join(log_dir, f"{run_name}_resources.csv")
        self.summary_path   = os.path.join(log_dir, f"{run_name}_summary.log")
        self.process        = psutil.Process()
        self.gpu_handle     = None
        self.rows           = []
        psutil.cpu_percent(interval=None)
        if PYNVML_AVAILABLE and torch.cuda.is_available():
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(cuda_device_index)

    def sample(self) -> dict:
        stats = {
            "cpu_percent":        psutil.cpu_percent(interval=None),
            "system_cpu_percent": psutil.cpu_percent(interval=None),
            "ram_mb":             self.process.memory_info().rss / (1024 ** 2),
            "gpu_util_percent":   None,
            "gpu_mem_used_mb":    None,
            "gpu_mem_total_mb":   None,
        }
        if self.gpu_handle is not None:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            mem  = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            stats["gpu_util_percent"] = float(util.gpu)
            stats["gpu_mem_used_mb"]  = mem.used  / (1024 ** 2)
            stats["gpu_mem_total_mb"] = mem.total / (1024 ** 2)
        return stats

    def log_frame(self, frame_idx, time_sec, gpu_flow_ms, cpu_ms, frame_ms, detections):
        stats = self.sample()
        self.rows.append({
            "frame": frame_idx, "time_sec": round(time_sec, 4),
            "gpu_flow_ms": round(gpu_flow_ms, 2), "cpu_ms": round(cpu_ms, 2),
            "frame_ms": round(frame_ms, 2), "detections": detections, **stats,
        })
        gpu_str = (f"{stats['gpu_util_percent']:.0f}%"
                   if stats["gpu_util_percent"] is not None else "n/a")
        print(f"Frame {frame_idx}: GPU flow {gpu_flow_ms:.0f} ms | CPU {cpu_ms:.0f} ms | "
              f"CPU {stats['cpu_percent']:.1f}% | RAM {stats['ram_mb']:.0f} MB | GPU {gpu_str}")

    def write_logs(self, total_frames, elapsed_sec, input_video, output_video, backend: str):
        fieldnames = [
            "frame", "time_sec", "gpu_flow_ms", "cpu_ms", "frame_ms", "detections",
            "cpu_percent", "system_cpu_percent", "ram_mb",
            "gpu_util_percent", "gpu_mem_used_mb", "gpu_mem_total_mb",
        ]
        with open(self.per_frame_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({k: ("" if row[k] is None else row[k]) for k in fieldnames})
        with open(self.summary_path, "w") as f:
            f.write(f"Backend: {backend}\nInput: {input_video}\nOutput: {output_video}\n"
                    f"Frames: {total_frames}\nElapsed: {elapsed_sec:.1f} s\n"
                    f"CSV: {self.per_frame_path}\n")
        print(f"Saved resource log: {self.per_frame_path}")
        print(f"Saved summary log:  {self.summary_path}")

    def close(self):
        if self.gpu_handle is not None and PYNVML_AVAILABLE:
            pynvml.nvmlShutdown()


# ──────────────────────────────────────────────────────────────────────────────
# Mask loader
# ──────────────────────────────────────────────────────────────────────────────

def load_valid_region_mask(mask_path, width, height,
                           threshold=127, invert=False) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not load mask: {mask_path}")
    if mask.shape[1] != width or mask.shape[0] != height:
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    valid = (mask > threshold).astype(np.uint8) * 255
    if invert:
        valid = 255 - valid
    valid_pixels = int(np.count_nonzero(valid))
    print(f"Mask loaded: {mask_path}")
    print(f"Valid pixels: {valid_pixels}/{valid.size} "
          f"({100 * valid_pixels / valid.size:.1f}%) — applied first every frame")
    return valid


# ──────────────────────────────────────────────────────────────────────────────
# Flow estimator
# ──────────────────────────────────────────────────────────────────────────────

class HybridFlowEstimator:
    RAFT_MIN_SIZE = 128

    def __init__(self, model_name: str, device: torch.device,
                 flow_w: int, flow_h: int, full_w: int, full_h: int):
        self.device       = device
        self.flow_w       = flow_w
        self.flow_h       = flow_h
        self.full_w       = full_w
        self.full_h       = full_h
        self.use_gpu      = device.type == "cuda"
        self.model        = None
        self.transforms   = None
        self.nvof         = None
        self.nvof_w       = None
        self.nvof_h       = None
        self.backend_name = "cpu_farneback"
        self._prev_hint: cv2.cuda.GpuMat | None = None

        if self.use_gpu:
            self._init_nvof()
            if self.nvof is None:
                self._init_raft(model_name)
        else:
            print("Flow backend: CPU Farneback (CUDA unavailable)")

    def _init_nvof(self) -> None:
        nvof_w = max(8, int(self.full_w * NVOF_SCALE) // 4 * 4)
        nvof_h = max(8, int(self.full_h * NVOF_SCALE) // 4 * 4)
        try:
            self.nvof = cv2.cuda.NvidiaOpticalFlow_2_0.create(
                imageSize=(nvof_w, nvof_h),
                perfPreset=5,
                enableTemporalHints=True,
                enableExternalHints=False,
                enableCostBuffer=False,
                gpuId=0,
            )
            self.nvof_w = nvof_w
            self.nvof_h = nvof_h
            self.backend_name = "nvof_2_0"
            print(f"Flow backend: NVIDIA OFA (NvidiaOpticalFlow_2_0) "
                  f"at {nvof_w}x{nvof_h} ({NVOF_SCALE:.0%} of {self.full_w}x{self.full_h})")
        except Exception as exc:
            print(f"NvOF 2.0 unavailable ({exc}) — falling back to RAFT")
            self.nvof = None

    def _init_raft(self, model_name: str) -> None:
        if model_name == "raft_small":
            weights    = Raft_Small_Weights.DEFAULT
            self.model = raft_small(weights=weights, progress=False)
        else:
            weights    = Raft_Large_Weights.DEFAULT
            self.model = raft_large(weights=weights, progress=False)
        self.model = self.model.to(self.device).eval()
        self.transforms   = weights.transforms()
        self.backend_name = f"raft_{model_name}"
        print(f"Flow backend: GPU RAFT ({model_name})")

    def _nvof_compute(self, prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> np.ndarray:
        gpu_prev = cv2.cuda_GpuMat()
        gpu_curr = cv2.cuda_GpuMat()
        gpu_prev.upload(prev_bgr)
        gpu_curr.upload(curr_bgr)
        gpu_prev_s = cv2.cuda.resize(gpu_prev, (self.nvof_w, self.nvof_h),
                                     interpolation=cv2.INTER_LINEAR)
        gpu_curr_s = cv2.cuda.resize(gpu_curr, (self.nvof_w, self.nvof_h),
                                     interpolation=cv2.INTER_LINEAR)
        gpu_prev_g = cv2.cuda.cvtColor(gpu_prev_s, cv2.COLOR_BGR2GRAY)
        gpu_curr_g = cv2.cuda.cvtColor(gpu_curr_s, cv2.COLOR_BGR2GRAY)
        flow_gpu, _ = self.nvof.calc(gpu_prev_g, gpu_curr_g, self._prev_hint)
        self._prev_hint = flow_gpu
        float_flow = self.nvof.convertToFloat(flow_gpu, None)
        flow_cpu = float_flow.download()
        flow_resized = cv2.resize(flow_cpu, (self.full_w, self.full_h),
                                  interpolation=cv2.INTER_LINEAR)
        return np.sqrt(flow_resized[..., 0] ** 2 + flow_resized[..., 1] ** 2)

    @torch.inference_mode()
    def _raft_compute(self, prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> np.ndarray:
        prev_rgb = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2RGB)
        curr_rgb = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2RGB)
        prev_t = (torch.from_numpy(prev_rgb).permute(2, 0, 1)
                  .unsqueeze(0).float().to(self.device))
        curr_t = (torch.from_numpy(curr_rgb).permute(2, 0, 1)
                  .unsqueeze(0).float().to(self.device))
        if self.flow_w != self.full_w or self.flow_h != self.full_h:
            prev_t = F.interpolate(prev_t, size=(self.flow_h, self.flow_w),
                                   mode="bilinear", align_corners=False)
            curr_t = F.interpolate(curr_t, size=(self.flow_h, self.flow_w),
                                   mode="bilinear", align_corners=False)
        img1, img2 = self.transforms(prev_t, curr_t)
        flow = self.model(img1, img2)[-1]
        flow_x   = flow[0, 0].float() * (self.full_w / self.flow_w)
        flow_y   = flow[0, 1].float() * (self.full_h / self.flow_h)
        flow_mag = torch.sqrt(flow_x ** 2 + flow_y ** 2)
        if self.flow_w != self.full_w or self.flow_h != self.full_h:
            flow_mag = F.interpolate(
                flow_mag.unsqueeze(0).unsqueeze(0),
                size=(self.full_h, self.full_w),
                mode="bilinear", align_corners=False,
            ).squeeze()
        result = flow_mag.cpu().numpy()
        del prev_t, curr_t, img1, img2, flow, flow_mag
        return result

    def compute(self, prev_bgr, curr_bgr, prev_gray, gray) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        if self.nvof is not None:
            flow_mag = self._nvof_compute(prev_bgr, curr_bgr)
        elif self.use_gpu and self.model is not None:
            flow_mag = self._raft_compute(prev_bgr, curr_bgr)
        else:
            flow_mag = cpu_farneback_flow(prev_gray, gray)
        return flow_mag, (time.perf_counter() - t0) * 1000

    def reset_temporal_hints(self) -> None:
        self._prev_hint = None


# ──────────────────────────────────────────────────────────────────────────────
# Remux helper
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Trail renderer
# ──────────────────────────────────────────────────────────────────────────────

class TrailRenderer:
    """
    Keeps a rolling buffer of recent detection centroids and draws
    fading trails on the output frame.

    Each entry: (cx, cy, color) stored in order oldest→newest.
    When drawing, older entries are rendered with lower alpha (darker).
    """

    def __init__(self, trail_length: int = 30):
        self.trail_length = trail_length
        # deque of (cx, cy, color_bgr)
        self._points: deque = deque(maxlen=trail_length)

    def add(self, cx: float, cy: float, color: tuple) -> None:
        self._points.append((cx, cy, color))

    def draw(self, frame: np.ndarray) -> None:
        n = len(self._points)
        for i, (cx, cy, color) in enumerate(self._points):
            # alpha goes from 0.1 (oldest) to 1.0 (newest)
            alpha = 0.1 + 0.9 * (i / max(n - 1, 1))
            faded = tuple(int(c * alpha) for c in color)
            radius = max(1, int(2 * alpha))
            cv2.circle(frame, (int(cx), int(cy)), radius, faded, -1)

    def clear(self) -> None:
        self._points.clear()


def remux_mp4_faststart(video_path: str) -> None:
    tmp_path = video_path + ".faststart.tmp.mp4"
    result = subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-i", video_path, "-c", "copy", "-movflags", "+faststart", tmp_path],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and os.path.isfile(tmp_path):
        os.replace(tmp_path, video_path)
    elif os.path.isfile(tmp_path):
        os.remove(tmp_path)


# ──────────────────────────────────────────────────────────────────────────────
# Main processing loop
# ──────────────────────────────────────────────────────────────────────────────

def process_video(
    input_video,
    output_video,
    output_csv,
    log_dir,
    run_name,
    min_area=3,
    max_area=500000,
    # ── Path A: brightness ────────────────────────────────────────────────────
    bright_threshold=30,      # pixel must be brighter than this to be a candidate
    bright_min_frames=3,      # must appear in this many consecutive frames
    bright_max_frames=0,      # suppress if seen this long (0 = no limit)
    # ── Path B: motion ───────────────────────────────────────────────────────
    diff_threshold=20,        # absdiff threshold for motion path
    flow_threshold=0.5,       # NvOF flow magnitude threshold
    # ── shared ───────────────────────────────────────────────────────────────
    flow_model="raft_small",
    flow_scale=0.0,
    mask_path=None,
    mask_threshold=127,
    mask_invert=False,
    log_interval=50,
    tile_size=384,
    tile_batch=6,
    process_scale=1.0,
    trail_length=30,
    output_mode="video",   # "video" | "flow" | "mask"
):
    torch_device    = resolve_device()
    resource_logger = ResourceLogger(log_dir, run_name, cuda_device_index=0)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        resource_logger.close()
        raise RuntimeError(f"Could not open video: {input_video}")

    fps         = cap.get(cv2.CAP_PROP_FPS) or 30
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    proc_w = max(8, int(width  * process_scale)) if process_scale != 1.0 else width
    proc_h = max(8, int(height * process_scale)) if process_scale != 1.0 else height
    if process_scale != 1.0:
        print(f"Process scale: {process_scale} → internal {proc_w}x{proc_h}")

    flow_w, flow_h = proc_w, proc_h
    flow_estimator = HybridFlowEstimator(
        flow_model, torch_device, flow_w, flow_h, proc_w, proc_h)
    if flow_estimator.nvof is None and flow_estimator.model is not None:
        if flow_scale <= 0:
            flow_w, flow_h = pick_flow_size(proc_w, proc_h, torch_device)
        else:
            flow_w, flow_h = _scaled_flow_size(proc_w, proc_h, flow_scale)
        flow_estimator.flow_w = flow_w
        flow_estimator.flow_h = flow_h

    # Persistence tracker for brightness path
    persistence = PersistenceTracker(
        min_frames=bright_min_frames,
        max_frames=bright_max_frames,
    )

    # Trail renderer — draws fading centroid history on output frames
    trail = TrailRenderer(trail_length=trail_length)

    backend = (f"{flow_estimator.backend_name} + dual-path (bright+motion) "
               f"({proc_w}x{proc_h})")

    print(f"Input:      {input_video}")
    print(f"Resolution: {width}x{height} @ {fps:.2f} fps")
    if torch_device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(torch_device)}")
    print(f"Backend:    {backend}")
    print(f"Trail length: {trail_length} frames")
    print(f"Path A (bright): threshold={bright_threshold}, "
          f"min_frames={bright_min_frames}, max_frames={bright_max_frames}")
    print(f"Path B (motion): diff={diff_threshold}, flow={flow_threshold}")
    print(f"Output mode: {output_mode}")
    print(f"Output:     {output_video}")
    print(f"CSV:        {output_csv}")

    mask_full = None
    mask_proc = None
    if mask_path:
        mask_full = load_valid_region_mask(
            mask_path, width, height, mask_threshold, mask_invert)
        mask_proc = (cv2.resize(mask_full, (proc_w, proc_h),
                                interpolation=cv2.INTER_NEAREST)
                     if process_scale != 1.0 else mask_full)

    os.makedirs(os.path.dirname(os.path.abspath(output_video)), exist_ok=True)
    writer = cv2.VideoWriter(
        output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        resource_logger.close(); cap.release()
        raise RuntimeError(f"Could not open output writer: {output_video}")

    ret, prev_frame_raw = cap.read()
    if not ret:
        resource_logger.close(); cap.release(); writer.release()
        raise RuntimeError("Could not read first frame")

    prev_masked = (cv2.bitwise_and(prev_frame_raw, prev_frame_raw, mask=mask_full)
                   if mask_full is not None else prev_frame_raw)
    prev_proc = (cv2.resize(prev_masked, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
                 if process_scale != 1.0 else prev_masked)
    prev_gray = cv2.cvtColor(prev_proc, cv2.COLOR_BGR2GRAY)

    kernel    = np.ones((3, 3), np.uint8)
    frame_idx = 0
    run_start = time.perf_counter()

    try:
        with open(output_csv, "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                "frame", "time_sec", "object_id", "x", "y", "w", "h",
                "area", "cx", "cy", "mean_flow_mag", "detection_type",
            ])

            while True:
                ret, frame_raw = cap.read()
                if not ret:
                    break

                frame_idx += 1
                time_sec    = frame_idx / fps
                frame_start = time.perf_counter()
                cpu_start   = time.perf_counter()

                # ── STEP 1: mask ──────────────────────────────────────────
                if mask_full is not None:
                    frame_masked = cv2.bitwise_and(
                        frame_raw, frame_raw, mask=mask_full)
                else:
                    frame_masked = frame_raw

                # ── STEP 2: rescale ───────────────────────────────────────
                if process_scale != 1.0:
                    proc_frame = cv2.resize(
                        frame_masked, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
                else:
                    proc_frame = frame_masked

                # ── STEP 3: grayscale ─────────────────────────────────────
                gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)

                # ── PATH A: brightness ────────────────────────────────────
                # Detects bright objects regardless of motion.
                # Persistence filter removes noise (< min_frames)
                # and optionally stars (>= max_frames if set).
                _, bright_mask = cv2.threshold(
                    gray, bright_threshold, 255, cv2.THRESH_BINARY)
                bright_contours, _ = cv2.findContours(
                    bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                bright_contours = persistence.update(bright_contours, gray)

                # Rebuild bright mask from persisted contours only
                bright_mask_filtered = np.zeros_like(gray)
                cv2.drawContours(bright_mask_filtered, bright_contours, -1, 255, -1)

                # ── PATH B: motion (absdiff + NvOF) ──────────────────────
                frame_diff = cv2.absdiff(prev_gray, gray)
                _, diff_mask = cv2.threshold(
                    frame_diff, diff_threshold, 255, cv2.THRESH_BINARY)

                cpu_before_flow_ms = (time.perf_counter() - cpu_start) * 1000

                if torch_device.type == "cuda":
                    torch.cuda.synchronize()

                flow_mag, gpu_flow_ms = flow_estimator.compute(
                    prev_proc, proc_frame, prev_gray, gray)

                if torch_device.type == "cuda":
                    torch.cuda.synchronize()

                cpu_start = time.perf_counter()

                if process_scale != 1.0:
                    flow_mag             = cv2.resize(flow_mag, (width, height),
                                                      interpolation=cv2.INTER_LINEAR)
                    diff_mask            = cv2.resize(diff_mask, (width, height),
                                                      interpolation=cv2.INTER_NEAREST)
                    bright_mask_filtered = cv2.resize(bright_mask_filtered, (width, height),
                                                      interpolation=cv2.INTER_NEAREST)

                _, flow_mask = cv2.threshold(
                    flow_mag.astype(np.float32), flow_threshold, 255, cv2.THRESH_BINARY)
                flow_mask = flow_mask.astype(np.uint8)

                motion_mask = cv2.bitwise_and(diff_mask, flow_mask)
                motion_mask = cv2.morphologyEx(
                    motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
                motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)

                # ── COMBINE: bright OR motion ─────────────────────────────
                combined_mask = cv2.bitwise_or(bright_mask_filtered, motion_mask)
                combined_mask = cv2.morphologyEx(
                    combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

                # ── STEP 7: contours → CSV → annotate ────────────────────
                contours, _ = cv2.findContours(
                    combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                object_id = 0

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < min_area or area > max_area:
                        continue
                    x, y, w, h = cv2.boundingRect(cnt)
                    cx = x + w / 2
                    cy = y + h / 2
                    roi       = flow_mag[y:y + h, x:x + w]
                    mean_flow = float(np.mean(roi)) if roi.size else 0.0

                    # determine which path(s) detected this object
                    cx_i, cy_i = int(cx), int(cy)
                    in_bright = bright_mask_filtered[cy_i, cx_i] > 0
                    in_motion = motion_mask[cy_i, cx_i] > 0
                    if in_bright and in_motion:
                        det_type = "both"
                        color = (0, 165, 255)    # orange
                    elif in_bright:
                        det_type = "bright"
                        color = (255, 255, 0)    # cyan
                    else:
                        det_type = "motion"
                        color = (0, 255, 0)      # green

                    # Add centroid to trail
                    trail.add(cx, cy, color)
                    object_id += 1
                    csv_writer.writerow([
                        frame_idx, round(time_sec, 4), object_id,
                        x, y, w, h,
                        round(area, 2), round(cx, 2), round(cy, 2),
                        round(mean_flow, 4), det_type,
                    ])
                    cv2.rectangle(frame_raw, (x, y), (x + w, y + h), color, 2)
                    cv2.circle(frame_raw, (int(cx), int(cy)), 3, (0, 0, 255), -1)
                    cv2.putText(frame_raw,
                                f"ID:{object_id} A:{int(area)} F:{mean_flow:.2f} [{det_type[0]}]",
                                (x, max(20, y - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                # ── Build output frame based on --output-mode ──────────
                if output_mode == "flow":
                    # Flow magnitude visualised as grayscale
                    flow_vis = cv2.normalize(
                        flow_mag, None, 0, 255, cv2.NORM_MINMAX)
                    flow_vis = flow_vis.astype(np.uint8)
                    output_frame = cv2.cvtColor(flow_vis, cv2.COLOR_GRAY2BGR)
                elif output_mode == "mask":
                    # Combined motion+bright mask as grayscale
                    output_frame = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
                else:
                    # Default: original video frame
                    output_frame = frame_raw

                cv2.putText(output_frame, f"Frame: {frame_idx}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Redraw boxes/labels on output_frame (needed for flow/mask modes)
                if output_mode != "video":
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area < min_area or area > max_area:
                            continue
                        x, y, w, h = cv2.boundingRect(cnt)
                        cx_i = int(x + w / 2)
                        cy_i = int(y + h / 2)
                        in_bright = bright_mask_filtered[cy_i, cx_i] > 0
                        in_motion = motion_mask[cy_i, cx_i] > 0
                        if in_bright and in_motion:
                            c = (0, 165, 255)
                        elif in_bright:
                            c = (255, 255, 0)
                        else:
                            c = (0, 255, 0)
                        cv2.rectangle(output_frame, (x, y), (x + w, y + h), c, 2)

                # Draw fading trail of past detections
                trail.draw(output_frame)
                writer.write(output_frame)

                prev_proc = proc_frame
                prev_gray = gray.copy()

                cpu_ms   = cpu_before_flow_ms + (time.perf_counter() - cpu_start) * 1000
                frame_ms = (time.perf_counter() - frame_start) * 1000

                if frame_idx % log_interval == 0 or frame_idx == max(frame_count - 1, 1):
                    resource_logger.log_frame(
                        frame_idx, time_sec, gpu_flow_ms, cpu_ms, frame_ms, object_id)
                    print(f"Progress: {frame_idx}/{max(frame_count - 1, 0)}")

                if frame_idx % 100 == 0 and torch_device.type == "cuda":
                    gc.collect()
                    torch.cuda.empty_cache()

    finally:
        cap.release()
        writer.release()

    remux_mp4_faststart(output_video)
    elapsed = time.perf_counter() - run_start
    resource_logger.write_logs(frame_idx, elapsed, input_video, output_video, backend)
    resource_logger.close()
    print("Done.")
    print(f"Saved: {output_video}")
    print(f"Saved: {output_csv}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_output_paths(script_dir, input_video, output_arg, csv_arg):
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(input_video))[0]
    return (
        output_arg or os.path.join(output_dir, f"{stem}_motion_nvof.mp4"),
        csv_arg    or os.path.join(output_dir, f"{stem}_detections_nvof.csv"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NvOF 2.0 dual-path detection (bright + motion)")
    parser.add_argument("--input",              required=True)
    parser.add_argument("--output",             default=None)
    parser.add_argument("--csv",                default=None)
    parser.add_argument("--min-area",           type=float, default=3)
    parser.add_argument("--max-area",           type=float, default=500000)
    # Path A
    parser.add_argument("--bright-threshold",   type=int,   default=30,
        help="Brightness threshold for Path A (default 30)")
    parser.add_argument("--bright-min-frames",  type=int,   default=3,
        help="Min consecutive frames a bright object must appear (default 3)")
    parser.add_argument("--bright-max-frames",  type=int,   default=0,
        help="Suppress bright objects seen this many frames — use to filter stars (0=off)")
    # Path B
    parser.add_argument("--diff-threshold",     type=int,   default=20,
        help="Frame diff threshold for motion path (default 20)")
    parser.add_argument("--flow-threshold",     type=float, default=0.5,
        help="NvOF flow magnitude threshold (default 0.5)")
    # shared
    parser.add_argument("--mask",               default=None)
    parser.add_argument("--mask-threshold",     type=int,   default=127)
    parser.add_argument("--mask-invert",        action="store_true")
    parser.add_argument("--flow-model",         default="raft_small",
                        choices=["raft_small", "raft_large"])
    parser.add_argument("--flow-scale",         type=float, default=0.0)
    parser.add_argument("--log-interval",       type=int,   default=50)
    parser.add_argument("--process-scale",      type=float, default=1.0)
    parser.add_argument("--output-mode",        default="video",
        choices=["video", "flow", "mask"],
        help="Output background: video=original, flow=NvOF magnitude, mask=motion mask")
    parser.add_argument("--trail-length",       type=int,   default=30,
        help="Frames of fading trail behind detections (default 30, 0=off)")
    parser.add_argument("--tile-size",          type=int,   default=384)
    parser.add_argument("--tile-batch",         type=int,   default=6)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir    = os.path.join(script_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    default_mask = os.path.join(script_dir, "mask", "mask.png")
    mask_path    = (args.mask if args.mask
                    else (default_mask if os.path.isfile(default_mask) else None))

    stem     = os.path.splitext(os.path.basename(args.input))[0]
    run_name = f"{stem}_nvof_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_video, output_csv = build_output_paths(
        script_dir, args.input, args.output, args.csv)

    process_video(
        input_video        = args.input,
        output_video       = output_video,
        output_csv         = output_csv,
        log_dir            = log_dir,
        run_name           = run_name,
        min_area           = args.min_area,
        max_area           = args.max_area,
        bright_threshold   = args.bright_threshold,
        bright_min_frames  = args.bright_min_frames,
        bright_max_frames  = args.bright_max_frames,
        diff_threshold     = args.diff_threshold,
        flow_threshold     = args.flow_threshold,
        flow_model         = args.flow_model,
        flow_scale         = args.flow_scale,
        mask_path          = mask_path,
        mask_threshold     = args.mask_threshold,
        mask_invert        = args.mask_invert,
        log_interval       = args.log_interval,
        tile_size          = args.tile_size,
        tile_batch         = args.tile_batch,
        process_scale      = args.process_scale,
        trail_length       = args.trail_length,
        output_mode        = args.output_mode,
    )
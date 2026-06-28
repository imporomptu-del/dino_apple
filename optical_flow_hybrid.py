"""
Hybrid optical flow: GPU (RAFT) for flow estimation + CPU (OpenCV) for everything else.

- CPU: video I/O, grayscale frame diff, mask, morphology, contours, annotations
- GPU: RAFT optical flow at auto-selected resolution (fits VRAM), upscaled back to full frame
- Falls back to CPU Farneback if CUDA is unavailable
"""

import argparse
import csv
import gc
import os
import subprocess
import time
from datetime import datetime

import cv2
import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights, raft_large, raft_small

try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

_REF_WIDTH = 2592
_REF_HEIGHT = 1944
_REF_VRAM_GB = 23.0


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    torch.backends.cudnn.benchmark = True
    return torch.device("cuda")


def _scaled_flow_size(width: int, height: int, scale: float) -> tuple[int, int]:
    flow_w = max(8, int(width * scale) // 8 * 8)
    flow_h = max(8, int(height * scale) // 8 * 8)
    return flow_w, flow_h


def _estimate_vram_gb(width: int, height: int, flow_w: int, flow_h: int) -> float:
    pixel_ratio = (width * height) / (_REF_WIDTH * _REF_HEIGHT)
    scale_ratio = (flow_w * flow_h) / (width * height)
    return _REF_VRAM_GB * pixel_ratio * (scale_ratio ** 2)


def pick_flow_size(width: int, height: int, device: torch.device, safety: float = 0.65) -> tuple[int, int]:
    if device.type != "cuda":
        return width, height

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    free_gb = free_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)
    print(f"GPU memory free: {free_gb:.1f} GB / {total_gb:.1f} GB")

    if free_gb < 2.0:
        print("Warning: little GPU memory free — close SAM/Jupyter before running.")

    for scale in (1.0, 0.75, 0.5, 0.375, 0.25):
        flow_w, flow_h = _scaled_flow_size(width, height, scale)
        needed_gb = _estimate_vram_gb(width, height, flow_w, flow_h)
        if needed_gb <= free_gb * safety:
            print(
                f"GPU flow size: {flow_w}x{flow_h} "
                f"(~{flow_w / width:.0%} of frame, est. VRAM {needed_gb:.1f} GB)"
            )
            return flow_w, flow_h

    flow_w, flow_h = _scaled_flow_size(width, height, 0.25)
    print(f"GPU flow size (minimum): {flow_w}x{flow_h}")
    return flow_w, flow_h


class ResourceLogger:
    def __init__(self, log_dir: str, run_name: str, cuda_device_index: int = 0):
        os.makedirs(log_dir, exist_ok=True)
        self.per_frame_path = os.path.join(log_dir, f"{run_name}_resources.csv")
        self.summary_path = os.path.join(log_dir, f"{run_name}_summary.log")
        self.process = psutil.Process()
        self.cuda_device_index = cuda_device_index
        self.gpu_handle = None
        self.rows = []
        psutil.cpu_percent(interval=None)

        if PYNVML_AVAILABLE and torch.cuda.is_available():
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(cuda_device_index)

    def sample(self) -> dict:
        stats = {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "system_cpu_percent": psutil.cpu_percent(interval=None),
            "ram_mb": self.process.memory_info().rss / (1024 ** 2),
            "gpu_util_percent": None,
            "gpu_mem_used_mb": None,
            "gpu_mem_total_mb": None,
        }
        if self.gpu_handle is not None:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            stats["gpu_util_percent"] = float(util.gpu)
            stats["gpu_mem_used_mb"] = mem.used / (1024 ** 2)
            stats["gpu_mem_total_mb"] = mem.total / (1024 ** 2)
        return stats

    def log_frame(self, frame_idx, time_sec, gpu_flow_ms, cpu_ms, frame_ms, detections):
        stats = self.sample()
        self.rows.append({
            "frame": frame_idx,
            "time_sec": round(time_sec, 4),
            "gpu_flow_ms": round(gpu_flow_ms, 2),
            "cpu_ms": round(cpu_ms, 2),
            "frame_ms": round(frame_ms, 2),
            "detections": detections,
            **stats,
        })
        gpu_str = (
            f"{stats['gpu_util_percent']:.0f}%"
            if stats["gpu_util_percent"] is not None else "n/a"
        )
        print(
            f"Frame {frame_idx}: GPU flow {gpu_flow_ms:.0f} ms | CPU {cpu_ms:.0f} ms | "
            f"CPU load {stats['cpu_percent']:.1f}% | RAM {stats['ram_mb']:.0f} MB | GPU {gpu_str}"
        )

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
            f.write(
                f"Backend: {backend}\n"
                f"Input: {input_video}\n"
                f"Output: {output_video}\n"
                f"Frames: {total_frames}\n"
                f"Elapsed: {elapsed_sec:.1f} s\n"
                f"CSV: {self.per_frame_path}\n"
            )
        print(f"Saved resource log: {self.per_frame_path}")
        print(f"Saved summary log: {self.summary_path}")

    def close(self):
        if self.gpu_handle is not None and PYNVML_AVAILABLE:
            pynvml.nvmlShutdown()


def load_valid_region_mask(mask_path, width, height, threshold=127, invert=False) -> np.ndarray:
    """Return uint8 mask: 255 = valid (detect here), 0 = excluded.

    Default convention for mask.png: white = sky (valid), black = trees/ground (excluded).
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not load mask: {mask_path}")
    if mask.shape[1] != width or mask.shape[0] != height:
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    valid = (mask > threshold).astype(np.uint8) * 255
    if invert:
        valid = 255 - valid
    valid_pixels = int(np.count_nonzero(valid))
    print(f"Mask: {mask_path}")
    print(f"Valid pixels (sky): {valid_pixels}/{valid.size} ({100 * valid_pixels / valid.size:.1f}%)")
    return valid


def cpu_farneback_flow(prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0,
    )
    return np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)


class HybridFlowEstimator:
    """GPU RAFT flow with CPU fallback and optional mask-aware tiling."""

    def __init__(self, model_name: str, device: torch.device, flow_w: int, flow_h: int, full_w: int, full_h: int):
        self.device = device
        self.flow_w = flow_w
        self.flow_h = flow_h
        self.full_w = full_w
        self.full_h = full_h
        self.use_gpu = device.type == "cuda"
        self.model = None
        self.transforms = None
        self.tiles_skipped_last = 0
        self.tiles_total_last = 0

        if self.use_gpu:
            if model_name == "raft_small":
                weights = Raft_Small_Weights.DEFAULT
                self.model = raft_small(weights=weights, progress=False)
            else:
                weights = Raft_Large_Weights.DEFAULT
                self.model = raft_large(weights=weights, progress=False)
            self.model = self.model.to(device).eval()
            self.transforms = weights.transforms()
            print(f"Flow backend: GPU RAFT ({model_name})")
        else:
            print("Flow backend: CPU Farneback (CUDA unavailable)")

    RAFT_MIN_SIZE = 128

    @staticmethod
    def _flow_size_for_crop(crop_h: int, crop_w: int) -> tuple[int, int]:
        flow_h = max(HybridFlowEstimator.RAFT_MIN_SIZE, crop_h // 8 * 8)
        flow_w = max(HybridFlowEstimator.RAFT_MIN_SIZE, crop_w // 8 * 8)
        return flow_h, flow_w

    @torch.inference_mode()
    def _gpu_flow_tiles_batch(
        self,
        prev_crops: list[np.ndarray],
        curr_crops: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Run RAFT on a batch of BGR crops; return flow magnitudes at crop resolution."""
        if not prev_crops:
            return []

        metas = []
        prev_tensors = []
        curr_tensors = []

        for prev_bgr, curr_bgr in zip(prev_crops, curr_crops):
            crop_h, crop_w = prev_bgr.shape[:2]
            pad_h = max(0, self.RAFT_MIN_SIZE - crop_h)
            pad_w = max(0, self.RAFT_MIN_SIZE - crop_w)
            if pad_h > 0 or pad_w > 0:
                prev_bgr = cv2.copyMakeBorder(prev_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
                curr_bgr = cv2.copyMakeBorder(curr_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
            proc_h, proc_w = prev_bgr.shape[:2]
            prev_rgb = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2RGB)
            curr_rgb = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2RGB)
            prev_tensors.append(torch.from_numpy(prev_rgb).permute(2, 0, 1).float())
            curr_tensors.append(torch.from_numpy(curr_rgb).permute(2, 0, 1).float())
            metas.append((crop_h, crop_w, proc_h, proc_w))

        batch_max_h = max(t.shape[1] for t in prev_tensors)
        batch_max_w = max(t.shape[2] for t in prev_tensors)
        flow_h, flow_w = self._flow_size_for_crop(batch_max_h, batch_max_w)

        prev_batch = []
        curr_batch = []
        for prev_t, curr_t, meta in zip(prev_tensors, curr_tensors, metas):
            _, _, proc_h, proc_w = meta
            if proc_h != batch_max_h or proc_w != batch_max_w:
                prev_t = F.pad(
                    prev_t.unsqueeze(0),
                    (0, batch_max_w - proc_w, 0, batch_max_h - proc_h),
                    mode="replicate",
                ).squeeze(0)
                curr_t = F.pad(
                    curr_t.unsqueeze(0),
                    (0, batch_max_w - proc_w, 0, batch_max_h - proc_h),
                    mode="replicate",
                ).squeeze(0)
            prev_batch.append(prev_t)
            curr_batch.append(curr_t)

        prev_batch = torch.stack(prev_batch).to(self.device)
        curr_batch = torch.stack(curr_batch).to(self.device)

        if flow_h != batch_max_h or flow_w != batch_max_w:
            prev_batch = F.interpolate(prev_batch, size=(flow_h, flow_w), mode="bilinear", align_corners=False)
            curr_batch = F.interpolate(curr_batch, size=(flow_h, flow_w), mode="bilinear", align_corners=False)

        img1, img2 = self.transforms(prev_batch, curr_batch)
        flow = self.model(img1, img2)[-1]

        results = []
        for b, (crop_h, crop_w, proc_h, proc_w) in enumerate(metas):
            flow_x = flow[b, 0].float() * (proc_w / flow_w)
            flow_y = flow[b, 1].float() * (proc_h / flow_h)
            flow_mag = torch.sqrt(flow_x ** 2 + flow_y ** 2)
            if flow_h != proc_h or flow_w != proc_w:
                flow_mag = F.interpolate(
                    flow_mag.unsqueeze(0).unsqueeze(0),
                    size=(proc_h, proc_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
            results.append(flow_mag[:crop_h, :crop_w].cpu().numpy())

        del prev_batch, curr_batch, img1, img2, flow
        return results

    @torch.inference_mode()
    def _gpu_flow_tile(self, prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> np.ndarray:
        """Run RAFT on a single BGR crop."""
        return self._gpu_flow_tiles_batch([prev_bgr], [curr_bgr])[0]

    @torch.inference_mode()
    def _gpu_flow_full(self, prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> np.ndarray:
        prev_rgb = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2RGB)
        curr_rgb = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2RGB)

        prev_t = torch.from_numpy(prev_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        curr_t = torch.from_numpy(curr_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        if self.flow_w != self.full_w or self.flow_h != self.full_h:
            prev_t = F.interpolate(prev_t, size=(self.flow_h, self.flow_w), mode="bilinear", align_corners=False)
            curr_t = F.interpolate(curr_t, size=(self.flow_h, self.flow_w), mode="bilinear", align_corners=False)

        img1, img2 = self.transforms(prev_t, curr_t)
        flow = self.model(img1, img2)[-1]

        flow_x = flow[0, 0].float() * (self.full_w / self.flow_w)
        flow_y = flow[0, 1].float() * (self.full_h / self.flow_h)
        flow_mag = torch.sqrt(flow_x ** 2 + flow_y ** 2)

        if self.flow_w != self.full_w or self.flow_h != self.full_h:
            flow_mag = F.interpolate(
                flow_mag.unsqueeze(0).unsqueeze(0),
                size=(self.full_h, self.full_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

        result = flow_mag.cpu().numpy()
        del prev_t, curr_t, img1, img2, flow, flow_mag
        return result

    def _gpu_flow_masked_tiles(
        self,
        prev_bgr: np.ndarray,
        curr_bgr: np.ndarray,
        valid_mask: np.ndarray,
        tile_size: int = 256,
        min_valid_frac: float = 0.05,
        flow_pad: int = 32,
        tile_batch: int = 6,
    ) -> np.ndarray:
        h, w = valid_mask.shape
        flow_mag = np.zeros((h, w), dtype=np.float32)
        tiles_skipped = 0
        tiles_total = 0
        pending_prev = []
        pending_curr = []
        pending_meta = []

        def flush_batch():
            if not pending_prev:
                return
            for tile_flow, (y, x, y2, x2, y0, x0) in zip(
                self._gpu_flow_tiles_batch(pending_prev, pending_curr),
                pending_meta,
            ):
                sy, sx = y - y0, x - x0
                flow_mag[y:y2, x:x2] = tile_flow[sy:sy + (y2 - y), sx:sx + (x2 - x)]
            pending_prev.clear()
            pending_curr.clear()
            pending_meta.clear()

        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                tiles_total += 1
                y2 = min(y + tile_size, h)
                x2 = min(x + tile_size, w)
                if (valid_mask[y:y2, x:x2] > 0).mean() < min_valid_frac:
                    tiles_skipped += 1
                    continue

                y0 = max(0, y - flow_pad)
                x0 = max(0, x - flow_pad)
                y1 = min(h, y2 + flow_pad)
                x1 = min(w, x2 + flow_pad)

                pending_prev.append(prev_bgr[y0:y1, x0:x1].copy())
                pending_curr.append(curr_bgr[y0:y1, x0:x1].copy())
                pending_meta.append((y, x, y2, x2, y0, x0))
                if len(pending_prev) >= tile_batch:
                    flush_batch()

        flush_batch()

        self.tiles_skipped_last = tiles_skipped
        self.tiles_total_last = tiles_total
        return flow_mag

    def compute(
        self,
        prev_bgr: np.ndarray,
        curr_bgr: np.ndarray,
        prev_gray: np.ndarray,
        gray: np.ndarray,
        valid_mask: np.ndarray | None = None,
        mask_before_flow: bool = True,
        tile_size: int = 256,
        tile_batch: int = 6,
    ) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        if self.use_gpu:
            if valid_mask is not None and mask_before_flow:
                flow_mag = self._gpu_flow_masked_tiles(
                    prev_bgr, curr_bgr, valid_mask,
                    tile_size=tile_size, tile_batch=tile_batch,
                )
            else:
                flow_mag = self._gpu_flow_full(prev_bgr, curr_bgr)
        else:
            if valid_mask is not None and mask_before_flow:
                flow_mag = np.zeros(prev_gray.shape, dtype=np.float32)
                tiles_skipped = 0
                tiles_total = 0
                h, w = prev_gray.shape
                for y in range(0, h, tile_size):
                    for x in range(0, w, tile_size):
                        tiles_total += 1
                        y2 = min(y + tile_size, h)
                        x2 = min(x + tile_size, w)
                        if (valid_mask[y:y2, x:x2] > 0).mean() < 0.05:
                            tiles_skipped += 1
                            continue
                        y0 = max(0, y - 16)
                        x0 = max(0, x - 16)
                        y1 = min(h, y2 + 16)
                        x1 = min(w, x2 + 16)
                        tile_flow = cpu_farneback_flow(prev_gray[y0:y1, x0:x1], gray[y0:y1, x0:x1])
                        sy, sx = y - y0, x - x0
                        flow_mag[y:y2, x:x2] = tile_flow[sy:sy + (y2 - y), sx:sx + (x2 - x)]
                self.tiles_skipped_last = tiles_skipped
                self.tiles_total_last = tiles_total
            else:
                flow_mag = cpu_farneback_flow(prev_gray, gray)
        return flow_mag, (time.perf_counter() - t0) * 1000


def remux_mp4_faststart(video_path: str) -> None:
    """Rewrite MP4 with moov at start so partial crashes are less likely to lose the file."""
    tmp_path = video_path + ".faststart.tmp.mp4"
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", video_path,
            "-c", "copy",
            "-movflags", "+faststart",
            tmp_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and os.path.isfile(tmp_path):
        os.replace(tmp_path, video_path)
    elif os.path.isfile(tmp_path):
        os.remove(tmp_path)


def process_video(
    input_video,
    output_video,
    output_csv,
    log_dir,
    run_name,
    min_area=3,
    max_area=5000,
    diff_threshold=20,
    flow_threshold=0.5,
    blur_size=5,
    flow_model="raft_small",
    flow_scale=0.0,
    mask_path=None,
    mask_threshold=127,
    mask_invert=False,
    log_interval=50,
    mask_before_flow=True,
    tile_size=256,
    tile_batch=6,
    process_scale=1.0,
):
    torch_device = resolve_device()
    cuda_index = 0
    resource_logger = ResourceLogger(log_dir, run_name, cuda_device_index=cuda_index)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        resource_logger.close()
        raise RuntimeError(f"Could not open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if flow_scale <= 0 and not (mask_path and mask_before_flow):
        flow_w, flow_h = pick_flow_size(width, height, torch_device)
    elif flow_scale <= 0:
        # With mask-before-flow, RAFT runs on small tiles — full-frame scale not needed
        flow_w, flow_h = width, height
        print("GPU flow: tile-based (mask-before-flow) — RAFT runs per-tile, not full frame")
    else:
        flow_w, flow_h = _scaled_flow_size(width, height, flow_scale)

    proc_w, proc_h = width, height
    if process_scale != 1.0:
        proc_w = max(8, int(width * process_scale))
        proc_h = max(8, int(height * process_scale))
        print(f"Process scale: {process_scale} -> internal {proc_w}x{proc_h}")

    flow_estimator = HybridFlowEstimator(flow_model, torch_device, flow_w, flow_h, proc_w, proc_h)
    backend = f"hybrid GPU RAFT + CPU OpenCV ({proc_w}x{proc_h}, mask-before-flow={mask_before_flow})"

    print(f"Input: {input_video}")
    print(f"Resolution: {width}x{height} @ {fps:.1f} fps")
    if torch_device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(torch_device)}")
    print(f"Backend: {backend}")
    print(f"Output video: {output_video}")
    print(f"Output CSV: {output_csv}")

    valid_region_mask = None
    if mask_path:
        valid_region_mask = load_valid_region_mask(mask_path, width, height, mask_threshold, mask_invert)
        if mask_before_flow:
            print(f"Mask-before-flow: ON (tile {tile_size}, batch {tile_batch}, skips empty masked tiles)")
        if process_scale != 1.0:
            valid_region_mask = cv2.resize(
                valid_region_mask, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST
            )

    os.makedirs(os.path.dirname(os.path.abspath(output_video)), exist_ok=True)
    writer = cv2.VideoWriter(
        output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        resource_logger.close()
        cap.release()
        raise RuntimeError(f"Could not open output video writer: {output_video}")

    ret, prev_frame = cap.read()
    if not ret:
        resource_logger.close()
        cap.release()
        writer.release()
        raise RuntimeError("Could not read first frame")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    if process_scale != 1.0:
        prev_frame = cv2.resize(prev_frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    if blur_size > 0:
        prev_gray = cv2.GaussianBlur(prev_gray, (blur_size, blur_size), 0)

    kernel = np.ones((3, 3), np.uint8)
    frame_idx = 0
    run_start = time.perf_counter()

    try:
        with open(output_csv, "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                "frame", "time_sec", "object_id", "x", "y", "w", "h",
                "area", "cx", "cy", "mean_flow_mag",
            ])

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                time_sec = frame_idx / fps
                frame_start = time.perf_counter()
                cpu_start = time.perf_counter()

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                proc_frame = frame
                if process_scale != 1.0:
                    proc_frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
                    gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
                if blur_size > 0:
                    gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

                frame_diff = cv2.absdiff(prev_gray, gray)
                _, diff_mask = cv2.threshold(frame_diff, diff_threshold, 255, cv2.THRESH_BINARY)

                cpu_before_flow_ms = (time.perf_counter() - cpu_start) * 1000

                if torch_device.type == "cuda":
                    torch.cuda.synchronize()
                flow_mag, gpu_flow_ms = flow_estimator.compute(
                    prev_frame,
                    proc_frame,
                    prev_gray,
                    gray,
                    valid_mask=valid_region_mask,
                    mask_before_flow=mask_before_flow and valid_region_mask is not None,
                    tile_size=tile_size,
                    tile_batch=tile_batch,
                )
                if frame_idx == 1 and flow_estimator.tiles_total_last:
                    ts = flow_estimator.tiles_skipped_last
                    tt = flow_estimator.tiles_total_last
                    print(f"GPU tiles skipped: {ts}/{tt} ({100 * ts / tt:.0f}%)")
                if torch_device.type == "cuda":
                    torch.cuda.synchronize()

                cpu_start = time.perf_counter()
                if process_scale != 1.0:
                    flow_mag = cv2.resize(flow_mag, (width, height), interpolation=cv2.INTER_LINEAR)
                    diff_mask = cv2.resize(diff_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    valid_for_frame = (
                        cv2.resize(valid_region_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                        if valid_region_mask is not None else None
                    )
                else:
                    valid_for_frame = valid_region_mask
                _, flow_mask = cv2.threshold(flow_mag.astype(np.float32), flow_threshold, 255, cv2.THRESH_BINARY)
                flow_mask = flow_mask.astype(np.uint8)

                motion_mask = cv2.bitwise_and(diff_mask, flow_mask)
                if valid_for_frame is not None:
                    motion_mask = cv2.bitwise_and(motion_mask, valid_for_frame)
                motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
                motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)
                if valid_for_frame is not None:
                    motion_mask = cv2.bitwise_and(motion_mask, valid_for_frame)

                contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                object_id = 0

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < min_area or area > max_area:
                        continue
                    x, y, w, h = cv2.boundingRect(cnt)
                    if valid_for_frame is not None:
                        cx_i = int(x + w / 2)
                        cy_i = int(y + h / 2)
                        if valid_for_frame[cy_i, cx_i] == 0:
                            continue
                    cx = x + w / 2
                    cy = y + h / 2
                    roi = flow_mag[y:y + h, x:x + w]
                    mean_flow = float(np.mean(roi)) if roi.size else 0.0
                    object_id += 1
                    csv_writer.writerow([
                        frame_idx, round(time_sec, 4), object_id,
                        x, y, w, h, round(area, 2), round(cx, 2), round(cy, 2), round(mean_flow, 4),
                    ])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)
                    cv2.putText(frame, f"ID:{object_id} A:{int(area)} F:{mean_flow:.2f}",
                                (x, max(20, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                cv2.putText(frame, f"Frame: {frame_idx}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                writer.write(frame)
                prev_frame = proc_frame
                prev_gray = gray.copy()

                cpu_ms = cpu_before_flow_ms + (time.perf_counter() - cpu_start) * 1000
                frame_ms = (time.perf_counter() - frame_start) * 1000

                if frame_idx % log_interval == 0 or frame_idx == max(frame_count - 1, 1):
                    resource_logger.log_frame(frame_idx, time_sec, gpu_flow_ms, cpu_ms, frame_ms, object_id)
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


def build_output_paths(script_dir, input_video, output_arg, csv_arg):
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(input_video))[0]
    return (
        output_arg or os.path.join(output_dir, f"{stem}_motion_hybrid.mp4"),
        csv_arg or os.path.join(output_dir, f"{stem}_detections_hybrid.csv"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid GPU+CPU optical flow motion detection")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--min-area", type=float, default=3)
    parser.add_argument("--max-area", type=float, default=5000)
    parser.add_argument("--diff-threshold", type=int, default=20)
    parser.add_argument("--flow-threshold", type=float, default=0.5)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--mask-threshold", type=int, default=127,
                        help="Pixels > threshold are valid (white=sky). Use --mask-invert for black=valid.")
    parser.add_argument("--mask-invert", action="store_true")
    parser.add_argument("--flow-model", default="raft_small", choices=["raft_small", "raft_large"])
    parser.add_argument("--flow-scale", type=float, default=0.0,
                        help="GPU flow scale (0=auto, fits VRAM)")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument(
        "--no-mask-before-flow",
        action="store_true",
        help="Run RAFT on full frame instead of mask-aware tiles",
    )
    parser.add_argument(
        "--process-scale",
        type=float,
        default=1.0,
        help="Internal processing scale (<1.0 is faster, e.g. 0.5)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=384,
        help="Tile size for mask-before-flow (default: 384, larger=fewer GPU calls)",
    )
    parser.add_argument(
        "--tile-batch",
        type=int,
        default=6,
        help="RAFT tiles per GPU batch (default: 6)",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    default_mask = os.path.join(script_dir, "mask", "mask.png")
    mask_path = args.mask if args.mask else (default_mask if os.path.isfile(default_mask) else None)

    stem = os.path.splitext(os.path.basename(args.input))[0]
    run_name = f"{stem}_hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_video, output_csv = build_output_paths(script_dir, args.input, args.output, args.csv)

    process_video(
        input_video=args.input,
        output_video=output_video,
        output_csv=output_csv,
        log_dir=log_dir,
        run_name=run_name,
        min_area=args.min_area,
        max_area=args.max_area,
        diff_threshold=args.diff_threshold,
        flow_threshold=args.flow_threshold,
        flow_model=args.flow_model,
        flow_scale=args.flow_scale,
        mask_path=mask_path,
        mask_threshold=args.mask_threshold,
        mask_invert=args.mask_invert,
        log_interval=args.log_interval,
        mask_before_flow=not args.no_mask_before_flow,
        tile_size=args.tile_size,
        tile_batch=args.tile_batch,
        process_scale=args.process_scale,
    )

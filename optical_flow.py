import argparse
import csv
import os
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


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available")
        torch.backends.cudnn.benchmark = True
    return device


# Empirical VRAM for RAFT-small correlation volume at reference resolution.
_REF_WIDTH = 2592
_REF_HEIGHT = 1944
_REF_VRAM_GB = 23.0


def _scaled_flow_size(width: int, height: int, scale: float) -> tuple[int, int]:
    flow_w = max(8, int(width * scale) // 8 * 8)
    flow_h = max(8, int(height * scale) // 8 * 8)
    return flow_w, flow_h


def _estimate_vram_gb(width: int, height: int, flow_w: int, flow_h: int) -> float:
    pixel_ratio = (width * height) / (_REF_WIDTH * _REF_HEIGHT)
    scale_ratio = (flow_w * flow_h) / (width * height)
    return _REF_VRAM_GB * pixel_ratio * (scale_ratio ** 2)


def pick_flow_scale(width: int, height: int, device: torch.device, safety: float = 0.65) -> tuple[int, int]:
    """Pick the largest flow resolution that fits in free GPU memory."""
    if device.type != "cuda":
        return width, height

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    free_gb = free_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)
    print(f"GPU memory free: {free_gb:.1f} GB / {total_gb:.1f} GB")

    if free_gb < 2.0:
        print(
            "Warning: very little GPU memory is free. "
            "Close SAM/Jupyter kernels or other GPU apps before running."
        )

    for scale in (1.0, 0.75, 0.5, 0.375, 0.25):
        flow_w, flow_h = _scaled_flow_size(width, height, scale)
        needed_gb = _estimate_vram_gb(width, height, flow_w, flow_h)
        if needed_gb <= free_gb * safety:
            print(
                f"Auto flow scale: {flow_w}x{flow_h} "
                f"(~{flow_w / width:.0%} of frame, est. VRAM {needed_gb:.1f} GB)"
            )
            return flow_w, flow_h

    flow_w, flow_h = _scaled_flow_size(width, height, 0.25)
    needed_gb = _estimate_vram_gb(width, height, flow_w, flow_h)
    print(
        f"Using minimum flow size: {flow_w}x{flow_h} (est. VRAM {needed_gb:.1f} GB)"
    )
    return flow_w, flow_h


class ResourceLogger:
    """Log CPU/GPU usage during optical flow processing."""

    def __init__(self, log_dir: str, run_name: str, cuda_device_index: int = 0):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.run_name = run_name
        self.cuda_device_index = cuda_device_index
        self.per_frame_path = os.path.join(log_dir, f"{run_name}_resources.csv")
        self.summary_path = os.path.join(log_dir, f"{run_name}_summary.log")
        self.process = psutil.Process()
        self.gpu_handle = None
        self.rows = []

        psutil.cpu_percent(interval=None)

        if PYNVML_AVAILABLE and torch.cuda.is_available():
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(cuda_device_index)

    def _sync_cuda(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def sample(self) -> dict:
        cpu_percent = psutil.cpu_percent(interval=None)
        ram_mb = self.process.memory_info().rss / (1024 ** 2)
        system_cpu_percent = psutil.cpu_percent(interval=None)

        gpu_util = None
        gpu_mem_used_mb = None
        gpu_mem_total_mb = None

        if self.gpu_handle is not None:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            gpu_util = float(util.gpu)
            gpu_mem_used_mb = mem.used / (1024 ** 2)
            gpu_mem_total_mb = mem.total / (1024 ** 2)
        elif torch.cuda.is_available():
            gpu_mem_used_mb = torch.cuda.memory_allocated(self.cuda_device_index) / (1024 ** 2)
            gpu_mem_total_mb = torch.cuda.get_device_properties(self.cuda_device_index).total_memory / (1024 ** 2)

        return {
            "cpu_percent": cpu_percent,
            "system_cpu_percent": system_cpu_percent,
            "ram_mb": ram_mb,
            "gpu_util_percent": gpu_util,
            "gpu_mem_used_mb": gpu_mem_used_mb,
            "gpu_mem_total_mb": gpu_mem_total_mb,
        }

    def log_frame(
        self,
        frame_idx: int,
        time_sec: float,
        flow_ms: float,
        frame_ms: float,
        detections: int,
    ):
        stats = self.sample()
        row = {
            "frame": frame_idx,
            "time_sec": round(time_sec, 4),
            "flow_ms": round(flow_ms, 2),
            "frame_ms": round(frame_ms, 2),
            "detections": detections,
            **stats,
        }
        self.rows.append(row)

        gpu_str = f"{stats['gpu_util_percent']:.0f}%" if stats["gpu_util_percent"] is not None else "n/a"
        gpu_mem_str = (
            f"{stats['gpu_mem_used_mb']:.0f}/{stats['gpu_mem_total_mb']:.0f} MB"
            if stats["gpu_mem_used_mb"] is not None and stats["gpu_mem_total_mb"] is not None
            else "n/a"
        )
        print(
            f"Frame {frame_idx}: flow {flow_ms:.0f} ms | "
            f"CPU {stats['cpu_percent']:.1f}% (system {stats['system_cpu_percent']:.1f}%) | "
            f"RAM {stats['ram_mb']:.0f} MB | GPU {gpu_str} | GPU mem {gpu_mem_str}"
        )

    def write_logs(self, total_frames: int, elapsed_sec: float, input_video: str, output_video: str):
        fieldnames = [
            "frame",
            "time_sec",
            "flow_ms",
            "frame_ms",
            "detections",
            "cpu_percent",
            "system_cpu_percent",
            "ram_mb",
            "gpu_util_percent",
            "gpu_mem_used_mb",
            "gpu_mem_total_mb",
        ]

        with open(self.per_frame_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({k: ("" if row[k] is None else row[k]) for k in fieldnames})

        def _avg(key):
            values = [r[key] for r in self.rows if r[key] is not None]
            return sum(values) / len(values) if values else None

        def _max(key):
            values = [r[key] for r in self.rows if r[key] is not None]
            return max(values) if values else None

        summary_lines = [
            f"Run: {self.run_name}",
            f"Timestamp: {datetime.now().isoformat(timespec='seconds')}",
            f"Input: {input_video}",
            f"Output: {output_video}",
            f"Frames processed: {total_frames}",
            f"Total elapsed: {elapsed_sec:.1f} s",
            f"Avg frame time: {1000 * elapsed_sec / total_frames:.1f} ms" if total_frames else "Avg frame time: n/a",
            "",
            "Per-frame resource averages:",
            f"  CPU (process): {_avg('cpu_percent'):.1f}%" if _avg("cpu_percent") is not None else "  CPU (process): n/a",
            f"  CPU (system):  {_avg('system_cpu_percent'):.1f}%" if _avg("system_cpu_percent") is not None else "  CPU (system): n/a",
            f"  RAM:           {_avg('ram_mb'):.0f} MB" if _avg("ram_mb") is not None else "  RAM: n/a",
            f"  GPU util:      {_avg('gpu_util_percent'):.1f}%" if _avg("gpu_util_percent") is not None else "  GPU util: n/a",
            f"  GPU mem used:  {_avg('gpu_mem_used_mb'):.0f} MB" if _avg("gpu_mem_used_mb") is not None else "  GPU mem used: n/a",
            "",
            "Peaks:",
            f"  CPU (process): {_max('cpu_percent'):.1f}%" if _max("cpu_percent") is not None else "  CPU (process): n/a",
            f"  CPU (system):  {_max('system_cpu_percent'):.1f}%" if _max("system_cpu_percent") is not None else "  CPU (system): n/a",
            f"  GPU util:      {_max('gpu_util_percent'):.1f}%" if _max("gpu_util_percent") is not None else "  GPU util: n/a",
            f"  GPU mem used:  {_max('gpu_mem_used_mb'):.0f} MB" if _max("gpu_mem_used_mb") is not None else "  GPU mem used: n/a",
            f"  Flow time:     {_max('flow_ms'):.0f} ms" if _max("flow_ms") is not None else "  Flow time: n/a",
            "",
            f"Per-frame CSV: {self.per_frame_path}",
        ]

        with open(self.summary_path, "w") as f:
            f.write("\n".join(summary_lines) + "\n")

        print(f"Saved resource log: {self.per_frame_path}")
        print(f"Saved summary log: {self.summary_path}")

    def close(self):
        if self.gpu_handle is not None and PYNVML_AVAILABLE:
            pynvml.nvmlShutdown()


def load_flow_model(model_name: str, device: torch.device):
    if model_name == "raft_small":
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights, progress=False)
    elif model_name == "raft_large":
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights, progress=False)
    else:
        raise ValueError(f"Unknown flow model: {model_name}")

    model = model.to(device).eval()
    transforms = weights.transforms()
    return model, transforms


def gaussian_blur_gpu(gray: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    if kernel_size <= 0:
        return gray

    coords = torch.arange(kernel_size, device=gray.device, dtype=torch.float32) - kernel_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

    pad = kernel_size // 2
    padded = F.pad(gray, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(padded, kernel_2d)


def morph_open_gpu(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    pad = kernel_size // 2
    inverted = 1.0 - mask
    eroded = 1.0 - F.max_pool2d(
        F.pad(inverted, (pad, pad, pad, pad), mode="constant", value=1.0),
        kernel_size,
        stride=1,
    )
    return F.max_pool2d(
        F.pad(eroded, (pad, pad, pad, pad), mode="constant", value=0.0),
        kernel_size,
        stride=1,
    )


def morph_dilate_gpu(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    pad = kernel_size // 2
    return F.max_pool2d(
        F.pad(mask, (pad, pad, pad, pad), mode="constant", value=0.0),
        kernel_size,
        stride=1,
    )


def bgr_to_gray_tensor(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return torch.from_numpy(gray).to(device).float().unsqueeze(0).unsqueeze(0) / 255.0


def bgr_to_rgb_tensor(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)


def load_valid_region_mask(
    mask_path: str,
    width: int,
    height: int,
    device: torch.device,
    threshold: int = 127,
    invert: bool = False,
) -> torch.Tensor:
    """Load mask PNG; return 1.0 for pixels to process (non-masked), 0.0 for excluded."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not load mask: {mask_path}")

    if mask.shape[1] != width or mask.shape[0] != height:
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    valid = (mask > threshold).astype(np.float32)
    if invert:
        valid = 1.0 - valid

    valid_pixels = int(valid.sum())
    total_pixels = valid.size
    print(f"Mask: {mask_path}")
    print(f"Valid (non-masked) pixels: {valid_pixels}/{total_pixels} ({100 * valid_pixels / total_pixels:.1f}%)")

    return torch.from_numpy(valid).to(device).unsqueeze(0).unsqueeze(0)


@torch.inference_mode()
def compute_flow_magnitude(
    model,
    transforms,
    prev_rgb: torch.Tensor,
    curr_rgb: torch.Tensor,
    flow_w: int,
    flow_h: int,
) -> torch.Tensor:
    orig_h, orig_w = prev_rgb.shape[-2:]

    if flow_w != orig_w or flow_h != orig_h:
        prev_rgb = F.interpolate(
            prev_rgb,
            size=(flow_h, flow_w),
            mode="bilinear",
            align_corners=False,
        )
        curr_rgb = F.interpolate(
            curr_rgb,
            size=(flow_h, flow_w),
            mode="bilinear",
            align_corners=False,
        )

    img1, img2 = transforms(prev_rgb, curr_rgb)
    flow = model(img1, img2)[-1]
    flow_x = flow[:, 0] * (orig_w / flow_w)
    flow_y = flow[:, 1] * (orig_h / flow_h)

    flow_mag = torch.sqrt(flow_x ** 2 + flow_y ** 2).unsqueeze(1)

    if flow_w != orig_w or flow_h != orig_h:
        flow_mag = F.interpolate(
            flow_mag,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )

    return flow_mag


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
    device="cuda",
    flow_model="raft_small",
    flow_scale=0.0,
    mask_path=None,
    mask_threshold=127,
    mask_invert=False,
    log_interval=1,
):
    torch_device = resolve_device(device)
    cuda_index = torch_device.index if torch_device.index is not None else 0
    flow_net, flow_transforms = load_flow_model(flow_model, torch_device)
    resource_logger = ResourceLogger(log_dir, run_name, cuda_device_index=cuda_index)

    os.makedirs(os.path.dirname(os.path.abspath(output_video)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        resource_logger.close()
        raise RuntimeError(f"Could not open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input video: {input_video}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Device: {torch_device}")
    if torch_device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(torch_device)}")
    print(f"Flow model: {flow_model}")

    if flow_scale <= 0:
        flow_w, flow_h = pick_flow_scale(width, height, torch_device)
    else:
        flow_w, flow_h = _scaled_flow_size(width, height, flow_scale)
        needed_gb = _estimate_vram_gb(width, height, flow_w, flow_h)
        print(f"Flow compute size: {flow_w}x{flow_h} (est. VRAM {needed_gb:.1f} GB)")

    print(f"Output video: {output_video}")
    print(f"Output CSV: {output_csv}")
    print(f"Logs: {log_dir}")

    valid_region_mask = None
    if mask_path:
        valid_region_mask = load_valid_region_mask(
            mask_path, width, height, torch_device, mask_threshold, mask_invert
        )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    ret, prev_frame = cap.read()
    if not ret:
        resource_logger.close()
        raise RuntimeError("Could not read first frame")

    prev_gray = bgr_to_gray_tensor(prev_frame, torch_device)
    prev_rgb = bgr_to_rgb_tensor(prev_frame, torch_device)

    if blur_size > 0:
        sigma = 0.3 * ((blur_size - 1) * 0.5 - 1) + 0.8
        prev_gray = gaussian_blur_gpu(prev_gray, blur_size, sigma)

    diff_threshold_norm = diff_threshold / 255.0
    frame_idx = 0
    run_start = time.perf_counter()

    with open(output_csv, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            "frame",
            "time_sec",
            "object_id",
            "x",
            "y",
            "w",
            "h",
            "area",
            "cx",
            "cy",
            "mean_flow_mag",
        ])

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            time_sec = frame_idx / fps
            frame_start = time.perf_counter()

            gray = bgr_to_gray_tensor(frame, torch_device)
            curr_rgb = bgr_to_rgb_tensor(frame, torch_device)

            if blur_size > 0:
                gray = gaussian_blur_gpu(gray, blur_size, sigma)

            frame_diff = torch.abs(prev_gray - gray)
            diff_mask = (frame_diff > diff_threshold_norm).float()

            if torch_device.type == "cuda":
                torch.cuda.synchronize()
            flow_start = time.perf_counter()

            flow_mag = compute_flow_magnitude(
                flow_net,
                flow_transforms,
                prev_rgb,
                curr_rgb,
                flow_w,
                flow_h,
            )

            if torch_device.type == "cuda":
                torch.cuda.synchronize()
            flow_ms = (time.perf_counter() - flow_start) * 1000

            flow_mask = (flow_mag > flow_threshold).float()

            motion_mask = diff_mask * flow_mask
            if valid_region_mask is not None:
                motion_mask = motion_mask * valid_region_mask
            motion_mask = morph_open_gpu(motion_mask)
            motion_mask = morph_dilate_gpu(motion_mask)
            if valid_region_mask is not None:
                motion_mask = motion_mask * valid_region_mask

            motion_mask_cpu = (motion_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
            flow_mag_cpu = flow_mag.squeeze().cpu().numpy()

            contours, _ = cv2.findContours(
                motion_mask_cpu,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            object_id = 0

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                if valid_region_mask is not None:
                    cx_i = int(x + w / 2)
                    cy_i = int(y + h / 2)
                    if not valid_region_mask[0, 0, cy_i, cx_i]:
                        continue
                cx = x + w / 2
                cy = y + h / 2

                roi_flow = flow_mag_cpu[y:y + h, x:x + w]
                mean_flow_mag = float(np.mean(roi_flow)) if roi_flow.size > 0 else 0.0

                object_id += 1
                csv_writer.writerow([
                    frame_idx,
                    round(time_sec, 4),
                    object_id,
                    x,
                    y,
                    w,
                    h,
                    round(area, 2),
                    round(cx, 2),
                    round(cy, 2),
                    round(mean_flow_mag, 4),
                ])

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)

                label = f"ID:{object_id} A:{int(area)} F:{mean_flow_mag:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x, max(20, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                )

            cv2.putText(
                frame,
                f"Frame: {frame_idx}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            writer.write(frame)
            prev_gray = gray
            prev_rgb = curr_rgb

            frame_ms = (time.perf_counter() - frame_start) * 1000
            if frame_idx % log_interval == 0 or frame_idx == frame_count - 1:
                resource_logger.log_frame(frame_idx, time_sec, flow_ms, frame_ms, object_id)

    cap.release()
    writer.release()

    elapsed_sec = time.perf_counter() - run_start
    resource_logger.write_logs(frame_idx, elapsed_sec, input_video, output_video)
    resource_logger.close()

    print("Done.")
    print(f"Saved output video: {output_video}")
    print(f"Saved detections CSV: {output_csv}")


def build_output_paths(script_dir: str, input_video: str, output_arg: str | None, csv_arg: str | None):
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(input_video))[0]
    output_video = output_arg or os.path.join(output_dir, f"{stem}_motion.mp4")
    output_csv = csv_arg or os.path.join(output_dir, f"{stem}_detections.csv")
    return output_video, output_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", default=None, help="Path to output video (default: output/<input>_motion.mp4)")
    parser.add_argument("--csv", default=None, help="Path to output CSV (default: output/<input>_detections.csv)")
    parser.add_argument("--min-area", type=float, default=3, help="Minimum contour area in pixels")
    parser.add_argument("--max-area", type=float, default=5000, help="Maximum contour area in pixels")
    parser.add_argument("--diff-threshold", type=int, default=20, help="Frame subtraction threshold")
    parser.add_argument("--flow-threshold", type=float, default=0.5, help="Optical flow magnitude threshold")
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["auto", "cuda", "cpu"],
        help="Compute device (default: cuda)",
    )
    parser.add_argument(
        "--mask",
        default=None,
        help="Path to mask PNG; bright pixels are excluded, dark pixels are processed",
    )
    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=127,
        help="Pixels > threshold are valid (white=sky). Use --mask-invert for black=valid.",
    )
    parser.add_argument(
        "--mask-invert",
        action="store_true",
        help="Invert mask: bright pixels are valid instead of dark",
    )
    parser.add_argument(
        "--flow-model",
        default="raft_small",
        choices=["raft_small", "raft_large"],
        help="RAFT model variant (default: raft_small)",
    )
    parser.add_argument(
        "--flow-scale",
        type=float,
        default=0.0,
        help="Scale factor for flow (0 = auto based on GPU memory, default: 0)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Log CPU/GPU stats every N frames (default: 50)",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    default_mask = os.path.join(script_dir, "mask", "mask.png")
    mask_path = args.mask
    if mask_path is None and os.path.isfile(default_mask):
        mask_path = default_mask

    input_stem = os.path.splitext(os.path.basename(args.input))[0]
    run_name = f"{input_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        device=args.device,
        flow_model=args.flow_model,
        flow_scale=args.flow_scale,
        mask_path=mask_path,
        mask_threshold=args.mask_threshold,
        mask_invert=args.mask_invert,
        log_interval=args.log_interval,
    )

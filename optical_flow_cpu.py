import argparse
import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np
import psutil


class ResourceLogger:
    """Log CPU/RAM usage during optical flow processing."""

    def __init__(self, log_dir: str, run_name: str):
        os.makedirs(log_dir, exist_ok=True)
        self.per_frame_path = os.path.join(log_dir, f"{run_name}_resources.csv")
        self.summary_path = os.path.join(log_dir, f"{run_name}_summary.log")
        self.process = psutil.Process()
        self.rows = []
        psutil.cpu_percent(interval=None)

    def sample(self) -> dict:
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "system_cpu_percent": psutil.cpu_percent(interval=None),
            "ram_mb": self.process.memory_info().rss / (1024 ** 2),
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
        print(
            f"Frame {frame_idx}: flow {flow_ms:.0f} ms | "
            f"CPU {stats['cpu_percent']:.1f}% (system {stats['system_cpu_percent']:.1f}%) | "
            f"RAM {stats['ram_mb']:.0f} MB"
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
        ]

        with open(self.per_frame_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

        def _avg(key):
            values = [r[key] for r in self.rows]
            return sum(values) / len(values) if values else None

        def _max(key):
            return max(r[key] for r in self.rows) if self.rows else None

        summary_lines = [
            f"Run: {os.path.basename(self.summary_path).replace('_summary.log', '')}",
            f"Backend: CPU (OpenCV Farneback)",
            f"Timestamp: {datetime.now().isoformat(timespec='seconds')}",
            f"Input: {input_video}",
            f"Output: {output_video}",
            f"Frames processed: {total_frames}",
            f"Total elapsed: {elapsed_sec:.1f} s",
            f"Avg frame time: {1000 * elapsed_sec / total_frames:.1f} ms" if total_frames else "Avg frame time: n/a",
            "",
            "Per-frame resource averages:",
            f"  CPU (process): {_avg('cpu_percent'):.1f}%",
            f"  CPU (system):  {_avg('system_cpu_percent'):.1f}%",
            f"  RAM:           {_avg('ram_mb'):.0f} MB",
            "",
            "Peaks:",
            f"  CPU (process): {_max('cpu_percent'):.1f}%",
            f"  CPU (system):  {_max('system_cpu_percent'):.1f}%",
            f"  RAM:           {_max('ram_mb'):.0f} MB",
            f"  Flow time:     {_max('flow_ms'):.0f} ms",
            "",
            f"Per-frame CSV: {self.per_frame_path}",
        ]

        with open(self.summary_path, "w") as f:
            f.write("\n".join(summary_lines) + "\n")

        print(f"Saved resource log: {self.per_frame_path}")
        print(f"Saved summary log: {self.summary_path}")


def load_valid_region_mask(
    mask_path: str,
    width: int,
    height: int,
    threshold: int = 127,
    invert: bool = False,
) -> np.ndarray:
    """Return uint8 mask: 255 = valid (detect here), 0 = excluded.

    Default convention: white = sky (valid), black = trees/ground (excluded).
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
    total_pixels = valid.size
    print(f"Mask: {mask_path}")
    print(f"Valid (non-masked) pixels: {valid_pixels}/{total_pixels} ({100 * valid_pixels / total_pixels:.1f}%)")
    return valid


def compute_flow_magnitude(prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]
    return np.sqrt(flow_x ** 2 + flow_y ** 2)


def compute_flow_magnitude_masked_tiles(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    valid_mask: np.ndarray,
    tile_size: int = 256,
    min_valid_frac: float = 0.05,
    flow_pad: int = 16,
) -> np.ndarray:
    """Run Farneback only on tiles that contain enough valid (non-masked) pixels."""
    h, w = prev_gray.shape
    flow_mag = np.zeros((h, w), dtype=np.float32)
    tiles_total = 0
    tiles_skipped = 0

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tiles_total += 1
            y2 = min(y + tile_size, h)
            x2 = min(x + tile_size, w)
            tile_valid = valid_mask[y:y2, x:x2] > 0
            if tile_valid.mean() < min_valid_frac:
                tiles_skipped += 1
                continue

            y0 = max(0, y - flow_pad)
            x0 = max(0, x - flow_pad)
            y1 = min(h, y2 + flow_pad)
            x1 = min(w, x2 + flow_pad)

            tile_flow = compute_flow_magnitude(prev_gray[y0:y1, x0:x1], gray[y0:y1, x0:x1])
            sy, sx = y - y0, x - x0
            flow_mag[y:y2, x:x2] = tile_flow[sy:sy + (y2 - y), sx:sx + (x2 - x)]

    return flow_mag, tiles_skipped, tiles_total


def resize_gray(gray: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return gray
    new_w = max(8, int(gray.shape[1] * scale))
    new_h = max(8, int(gray.shape[0] * scale))
    return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)


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
    mask_path=None,
    mask_threshold=127,
    mask_invert=False,
    log_interval=50,
    mask_before_flow=True,
    process_scale=1.0,
    tile_size=256,
):
    resource_logger = ResourceLogger(log_dir, run_name)

    os.makedirs(os.path.dirname(os.path.abspath(output_video)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
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
    print(f"Backend: CPU (OpenCV Farneback)")
    print(f"Output video: {output_video}")
    print(f"Output CSV: {output_csv}")
    print(f"Logs: {log_dir}")

    valid_region_mask = None
    proc_w, proc_h = width, height
    if mask_path:
        valid_region_mask = load_valid_region_mask(
            mask_path, width, height, mask_threshold, mask_invert
        )
        if mask_before_flow:
            print("Mask-before-flow: ON (tile-based — skips empty masked tiles)")

    if process_scale != 1.0:
        proc_w = max(8, int(width * process_scale))
        proc_h = max(8, int(height * process_scale))
        print(f"Process scale: {process_scale} -> internal {proc_w}x{proc_h}, output {width}x{height}")
        if valid_region_mask is not None:
            valid_region_mask = cv2.resize(
                valid_region_mask, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST
            )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = resize_gray(prev_gray, process_scale)
    if blur_size > 0:
        prev_gray = cv2.GaussianBlur(prev_gray, (blur_size, blur_size), 0)

    kernel = np.ones((3, 3), np.uint8)
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

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = resize_gray(gray, process_scale)
            if blur_size > 0:
                gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

            frame_diff = cv2.absdiff(prev_gray, gray)
            _, diff_mask = cv2.threshold(frame_diff, diff_threshold, 255, cv2.THRESH_BINARY)

            flow_start = time.perf_counter()
            if valid_region_mask is not None and mask_before_flow:
                flow_mag, tiles_skipped, tiles_total = compute_flow_magnitude_masked_tiles(
                    prev_gray, gray, valid_region_mask, tile_size=tile_size
                )
                if frame_idx == 1:
                    print(f"Tiles skipped: {tiles_skipped}/{tiles_total} ({100 * tiles_skipped / tiles_total:.0f}%)")
            else:
                flow_mag = compute_flow_magnitude(prev_gray, gray)
            flow_ms = (time.perf_counter() - flow_start) * 1000

            if process_scale != 1.0:
                flow_mag = cv2.resize(flow_mag, (width, height), interpolation=cv2.INTER_LINEAR)
                diff_mask = cv2.resize(diff_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                if valid_region_mask is not None:
                    valid_for_frame = cv2.resize(
                        valid_region_mask, (width, height), interpolation=cv2.INTER_NEAREST
                    )
                else:
                    valid_for_frame = None
            else:
                valid_for_frame = valid_region_mask

            _, flow_mask = cv2.threshold(
                flow_mag.astype(np.float32),
                flow_threshold,
                255,
                cv2.THRESH_BINARY,
            )
            flow_mask = flow_mask.astype(np.uint8)

            motion_mask = cv2.bitwise_and(diff_mask, flow_mask)
            if valid_for_frame is not None:
                motion_mask = cv2.bitwise_and(motion_mask, valid_for_frame)

            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)
            if valid_for_frame is not None:
                motion_mask = cv2.bitwise_and(motion_mask, valid_for_frame)

            contours, _ = cv2.findContours(
                motion_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

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
                roi_flow = flow_mag[y:y + h, x:x + w]
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
            prev_gray = gray.copy()

            frame_ms = (time.perf_counter() - frame_start) * 1000
            if frame_idx % log_interval == 0 or frame_idx == frame_count - 1:
                resource_logger.log_frame(frame_idx, time_sec, flow_ms, frame_ms, object_id)
                print(f"Processed frame {frame_idx}/{max(frame_count - 1, 0)}")

    cap.release()
    writer.release()

    elapsed_sec = time.perf_counter() - run_start
    resource_logger.write_logs(frame_idx, elapsed_sec, input_video, output_video)

    print("Done.")
    print(f"Saved output video: {output_video}")
    print(f"Saved detections CSV: {output_csv}")


def build_output_paths(script_dir: str, input_video: str, output_arg, csv_arg):
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(input_video))[0]
    output_video = output_arg or os.path.join(output_dir, f"{stem}_motion_cpu.mp4")
    output_csv = csv_arg or os.path.join(output_dir, f"{stem}_detections_cpu.csv")
    return output_video, output_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optical flow motion detection (CPU / OpenCV Farneback)")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", default=None, help="Path to output video (default: output/<input>_motion_cpu.mp4)")
    parser.add_argument("--csv", default=None, help="Path to output CSV (default: output/<input>_detections_cpu.csv)")
    parser.add_argument("--min-area", type=float, default=3, help="Minimum contour area in pixels")
    parser.add_argument("--max-area", type=float, default=5000, help="Maximum contour area in pixels")
    parser.add_argument("--diff-threshold", type=int, default=20, help="Frame subtraction threshold")
    parser.add_argument("--flow-threshold", type=float, default=0.5, help="Optical flow magnitude threshold")
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
        "--log-interval",
        type=int,
        default=50,
        help="Log CPU/RAM stats every N frames (default: 50)",
    )
    parser.add_argument(
        "--no-mask-before-flow",
        action="store_true",
        help="Disable tile-based flow skip; run Farneback on full frame",
    )
    parser.add_argument(
        "--process-scale",
        type=float,
        default=1.0,
        help="Process internally at this scale (<1.0 is faster, e.g. 0.5)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="Tile size for mask-before-flow (default: 256)",
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
    run_name = f"{input_stem}_cpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        mask_path=mask_path,
        mask_threshold=args.mask_threshold,
        mask_invert=args.mask_invert,
        log_interval=args.log_interval,
        mask_before_flow=not args.no_mask_before_flow,
        process_scale=args.process_scale,
        tile_size=args.tile_size,
    )

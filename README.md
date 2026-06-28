command line :
python3 optical_flow_nvof_bright_length.py \
  --input "/media/a/E/MyProjects/Water/Projects/yablunivka/f_2020-03-16_03-42-13_c.mkv" \
  --output-mode flow \
  --trail-length 30 \
  --bright-threshold 30 \
  --bright-min-frames 3 \
  --diff-threshold 20 \
  --flow-threshold 0.5 \
  --min-area 2 \
  --max-area 500000

--bright-min-frames 3
Object must appear in 3 consecutive frames before it's reported. Filters out single-frame noise and camera flicker. For daytime leave it at 3 — it's fine.

--bright-max-frames 0
0 means disabled — no upper limit. If you set it to e.g. 500, objects that have been bright for 500+ consecutive frames get suppressed (useful for suppressing stars at night that are always there). For daytime you don't need this — stars aren't visible so leave it at 0.

--diff-threshold 20
This is Path B only — the motion path. Pixel must change by 20 DN between frames to be considered moving. During daytime this is fine — sunlight reflections on water change by much more than 20 DN. You might actually want to raise it to 30-40 during day to reduce water ripple false detections on the motion path.

Share the output and I'll tell you exactly what to set --bright-threshold to. But the rough rule is:
Night (background ~10 DN) → --bright-threshold 30
Dusk/dawn (background ~80 DN) → --bright-threshold 120
Day (background ~150 DN) → --bright-threshold 180

# Person Tracking & Re-Identification

A Python project for tracking people in video and re-identifying them using YOLO, DeepSORT, and feature extraction.

## Features
- Track people in video and save their features
- Re-identify people in new videos using saved features
- Configurable via `config.yaml`
- Color-coded similarity visualization

## Setup

1. Clone this repo
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Download YOLO weights and place as specified in `config.yaml`
4. Edit `config.yaml` to set your video paths and parameters

## Requirements

- Python 3.8+
- OpenCV
- torch
- torchvision
- ultralytics
- deep_sort_realtime
- pyyaml
- scipy

## Project Structure

```
miniproject3/
├── main.py
├── tracking.py
├── reidentify.py
├── reid_utils.py
├── utils.py
├── config.yaml
├── requirements.txt
├── readme.md
├── demo_vid/
│   ├── 853889-hd_1920_1080_25fps.mp4
│   └── part2.mp4
├── features/
│   └── features.pkl
├── models/
│   └── model_16_m3_0.8888.pth
├── output/
└── assests/
    ├── Screenshot 2025-07-28 124923.png
    └── Screenshot 2025-07-28 124945.png
 
```

## Usage

- Run the main menu:
  ```
  python main.py
  ```
- Choose:
  - `1` for tracking and feature extraction
  - `2` for re-identification

### Controls

- Click a bounding box to select an ID
- Press `s` to save features for the selected ID
- Press `q` to quit video and return to menu

## Configuration

See `config.yaml` for all paths and parameters.

## Example Output

![screenshot](PROJECT\miniproject3\assests\Screenshot 2025-07-28 124923.png)
![screenshot](PROJECT\miniproject3\assests\Screenshot 2025-07-28 124945.png)

---


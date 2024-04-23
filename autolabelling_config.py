from pathlib import Path

ROOT = Path('D:/Data')

INPUT_FOLDER = ROOT / 'Gustavo' / 'Video_corto_6_07_a_6_12'
INPUT_VIDEO = 'VID_20240409_180957_00_003(1).mp4'

# Deep Learning model configuration
YOLOV8_FOLDER = ROOT / 'models' / 'yolov8'
YOLOV8_WEIGHTS = 'yolov8x_cf_v3'
YOLOV9_FOLDER = ROOT / 'models' / 'yolov9'
YOLOV9_WEIGHTS = 'yolov9e'

# Inference configuration
IMAGE_SIZE = 640
CONFIDENCE = 0.5
SAMPLE_NUMBER = 100
SHOW_IMAGE = True

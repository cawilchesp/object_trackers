from pathlib import Path

ROOT = Path('D:/Data')

# INPUT_FOLDER = ROOT
# INPUT_VIDEO = 'paradero.mp4'

# Deep Learning model configuration
YOLOV8_FOLDER = ROOT / 'models' / 'yolov8'
YOLOV8_WEIGHTS = 'yolov8'
YOLOV9_FOLDER = ROOT / 'models' / 'yolov9'
YOLOV9_WEIGHTS = 'yolov9c'
MODEL_FOLDER = ROOT / 'models' / 'yolov10'
MODEL_WEIGHTS = 'yolov10x.pt'


# Inference configuration
IMAGE_SIZE = 640
CONFIDENCE = 0.25
IOU = 0.7
CLASS_FILTER = [0,1,2,3,5,7]
TRACK_LENGTH = 100
SHOW_IMAGE = True
SAVE_CSV = True
SAVE_VIDEO = True

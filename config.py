from pathlib import Path

ROOT = Path('D:/Data')

# INPUT_FOLDER = ROOT / 'MARZO_2024' / 'ARS-C57-C65'
# INPUT_VIDEO = 'ARS-Iglesia Santa Maria Magdalena CAM3.mp4'

# INPUT_FOLDER = ROOT / 'MARZO_2024' / 'ARS-Iguana'
# INPUT_VIDEO = 'ARS-Iguana CAM2.mp4'

INPUT_FOLDER = ROOT / 'MARZO_2024' / 'ARS-SanPedro'
INPUT_VIDEO = 'ARS-Metroplus San Pedro CAM1.mp4'

# Deep Learning model configuration
YOLOV8_FOLDER = ROOT / 'models' / 'yolov8'
YOLOV8_WEIGHTS = 'yolov8x_cf_v3'
YOLOV9_FOLDER = ROOT / 'models' / 'yolov9'
YOLOV9_WEIGHTS = 'yolov9c_cf_v6'
RTDETR_FOLDER = ROOT / 'models' / 'rt-detr'
RTDETR_WEIGHTS = 'rtdetr-l'

# Inference configuration
IMAGE_SIZE = 640
CONFIDENCE = 0.5
CLASS_FILTER = [0,1,2,3,5,7]
TRACK_LENGTH = 100
SHOW_IMAGE = True
SAVE_CSV = True
SAVE_VIDEO = True

# Autolabelling
SAMPLE_NUMBER = 100
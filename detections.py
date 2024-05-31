from ultralytics import YOLO
import supervision as sv

import cv2
import yaml
import torch
import numpy as np
from pathlib import Path
import itertools

from imutils.video import WebcamVideoStream, FileVideoStream
from imutils.video import FPS

import config
from tools.print_info import print_video_info, print_progress, step_message
from tools.video_info import from_camera, from_video_path
from tools.csv_sink import CSVSink
from tools.timers import FPSBasedTimer, ClockBasedTimer
from tools.general import load_zones

# For debugging
from icecream import ic


def main(
    source: str,
    output: str,
    weights: str,
    class_filter: list[int],
    image_size: int,
    confidence: float,
    iou: float,
    show_image: bool,
    save_csv: bool,
    save_video: bool
) -> None:
    step_count = itertools.count(1)
    
    # Initialize model
    model = YOLO(weights)
    step_message(next(step_count), f'{Path(weights).stem.upper()} Model Initialized')

    # Inicializar captura de video
    video_source = eval(source) if source.isnumeric() else source
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened(): quit()
    if source.isnumeric() or source.lower().startswith('rtsp://'):
        source_info = from_camera(cap)
    else:
        source_info = from_video_path(cap)
    cap.release()
    step_message(next(step_count), 'Origen del Video Inicializado')
    print_video_info(source, source_info)

    if show_image:
        scaled_width = 1280 if source_info.width > 1280 else source_info.width
        scaled_height = int(scaled_width * source_info.height / source_info.width)
        scaled_height = scaled_height if source_info.height > scaled_height else source_info.height

    step_message(next(step_count), f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

    # Annotators
    fps_monitor = sv.FPSMonitor()

    # Start video processing
    step_message(next(step_count), 'Start Video Processing')
    if source.isnumeric() or source.lower().startswith('rtsp://'):
        fvs = WebcamVideoStream(video_source)
    else:
        fvs = FileVideoStream(source)
    
    video_sink = sv.VideoSink(target_path=f"{output}.mp4", video_info=source_info)
    csv_sink = CSVSink(file_name=f"{output}.csv")
    
    frame_number = 0
    fvs.start()
    fps = FPS().start()
    with csv_sink:
        while fvs.more(): # while True:
            fps_monitor.tick()
            fps_rt = fps_monitor.fps
            
            image = fvs.read()
            if image is None:
                print()
                break
            annotated_image = image.copy()
            
            # YOLOv8 inference
            results = model(
                source=image,
                imgsz=image_size,
                conf=confidence,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                verbose=False
            )[0]
            detections = sv.Detections.from_ultralytics(results)#.with_nms(threshold=iou)

            if save_csv: csv_sink.append(detections, custom_data={'frame_number': frame_number})

            print_progress(frame_number, source_info.total_frames)
            frame_number += 1
            
            fps.update()

    fps.stop()
    step_message(next(step_count), f"Elapsed Time: {fps.elapsed():.2f} s")
    step_message(next(step_count), f"FPS: {fps.fps():.2f}")

    cv2.destroyAllWindows()
    fvs.stop()


if __name__ == "__main__":
    main(
        source=f"{config.INPUT_FOLDER}/{config.INPUT_VIDEO}",
        output=f"{config.INPUT_FOLDER}/{Path(config.INPUT_VIDEO).stem}_detections",
        weights=f"{config.YOLOV8_FOLDER}/{config.YOLOV8_WEIGHTS}.pt",
        class_filter=config.CLASS_FILTER,
        image_size=config.IMAGE_SIZE,
        confidence=config.CONFIDENCE,
        iou=config.IOU,
        show_image=config.SHOW_IMAGE,
        save_csv=config.SAVE_CSV,
        save_video=config.SAVE_VIDEO
    )

from ultralytics import YOLO
import supervision as sv

import yaml
import cv2
import torch
import time
from tqdm import tqdm
from pathlib import Path

from tools.print_info import print_video_info, step_message
from tools.annotators import box_annotations, mask_annotations
from tools.write_csv import csv_detections_list, write_csv

# For debugging
from icecream import ic


def class_conversion(class_id: int) -> int:
    CLASS_LIST = {
        0: 0,
        1: 0,
        2: 1,
        3: 2,
        4: 2,
        5: 5,
        8: 4,
        9: 3
    }
    return CLASS_LIST[class_id] if class_id in CLASS_LIST.keys() else None


def main():
    # Initialize YOLOv8 Model
    model = YOLO(f"weights/{WEIGHTS}.pt")
    step_message('1', f"YOLOv8 Model {WEIGHTS} Initialized")

    # Initialize video capture
    step_message('2', 'Initializing Video Source')
    video_info = sv.VideoInfo.from_video_path(video_path=f"{FOLDER}/{SOURCE}")
    print_video_info(f"{FOLDER}/{SOURCE}", video_info)
    target = f"{FOLDER}/{Path(SOURCE).stem}"
    samples_number = 200
    stride = round(video_info.total_frames / samples_number)
    frame_generator = sv.get_video_frames_generator(source_path=f"{FOLDER}/{SOURCE}", stride=stride)
    
    # Start video processing
    step_message('3', 'Video Processing Start')
    t_start = time.time()
    results_data = []
    image_count = 0
    total = round(video_info.total_frames / stride)
    with sv.ImageSink(target_dir_path=target) as sink:
        for image in tqdm(frame_generator, total=total, unit='frames'):
            sink.save_image(image=image)
            txt_name = Path(sink.image_name_pattern.format(image_count)).stem

            annotated_image = image.copy()

            # Run YOLOv8 inference
            results = model(
                source=image,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                device=0,
                agnostic_nms=True,
                # classes=CLASS_FILTER,
                retina_masks=True,
                verbose=False
            )[0]

            for cls, xywhn in zip(results.boxes.cls.cpu().numpy(), results.boxes.xywhn.cpu().numpy()):
                new_cls = class_conversion(cls.astype(int))
                center_x, center_y, width, height = xywhn

                with open(f"{target}/{txt_name}.txt", 'a') as txt_file:
                    txt_file.write(f"{new_cls} {center_x} {center_y} {width} {height}\n")
            
            image_count += 1
            
            # Visualization
            detections = sv.Detections.from_ultralytics(results)
            labels = [f"{results.names[class_id]} - {score:.2f}" for _, _, score, class_id, _ in detections] if DRAW_LABELS else None
            
            # Draw boxes
            if DRAW_BOXES: annotated_image = box_annotations(annotated_image, detections, labels)

            # View live results
            if SHOW_RESULTS:
                cv2.namedWindow('Output', cv2.WINDOW_KEEPRATIO)
                cv2.imshow('Output', annotated_image)
                cv2.resizeWindow('Output', 1280, 720)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    # Print total time elapsed
    step_message('Final', f"Total Time: {(time.time() - t_start):.2f} s")


if __name__ == "__main__":
    # Initialize Configuration File
    with open('autolabelling.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Get configuration parameters
    FOLDER = config['INPUT']['FOLDER']
    SOURCE = config['INPUT']['SOURCE']
    WEIGHTS = config['YOLO']['YOLO_WEIGHTS']
    CONFIDENCE = config['DETECTION']['CONFIDENCE']
    CLASS_FILTER = config['DETECTION']['CLASS_FILTER']
    IMAGE_SIZE = config['DETECTION']['IMAGE_SIZE']
    DRAW_BOXES = config['DRAW']['BOXES']
    DRAW_LABELS = config['DRAW']['LABELS']
    DRAW_MASKS = config['DRAW']['MASKS']
    SHOW_RESULTS = config['SHOW']
    CLIP_LENGTH = config['SAVE']['CLIP_LENGTH']
    SAVE_VIDEO = config['SAVE']['VIDEO']
    SAVE_CSV = config['SAVE']['CSV']

    with torch.no_grad():
        main()
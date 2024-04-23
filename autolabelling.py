from ultralytics import YOLO, RTDETR
import supervision as sv

import sys
import argparse
import cv2
import time
import yaml
from tqdm import tqdm
from pathlib import Path
import itertools

from tools.print_info import print_video_info, step_message
from tools.video_info import from_video_path

import autolabelling_config as config

# For debugging
from icecream import ic


def main(
    source: str,
    output: str,
    weights: str,
    image_size: int,
    confidence: float,
    number_images: int,
    show_image: bool
) -> None:
    ic(output)
    quit()
    step_count = itertools.count(1)

    # Initialize model
    if 'v8' in weights or 'v9' in weights:
        model = YOLO(weights)
    elif 'rtdetr' in weights:
        model = RTDETR(weights)
    step_message(next(step_count), f'{Path(weights).stem.upper()} Model Initialized')

    # Initialize video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened(): quit()
    source_info = from_video_path(cap)
    cap.release()
    step_message(next(step_count), 'Video Source Initialized')
    print_video_info(source, source_info)

    if show_image:
        scaled_width = 1280 if source_info.width > 1280 else source_info.width
        k = int(scaled_width * source_info.height / source_info.width)
        scaled_height = k if source_info.height > k else source_info.height

    # Autolabelling settings




    target = OUTPUT
    samples_number = NUMBER_IMAGES
    stride = round(source_info.total_frames / samples_number)
    frame_generator = sv.get_video_frames_generator(source_path=SOURCE, stride=stride)
    
    # Annotators
    line_thickness = int(sv.calculate_dynamic_line_thickness(resolution_wh=(source_info.width, source_info.height)) * 0.5)
    text_scale = sv.calculate_dynamic_text_scale(resolution_wh=(source_info.width, source_info.height)) * 0.5

    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_padding=2, text_position=sv.Position.TOP_LEFT, text_thickness=line_thickness)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=line_thickness)

    # Start video processing
    step_message(process_step(), 'Video Processing Start')
    t_start = time.time()
    image_count = 0
    total = round(source_info.total_frames / stride)
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
                device=DEVICE,
                agnostic_nms=True,
                retina_masks=True,
                verbose=False
            )[0]

            if image_count == 0:
                for values in results.names.values():
                    with open(f"{target}/labelmap.txt", 'a') as txt_file:
                        txt_file.write(f"{values}\n")

            for cls, xywhn in zip(results.boxes.cls.cpu().numpy(), results.boxes.xywhn.cpu().numpy()):
                center_x, center_y, width, height = xywhn

                with open(f"{target}/{txt_name}.txt", 'a') as txt_file:
                    txt_file.write(f"{int(cls)} {center_x} {center_y} {width} {height}\n")
            
            image_count += 1
            
            if SHOW_IMAGE:
                detections = sv.Detections.from_ultralytics(results)
                
                # Draw labels
                object_labels = [f"{results.names[class_id]} ({score:.2f})" for _, _, score, class_id, _, _ in detections]
                annotated_image = label_annotator.annotate(
                    scene=annotated_image,
                    detections=detections,
                    labels=object_labels )
                
                # Draw boxes
                annotated_image = bounding_box_annotator.annotate(
                    scene=annotated_image,
                    detections=detections )

                # View live results
                cv2.namedWindow('Output', cv2.WINDOW_KEEPRATIO)
                cv2.imshow('Output', annotated_image)
                cv2.resizeWindow('Output', 1280, 720)
                
                # Stop if Esc key is pressed
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    # Print total time elapsed
    step_message('Final', f"Total Time: {(time.time() - t_start):.2f} s")


if __name__ == "__main__":
    main(
        source=Path(config.INPUT_FOLDER, config.INPUT_VIDEO),
        output=Path(config.INPUT_FOLDER, config.INPUT_VIDEO, "_dataset"),
        weights=Path(config.YOLOV9_FOLDER, config.YOLOV9_WEIGHTS".pt"),
        image_size=config.IMAGE_SIZE,
        confidence=config.CONFIDENCE,
        number_images=config.SAMPLE_NUMBER,
        show_image=config.SHOW_IMAGE
    )

from ultralytics import YOLO, RTDETR
import supervision as sv

import cv2
import yaml
import torch
import time
import json
import numpy as np
from pathlib import Path
import itertools
from collections import defaultdict, deque

from imutils.video import FileVideoStream
from imutils.video import FPS

from tools.print_info import print_video_info, print_progress, step_message
from tools.video_info import from_video_path
from tools.csv_sink import CSVSink
from tools.speed import ViewTransformer

# For debugging
from icecream import ic


def main(
    source: str,
    output: str,
    weights: str,
    class_filter: list[int],
    image_size: int,
    confidence: float,
    track_length: int,
    show_image: bool,
    save_csv: bool,
    save_video: bool,
    target_height: int,
    target_width: int
) -> None:
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
        scaled_height = int(scaled_width * source_info.height / source_info.width)
        scaled_height = scaled_height if source_info.height > scaled_height else source_info.height

    # Annotators
    line_thickness = int(sv.calculate_dynamic_line_thickness(resolution_wh=(source_info.width, source_info.height)) * 0.5)
    text_scale = sv.calculate_dynamic_text_scale(resolution_wh=(source_info.width, source_info.height)) * 0.5

    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_padding=2, text_position=sv.Position.TOP_LEFT, text_thickness=line_thickness)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=line_thickness)
    trace_annotator = sv.TraceAnnotator(position=sv.Position.CENTER, trace_length=track_length, thickness=line_thickness)

    # Scene calibration
    with open(f"{FOLDER}/{Path(SOURCE).stem}_region.json", 'r') as json_file:
        json_data = json.load(json_file)
        zone_analysis = np.array(json_data['shapes'][0]['points']).astype(np.int32)
        zone_target = np.array( [ [0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1] ] )
    
    polygon_zone = sv.PolygonZone(polygon=zone_analysis, frame_resolution_wh=(source_info.width,source_info.height))
    view_transformer = ViewTransformer(source=zone_analysis, target=zone_target)
    coordinates = defaultdict(lambda: deque(maxlen=int(source_info.fps)))
    speeds = defaultdict(lambda: deque(maxlen=10))

    # Start video processing
    step_message(next(step_count), 'Start Video Processing')
    fvs = FileVideoStream(source)
    video_sink = sv.VideoSink(target_path=f"{output}.mp4", video_info=source_info)
    csv_sink = CSVSink(file_name=f"{output}.csv")

    frame_number = 0
    fvs.start()
    fps = FPS().start()
    with video_sink, csv_sink:
        while fvs.more():
            image = fvs.read()
            if image is None:
                print()
                break
            annotated_image = image.copy()
            annotated_image = sv.draw_polygon(scene=annotated_image, polygon=zone_analysis, color=sv.Color.RED)

            # YOLO inference
            results = model.track(
                source=image,
                persist=True,
                imgsz=image_size,
                conf=confidence,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                classes=class_filter,
                retina_masks=True,
                verbose=False
            )[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = detections.with_nms()
                    
            # Polygon zone filter
            # detections = detections[polygon_zone.trigger(detections)]

            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            points = view_transformer.transform_points(points=points).astype(int)

            if show_image or save_video:
                # Draw labels
                object_labels =[]
                for tracker_id, [x, y] in zip(detections.tracker_id, points):
                    coordinates[tracker_id].append([frame_number, x, y])
                    if len(coordinates[tracker_id]) < source_info.fps:
                        object_labels.append(f"#{tracker_id}")
                    else:
                        t_0, x_0, y_0 = coordinates[tracker_id][0]
                        t_1, x_1, y_1 = coordinates[tracker_id][-1]

                        distance = abs(np.sqrt((y_1-y_0)**2 + (x_1-x_0)**2))
                        time_diff = (t_1 - t_0) / source_info.fps

                        speeds[tracker_id].append(distance / time_diff * 3.6)

                        mean_speed = sum(speeds[tracker_id]) / len(speeds[tracker_id])

                        object_labels.append(f"#{tracker_id} {int(mean_speed)} Km/h")

                annotated_image = label_annotator.annotate(
                    scene=annotated_image,
                    detections=detections,
                    labels=object_labels )

                # Draw boxes
                annotated_image = bounding_box_annotator.annotate(
                    scene=annotated_image,
                    detections=detections )
                
                # Draw tracks
                annotated_image = trace_annotator.annotate(
                    scene=annotated_image,
                    detections=detections )
                
            if save_video: video_sink.write_frame(frame=annotated_image)
            if save_csv: csv_sink.append(detections, custom_data={'frame_number': frame_number})

            print_progress(frame_number, source_info.total_frames)
            frame_number += 1

            if show_image:
                cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Output', int(scaled_width), int(scaled_height))
                cv2.imshow('Output', annotated_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n")
                    break
            
            fps.update()


    fps.stop()
    step_message(next(step_count), f"Elapsed Time: {fps.elapsed():.2f} s")
    step_message(next(step_count), f"FPS: {fps.fps():.2f}")
    
    cv2.destroyAllWindows()
    fvs.stop()


if __name__ == "__main__":
    # Initialize Configuration File
    with open('tracking_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Get configuration parameters
    MODEL_FOLDER = config['MODEL']['YOLOV9_FOLDER']
    MODEL_WEIGHTS = config['MODEL']['YOLOV9_WEIGHTS']
    FOLDER = config['INPUT']['FOLDER']
    SOURCE = config['INPUT']['SOURCE']
    CLASS_FILTER = config['DETECTION']['CLASS_FILTER']
    IMAGE_SIZE = config['DETECTION']['IMAGE_SIZE']
    CONFIDENCE = config['DETECTION']['CONFIDENCE']
    TRACK_LENGTH = config['DRAW']['TRACK_LENGTH']
    SHOW_RESULTS = config['SHOW']
    SAVE_VIDEO = config['SAVE']['VIDEO']
    SAVE_CSV = config['SAVE']['CSV']
    TARGET_HEIGHT = config['CALIBRATION']['LENGTH']
    TARGET_WIDTH = config['CALIBRATION']['WIDTH']
    
    main(
        source=f"{FOLDER}/{SOURCE}",
        output=f"{FOLDER}/{Path(SOURCE).stem}_tracking",
        weights=f"{MODEL_FOLDER}/{MODEL_WEIGHTS}.pt",
        class_filter=CLASS_FILTER,
        image_size=IMAGE_SIZE,
        confidence=CONFIDENCE,
        track_length=TRACK_LENGTH,
        show_image=SHOW_RESULTS,
        save_csv=SAVE_CSV,
        save_video=SAVE_VIDEO,
        target_height=TARGET_HEIGHT,
        target_width=TARGET_WIDTH,
    )

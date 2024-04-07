from ultralytics import YOLO
import supervision as sv

import yaml
import cv2
import time
from tqdm import tqdm
from pathlib import Path

from tools.print_info import print_video_info, print_progress, step_message
from tools.write_csv import output_data_list, write_csv

# For debugging
from icecream import ic


def process_step() -> str:
    """ Simple step counter """
    global step_count
    step_count = str(int(step_count) + 1)
    return step_count


def main():
    # Initialize YOLOv8 Model
    model = YOLO(f"{MODEL_FOLDER}/{MODEL_WEIGHTS}.pt")
    step_message(process_step(), 'YOLOv8 Model Initialized')

    # Initialize Byte Tracker
    if TRACKING:
        byte_tracker = sv.ByteTrack()
        step_message(process_step(), 'ByteTrack Tracker Initialized')

    # Initialize video capture
    step_message(process_step(), 'Initializing Video Source')
    video_info = sv.VideoInfo.from_video_path(video_path=f"{FOLDER}/{SOURCE}")
    frame_generator = sv.get_video_frames_generator(source_path=f"{FOLDER}/{SOURCE}")
    # print_video_info(f"{FOLDER}/{SOURCE}", video_info)
    
    if TRACKING:
        target = f"{FOLDER}/{Path(SOURCE).stem}_tracking"
    else:
        target = f"{FOLDER}/{Path(SOURCE).stem}_detection"

    line_thickness = int(sv.calculate_dynamic_line_thickness(resolution_wh=(video_info.width,video_info.height)) * 0.5)
    text_scale = sv.calculate_dynamic_text_scale(resolution_wh=(video_info.width,video_info.height)) * 0.5

    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_padding=2, text_position=sv.Position.TOP_LEFT, text_thickness=line_thickness)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=line_thickness)
    mask_annotator = sv.MaskAnnotator()
    trace_annotator = sv.TraceAnnotator(position=sv.Position.CENTER, trace_length=TRACK_LENGTH, thickness=line_thickness)
    heatmap_annotator = sv.HeatMapAnnotator()

    # Start video processing
    step_message(process_step(), 'Video Processing Started')
    t_start = time.time()
    results_data = []
    frame_number = 0
    with sv.VideoSink(target_path=f"{target}.mp4", video_info=video_info, codec="mp4v") as sink:
        for image in tqdm(frame_generator, total=video_info.total_frames, unit='frames'):
            annotated_image = image.copy()

            # Process YOLOv8 detections
            results = model(
                source=image,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                device=0,
                agnostic_nms=True,
                classes=CLASS_FILTER,
                retina_masks=True,
                verbose=False
            )[0]
            detections = sv.Detections.from_ultralytics(results)

            # Update tracks
            if TRACKING:
                tracks = byte_tracker.update_with_detections(detections)

            # Draw labels
            if DRAW_LABELS:
                if TRACKING:
                    object_labels = [f"{results.names[class_id]} {tracker_id} ({score:.2f})" for _, _, score, class_id, tracker_id, _ in tracks]
                else:
                    object_labels = [f"{results.names[class_id]} ({score:.2f})" for _, _, score, class_id, _, _ in detections]
                    
                annotated_image = label_annotator.annotate(
                    scene=annotated_image,
                    detections=tracks if TRACKING else detections,
                    labels=object_labels
                )

            # Draw boxes
            if DRAW_BOXES:
                annotated_image = bounding_box_annotator.annotate(
                    scene=annotated_image,
                    detections=tracks if TRACKING else detections
                )
                
            # Draw masks
            if DRAW_MASKS:
                annotated_image = mask_annotator.annotate(
                    scene=annotated_image,
                    detections=detections
                )

            # Draw tracks
            if DRAW_TRACKS and TRACKING:
                annotated_image = trace_annotator.annotate(
                    scene=annotated_image,
                    detections=tracks
                )

            # Draw heatmap
            if DRAW_HEATMAP:
                annotated_image = heatmap_annotator.annotate(
                    scene=annotated_image,
                    detections=tracks if TRACKING else detections
                )

            # Save video
            if SAVE_VIDEO: sink.write_frame(frame=annotated_image)

            # Save data in list

            if TRACKING:
                results_data = output_data_list(results_data, frame_number, tracks, results.names)
            else:
                results_data = output_data_list(results_data, frame_number, detections, results.names)

            frame_number += 1

            # View live results
            if SHOW_RESULTS:
                cv2.namedWindow('Output', cv2.WINDOW_KEEPRATIO)
                cv2.imshow('Output', annotated_image)
                cv2.resizeWindow('Output', 1280, 720)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    # Saving data in CSV
    if SAVE_CSV:
        step_message(process_step(), 'Saving Results in CSV file')
        write_csv(f"{target}.csv", results_data)
    
    # Print total time elapsed
    step_message('Final', f"Total Time: {(time.time() - t_start):.2f} s")


if __name__ == "__main__":
    # Initialize Configuration File
    with open('tracking_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Get configuration parameters
    MODEL_FOLDER = config['MODEL']['YOLOV8_FOLDER']
    MODEL_WEIGHTS = config['MODEL']['YOLOV8_WEIGHTS']
    FOLDER = config['INPUT']['FOLDER']
    SOURCE = config['INPUT']['SOURCE']
    CONFIDENCE = config['DETECTION']['CONFIDENCE']
    CLASS_FILTER = config['DETECTION']['CLASS_FILTER']
    IMAGE_SIZE = config['DETECTION']['IMAGE_SIZE']
    TRACKING = config['TRACKING']
    DRAW_BOXES = config['DRAW']['BOXES']
    DRAW_LABELS = config['DRAW']['LABELS']
    DRAW_MASKS = config['DRAW']['MASKS']
    DRAW_TRACKS = config['DRAW']['TRACKS']
    DRAW_HEATMAP = config['DRAW']['HEATMAP']
    TRACK_LENGTH = config['DRAW']['TRACK_LENGTH']
    SHOW_RESULTS = config['SHOW']
    SAVE_VIDEO = config['SAVE']['VIDEO']
    SAVE_CSV = config['SAVE']['CSV']

    step_count = '0'

    main()

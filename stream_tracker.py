from ultralytics import YOLO, RTDETR
import supervision as sv

import cv2
import yaml
import torch
import time
from pathlib import Path
import itertools

from imutils.video import WebcamVideoStream
from imutils.video import FPS

from tools.print_info import print_video_info, print_progress, step_message
from tools.video_info import from_camera
from tools.csv_sink import CSVSink

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
    save_source: bool
) -> None:
    step_count = itertools.count(1)
    
    # Initialize model
    if 'v8' in weights or 'v9' in weights:
        model = YOLO(weights)
    elif 'rtdetr' in weights:
        model = RTDETR(weights)
    step_message(next(step_count), f'{Path(weights).stem.upper()} Model Initialized')

    # Initialize video capture
    stream_source = eval(source) if source.isnumeric() else source
    cap = cv2.VideoCapture(stream_source)
    if not cap.isOpened(): quit()
    source_info = from_camera(cap)
    cap.release()
    step_message(next(step_count), 'Video Source Initialized')
    print_video_info(source, source_info)

    if show_image:
        scaled_width = 1280 if source_info.width > 1280 else source_info.width
        k = int(scaled_width * source_info.height / source_info.width)
        scaled_height = k if source_info.height > k else source_info.height

    # Annotators
    line_thickness = int(sv.calculate_dynamic_line_thickness(resolution_wh=(source_info.width, source_info.height)) * 0.5)
    text_scale = sv.calculate_dynamic_text_scale(resolution_wh=(source_info.width, source_info.height)) * 0.5

    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_padding=2, text_position=sv.Position.TOP_LEFT, text_thickness=line_thickness)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=line_thickness)
    trace_annotator = sv.TraceAnnotator(position=sv.Position.CENTER, trace_length=track_length, thickness=line_thickness)

    # Iniciar procesamiento de video
    step_message(next(step_count), 'Start Video Processing')
    vs = WebcamVideoStream(stream_source)
    video_sink = sv.VideoSink(target_path=f"{output}.mp4", video_info=source_info)
    csv_sink = CSVSink(file_name=f"{output}.csv")
    source_sink = sv.VideoSink(target_path=f"{output}_source.mp4", video_info=source_info)
    
    frame_number = 0
    vs.start()
    # time.sleep(1.0)
    fps = FPS().start()
    with video_sink, csv_sink, source_sink:
        while True:
            image = vs.read()
            annotated_image = image.copy()

            # YOLOv8 inference
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

            if show_image or save_video:
                # Draw labels
                object_labels = [f"{data['class_name']} {tracker_id} ({score:.2f})" for _, _, score, _, tracker_id, data in detections]
                annotated_image = label_annotator.annotate(
                    scene=annotated_image,
                    detections=detections,
                    labels=object_labels )

                # Draw boxes
                annotated_image = bounding_box_annotator.annotate(
                    scene=annotated_image,
                    detections=detections )
                
                # Draw tracks
                if detections.tracker_id is not None:
                    annotated_image = trace_annotator.annotate(
                        scene=annotated_image,
                        detections=detections )
                
            if save_video: video_sink.write_frame(frame=annotated_image)
            if save_csv: csv_sink.append(detections, custom_data={'frame_number': frame_number})
            if save_source: source_sink.write_frame(frame=image)

            print_progress(frame_number, None)
            frame_number += 1

            if show_image:
                cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Output', int(scaled_width), int(scaled_height))
                cv2.imshow("Output", annotated_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n")
                    break
            
            fps.update()

    fps.stop()
    step_message(next(step_count), f"Elapsed Time: {fps.elapsed():.2f} s")
    step_message(next(step_count), f"FPS: {fps.fps():.2f}")

    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":
        # Initialize Configuration File
    with open('tracking_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Get configuration parameters
    # MODEL_FOLDER = config['MODEL']['YOLOV8_FOLDER']
    # MODEL_WEIGHTS = config['MODEL']['YOLOV8_WEIGHTS']
    MODEL_FOLDER = config['MODEL']['YOLOV9_FOLDER']
    MODEL_WEIGHTS = config['MODEL']['YOLOV9_WEIGHTS']
    # MODEL_FOLDER = config['MODEL']['RTDETR_FOLDER']
    # MODEL_WEIGHTS = config['MODEL']['RTDETR_WEIGHTS']
    FOLDER = config['INPUT']['FOLDER']
    SOURCE = config['INPUT']['SOURCE']
    CLASS_FILTER = config['DETECTION']['CLASS_FILTER']
    IMAGE_SIZE = config['DETECTION']['IMAGE_SIZE']
    CONFIDENCE = config['DETECTION']['CONFIDENCE']
    TRACK_LENGTH = config['DRAW']['TRACK_LENGTH']
    SHOW_RESULTS = config['SHOW']
    SAVE_VIDEO = config['SAVE']['VIDEO']
    SAVE_CSV = config['SAVE']['CSV']
    SAVE_SOURCE = config['SAVE']['SOURCE']

    main(
        source=f"{SOURCE}",
        output=f"{FOLDER}/webcam_tracking",
        weights=f"{MODEL_FOLDER}/{MODEL_WEIGHTS}.pt",
        class_filter=CLASS_FILTER,
        image_size=IMAGE_SIZE,
        confidence=CONFIDENCE,
        track_length=TRACK_LENGTH,
        show_image=SHOW_RESULTS,
        save_csv=SAVE_CSV,
        save_video=SAVE_VIDEO,
        save_source=SAVE_SOURCE
    )

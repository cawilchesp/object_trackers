from ultralytics import YOLO
import supervision as sv

import yaml
import torch
import cv2
from pathlib import Path
import itertools

from imutils.video import WebcamVideoStream
from imutils.video import FPS

from tools.print_info import print_video_info, print_progress, step_message
from tools.video_info import from_camera
from tools.pose_annotator import PoseAnnotator
from tools.csv_sink import CSVSink

# For debugging
from icecream import ic


def main(
    source: str,
    output: str,
    weights: str,
    image_size: int,
    confidence: float,
    draw_circles: bool,
    draw_lines: bool,
    draw_labels: bool,
    show_image: bool,
    save_csv: bool,
    save_video: bool,
    save_source: bool
) -> None:
    step_count = itertools.count(1)

    # Initialize model
    model = YOLO(weights)
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
    pose_annotator = PoseAnnotator(thickness=2, radius=4)

    # Start video processing
    step_message(next(step_count), 'Start Video Processing')
    vs = WebcamVideoStream(stream_source)
    video_sink = sv.VideoSink(target_path=f"{output}.mp4", video_info=source_info)
    # csv_sink = CSVSink(file_name=f"{output}.csv")
    source_sink = sv.VideoSink(target_path=f"{output}_source.mp4", video_info=source_info)

    frame_number = 0
    vs.start()
    fps = FPS().start()
    with video_sink, source_sink:
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
                retina_masks=True,
                verbose=False
            )[0]

            if show_image or save_video:
                # Draw poses
                annotated_image = pose_annotator.annotate(
                    scene=annotated_image,
                    ultralytics_results=results,
                    circles=draw_circles,
                    lines=draw_lines,
                    labels=draw_labels
                )
            
            if save_video: video_sink.write_frame(frame=annotated_image)
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
    with open('pose_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Get configuration parameters
    MODEL_FOLDER = config['MODEL']['YOLOV8_FOLDER']
    MODEL_WEIGHTS = config['MODEL']['YOLOV8_WEIGHTS']
    FOLDER = config['INPUT']['FOLDER']
    SOURCE = config['INPUT']['SOURCE']
    IMAGE_SIZE = config['DETECTION']['IMAGE_SIZE']
    CONFIDENCE = config['DETECTION']['CONFIDENCE']
    DRAW_CIRCLES = config['DRAW']['CIRCLES']
    DRAW_LINES = config['DRAW']['LINES']
    DRAW_LABELS = config['DRAW']['LABELS']
    SHOW_IMAGE = config['SHOW']
    SAVE_VIDEO = config['SAVE']['VIDEO']
    SAVE_CSV = config['SAVE']['CSV']
    SAVE_SOURCE = config['SAVE']['SOURCE']

    main(
        source=f"{SOURCE}",
        output=f"{FOLDER}/stream_pose_tracking",
        weights=f"{MODEL_FOLDER}/{MODEL_WEIGHTS}-pose.pt",
        image_size=IMAGE_SIZE,
        confidence=CONFIDENCE,
        draw_circles=DRAW_CIRCLES,
        draw_lines=DRAW_LINES,
        draw_labels=DRAW_LABELS,
        show_image=SHOW_IMAGE,
        save_csv=SAVE_CSV,
        save_video=SAVE_VIDEO,
        save_source=SAVE_SOURCE
    )

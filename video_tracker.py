from ultralytics import YOLO
import supervision as sv

import cv2
import torch
import time
from pathlib import Path
import itertools

from imutils.video import FileVideoStream
from imutils.video import FPS

import config
from tools.print_info import print_video_info, print_progress, step_message
from tools.video_info import from_video_path
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
    save_video: bool
) -> None:
    step_count = itertools.count(1)

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
    
    # GPU availability
    step_message(next(step_count), f"Processor: {'GPU ✅' if torch.cuda.is_available() else 'CPU ⚠️'}")

    # Initialize model
    model = YOLO(weights)
    step_message(next(step_count), f'{Path(weights).stem.upper()} Model Initialized')

    # Annotators
    line_thickness = int(sv.calculate_optimal_line_thickness(resolution_wh=(source_info.width, source_info.height)) * 0.5)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=(source_info.width, source_info.height)) * 0.5

    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_padding=2, text_position=sv.Position.TOP_LEFT, text_thickness=line_thickness)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=line_thickness)
    trace_annotator = sv.TraceAnnotator(position=sv.Position.CENTER, trace_length=track_length, thickness=line_thickness)

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
            
            # YOLOv8 inference
            results = model.track(
                source=image,
                persist=True,
                imgsz=image_size,
                conf=confidence,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                # classes=class_filter,
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

            print_progress(frame_number, source_info.total_frames)
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
    fvs.stop()


if __name__ == "__main__":
    main(
        source=f"{config.INPUT_FOLDER}/{config.INPUT_VIDEO}",
        output=f"{config.INPUT_FOLDER}/{Path(config.INPUT_VIDEO).stem}_tracking",
        weights=f"{config.YOLOV9_FOLDER}/{config.YOLOV9_WEIGHTS}.pt",
        class_filter=config.CLASS_FILTER,
        image_size=config.IMAGE_SIZE,
        confidence=config.CONFIDENCE,
        track_length=config.TRACK_LENGTH,
        show_image=config.SHOW_IMAGE,
        save_csv=config.SAVE_CSV,
        save_video=config.SAVE_VIDEO
    )
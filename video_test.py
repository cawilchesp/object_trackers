import supervision as sv

import cv2
import torch
import time
from pathlib import Path
import itertools
from datetime import datetime

from inference.core.interfaces.camera.entities import VideoFrame

from imutils.video import FileVideoStream
from imutils.video import FPS

from sinks.model_sink import ModelSink
from sinks.track_sink import TrackSink
from sinks.annotate_sink import AnnotateSink

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

    # Initialize model
    model_sink = ModelSink(
        weights_path=weights,
        image_size=image_size,
        confidence=confidence,
        class_filter=class_filter
        )
    step_message(next(step_count), f'{Path(weights).stem.upper()} Model Initialized')

    # GPU availability
    step_message(next(step_count), f"Processor: {'GPU ✅' if torch.cuda.is_available() else 'CPU ⚠️'}")

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

    # Initialize track sink
    annotate_sink = AnnotateSink(
        source_info=source_info,
        track_length=track_length,
        box=False,
        trace=True
    )

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

            frame = [VideoFrame(
                image=image,
                frame_id=frame_number,
                frame_timestamp=datetime.now()
            )]

            detections = model_sink.detect(frame=frame)[0]

            if show_image or save_video:
                annotated_image, detections = annotate_sink.on_prediction(detections=detections, frame=frame)
                
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
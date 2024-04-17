from ultralytics import YOLO, RTDETR
import supervision as sv

import cv2
import yaml
import torch
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
    show_image: bool,
    save_video: bool,
    save_source: bool
) -> None:
    step_count = itertools.count(1)
    
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

    # Start video processing
    step_message(next(step_count), 'Start Video Processing')
    vs = WebcamVideoStream(stream_source)
    video_sink = sv.VideoSink(target_path=f"{output}.mp4", video_info=source_info)
    source_sink = sv.VideoSink(target_path=f"{output}_source.mp4", video_info=source_info)
    
    frame_number = 0
    vs.start()
    fps = FPS().start()
    with video_sink, source_sink:
        while True:
            image = vs.read()
            annotated_image = image.copy()
            # Processes

            # if show_image or save_video:
                # process to show
                
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
    main(
        source=f"{SOURCE}",
        output=f"{FOLDER}/stream_tracking",
        show_image=SHOW_RESULTS,
        save_video=SAVE_VIDEO,
        save_source=SAVE_SOURCE
    )

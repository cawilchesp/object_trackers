from ultralytics import YOLO, RTDETR
import supervision as sv

import cv2
import yaml
import torch
from pathlib import Path
import itertools
import easyocr

from imutils.video import WebcamVideoStream
from imutils.video import FPS

from tools.print_info import print_video_info, print_progress, step_message
from tools.video_info import from_camera
from tools.drawings import draw_info_rectangle
from tools.csv_sink import CSVSink

# For debugging
from icecream import ic


def main(
    source: str,
    output: str,
) -> None:
    step_count = itertools.count(1)

    # Initialize OCR model
    reader = easyocr.Reader(
        lang_list=['en'],
        gpu=True,
        model_storage_directory='D:/Data/models/ocr' )
    step_message(next(step_count), 'OCR Model Initialized')

    # Initialize video capture
    stream_source = eval(source) if source.isnumeric() else source
    cap = cv2.VideoCapture(stream_source)
    if not cap.isOpened(): quit()
    source_info = from_camera(cap)
    cap.release()
    step_message(next(step_count), 'Video Source Initialized')
    print_video_info(source, source_info)
   
    scaled_width = 1280 if source_info.width > 1280 else source_info.width
    k = int(scaled_width * source_info.height / source_info.width)
    scaled_height = k if source_info.height > k else source_info.height

    # Start video processing
    step_message(next(step_count), 'Start Video Streaming')
    vs = WebcamVideoStream(stream_source)

    vs.start()
    fps = FPS().start()
    while True:
        image = vs.read()
        annotated_image = image.copy()

        # OCR inference
        step_message(next(step_count), 'OCR Processing')
        results = reader.readtext(
            image=image,
            detail=1,
            allowlist='0123456789'
        )

        for result in results:
            xyxy, text, score = result
            x1, y1 = xyxy[0]
            x2, y2 = xyxy[2]

            cv2.rectangle(
                img=annotated_image,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=(0,0,255),
                thickness=2,
                lineType=cv2.LINE_AA
            )

            annotated_image = draw_info_rectangle(
                scene=annotated_image,
                texts=[f"Jugador: {text} ({score:.2f})"] )

        cv2.namedWindow('Stream', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Stream', int(scaled_width), int(scaled_height))
        cv2.imshow("Stream", annotated_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n")

            cv2.imwrite(f"{output}/webcam_ocr.png", annotated_image)

            break

        fps.update()

    fps.stop()
    step_message(next(step_count), f"Elapsed Time: {fps.elapsed():.2f} s")
    step_message(next(step_count), f"FPS: {fps.fps():.2f}")

    cv2.destroyAllWindows()
    vs.stop()
    

if __name__ == "__main__":
    main(
        source='0',
        output='D:/Data/Tejo',
    )
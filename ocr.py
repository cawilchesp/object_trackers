from ultralytics import YOLO, RTDETR
import supervision as sv

import cv2
import yaml
import torch
import time
from pathlib import Path
import itertools

from imutils.video import FileVideoStream
from imutils.video import FPS

from tools.print_info import print_video_info, print_progress, step_message
from tools.video_info import from_video_path
from tools.csv_sink import CSVSink

from tools.drawings import draw_info_rectangle

import easyocr

# For debugging
from icecream import ic


def main(
    source: str,
) -> None:
    reader = easyocr.Reader(
        lang_list=['en'],
        gpu=True,
        model_storage_directory='D:/Data/models/ocr',
    )
    image = cv2.imread(source)
    annotated_image = image.copy()
    
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

    cv2.imwrite(f"{FOLDER}/{Path(SOURCE).stem}_ocr.png", annotated_image)
    
    cv2.imshow("Output", annotated_image)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    FOLDER = 'D:/Data/Tejo'
    # SOURCE = 'ocr_test.jpg'
    # SOURCE = 'ocr_test2.jpg'
    SOURCE = 'ocr_test3.jpg'

    main(
        source=f"{FOLDER}/{SOURCE}",
    )
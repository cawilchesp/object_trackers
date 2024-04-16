from ultralytics import YOLO, RTDETR
import supervision as sv

import imutils
from imutils.video import FileVideoStream
from imutils.video import FPS

import cv2
import yaml
import torch
import time
from tqdm import tqdm
from pathlib import Path

from tools.print_info import print_video_info, print_progress, step_message
from tools.video_info import from_video_path
from tools.csv_sink import CSVSink

# For debugging
from icecream import ic


def time_synchronized():
    """ PyTorch accurate time """
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return time.time()


def process_step() -> str:
    """ Simple step counter """
    global step_count
    step_count = str(int(step_count) + 1)
    return step_count


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

    # Initialize Model
    if 'v8' in weights or 'v9' in weights:
        model = YOLO(weights)
        step_message(process_step(), f'{weights} Model Initialized')
    elif 'rtdetr' in weights:
        model = RTDETR(weights)
        step_message(process_step(), 'RT-DETR Model Initialized')

    # Inicializar captura de video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened(): quit()
    source_info = from_video_path(cap)
    print_video_info(source, source_info)
    cap.release()
    
    # Anotadores
    line_thickness = int(sv.calculate_dynamic_line_thickness(resolution_wh=(source_info.width, source_info.height)) * 0.5)
    text_scale = sv.calculate_dynamic_text_scale(resolution_wh=(source_info.width, source_info.height)) * 0.5

    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_padding=2, text_position=sv.Position.TOP_LEFT, text_thickness=line_thickness)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=line_thickness)
    trace_annotator = sv.TraceAnnotator(position=sv.Position.CENTER, trace_length=track_length, thickness=line_thickness)

    # Iniciar procesamiento de video
    step_message(process_step(), 'Procesamiento de video iniciado')
    fvs = FileVideoStream(source)
    video_sink = sv.VideoSink(target_path=f"{output}.mp4", video_info=source_info)
    csv_sink = CSVSink(file_name=f"{output}.csv")

    frame_number = 0
    fvs.start()
    time.sleep(1.0)
    fps = FPS().start()
    with video_sink, csv_sink:
        t_frame_start = time_synchronized()
        while fvs.more():
            print(frame_number)

            image = fvs.read()
            annotated_image = image.copy()
            
            # Process YOLOv8 detections
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
                # Dibujar etiquetas
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

            frame_number += 1

            if show_image:
                cv2.imshow("Resultado", annotated_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            fps.update()

    t_frame_end = time_synchronized()
    step_message('Final', f"Total Time: {(t_frame_end - t_frame_start):.2f} s")
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    
    cv2.destroyAllWindows()
    fvs.stop()


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

    step_count = '0'
    
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
        save_video=SAVE_VIDEO
    )
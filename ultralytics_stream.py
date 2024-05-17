from ultralytics import YOLO
import supervision as sv

import cv2
import signal
import sys
import torch
import numpy as np
from pathlib import Path
import itertools
from functools import partial
from typing import Union, Optional, List

from inference.core.interfaces.stream.inference_pipeline import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame

import config
from tools.general import find_in_list, load_zones
from tools.timers import ClockBasedTimer
from tools.print_info import print_video_info, print_progress, step_message
from tools.video_info import from_video_path, from_camera

# For debugging
from icecream import ic




COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
OUTPUT_WRITER: cv2.VideoWriter = None
PIPELINE: Optional[InferencePipeline] = None
def signal_handler(sig, frame):
    print("Terminating")
    if PIPELINE is not None:
        PIPELINE.terminate()
        PIPELINE.join()
    OUTPUT_WRITER.release()
    sys.exit(0)



def main(
    source: str,
    output: str,
    weights: str,
    zone_path: str,
    class_filter: list[int],
    image_size: int,
    confidence: float,
    iou: float,
    track_length: int,
    show_image: bool,
    save_csv: bool,
    save_video: bool
) -> None:
    step_count = itertools.count(1)

    # Initialize video capture
    video_source = eval(source) if source.isnumeric() else source
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened(): quit()
    if source.isnumeric() or source.lower().startswith('rtsp://'):
        source_info = from_camera(cap)
    else:
        source_info = from_video_path(cap)
    cap.release()
    step_message(next(step_count), 'Origen del Video Inicializado')
    print_video_info(source, source_info)

    scaled_width = 1280 if source_info.width > 1280 else source_info.width
    scaled_height = int(scaled_width * source_info.height / source_info.width)
    scaled_height = scaled_height if source_info.height > scaled_height else source_info.height

    # GPU availability
    step_message(next(step_count), f"Processor: {'GPU ✅' if torch.cuda.is_available() else 'CPU ⚠️'}")

    # Initialize model
    model = YOLO(weights)
    step_message(next(step_count), f'{Path(weights).stem.upper()} Model Initialized ✅')

    # Initialize tracker
    tracker = sv.ByteTrack(minimum_matching_threshold=0.8)
    step_message(next(step_count), 'ByteTrack Tracker Initialized ✅')

    # Annotators
    line_thickness = int(sv.calculate_optimal_line_thickness(resolution_wh=(source_info.width, source_info.height)) * 0.5)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=(source_info.width, source_info.height)) * 0.5

    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_padding=2, text_position=sv.Position.TOP_LEFT, text_thickness=line_thickness, color=COLORS)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=line_thickness, color=COLORS)
    trace_annotator = sv.TraceAnnotator(position=sv.Position.BOTTOM_CENTER, trace_length=track_length, thickness=line_thickness, color=COLORS)
    

    fps_monitor = sv.FPSMonitor()
    polygons = load_zones(file_path=zone_path)
    timers = [ClockBasedTimer() for _ in polygons]
    zones = [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=(sv.Position.CENTER,),
        ) for polygon in polygons
    ]

    source_info.fps = 11
    global OUTPUT_WRITER
    OUTPUT_WRITER = cv2.VideoWriter(f"{output}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), source_info.fps, (source_info.width, source_info.height))

    # Start video processing
    step_message(next(step_count), 'Start Video Processing')

    def inference_callback(frame: VideoFrame) -> sv.Detections:
        frame_number = frame[0].frame_id
        fps_monitor.tick()
        fps = fps_monitor.fps

        results = model(
            source=frame[0].image,
            imgsz=image_size,
            conf=confidence,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=False,
        )[0]
        
        detections = sv.Detections.from_ultralytics(results).with_nms(threshold=0.7)
        detections = tracker.update_with_detections(detections)

        annotated_image = frame[0].image.copy()

        annotated_image = sv.draw_text(
            scene=annotated_image,
            text=f"{fps:.1f}",
            text_anchor=sv.Point(40, 30),
            background_color=sv.Color.from_hex("#A351FB"),
            text_color=sv.Color.from_hex("#000000") )

        for idx, zone in enumerate(zones):
            annotated_image = sv.draw_polygon(
                scene=annotated_image,
                polygon=zone.polygon,
                color=COLORS.by_idx(idx) )

            if detections.tracker_id is not None:
                detections_in_zone = detections[zone.trigger(detections)]
                time_in_zone = timers[idx].tick(detections_in_zone)
                custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)

                # Draw labels
                # object_labels = [f"{data['class_name']} {tracker_id} ({score:.2f})" for _, _, score, _, tracker_id, data in detections]
                object_labels = [
                    f"#{tracker_id} {int(time // 60):02d}:{int(time % 60):02d}"
                    for tracker_id, time in zip(detections_in_zone.tracker_id, time_in_zone)
                ]
                annotated_image = label_annotator.annotate(
                    scene=annotated_image,
                    detections=detections_in_zone,
                    labels=object_labels,
                    custom_color_lookup=custom_color_lookup )
                
                # Draw boxes
                annotated_image = bounding_box_annotator.annotate(
                    scene=annotated_image,
                    detections=detections_in_zone,
                    custom_color_lookup=custom_color_lookup )
                
                # Draw tracks
                annotated_image = trace_annotator.annotate(
                    scene=annotated_image,
                    detections=detections_in_zone,
                custom_color_lookup=custom_color_lookup )
            

        print_progress(frame_number, None)
        OUTPUT_WRITER.write(annotated_image)
        cv2.imshow("Output", annotated_image)
        cv2.waitKey(1)
        
    

    
    
    
    
    
    global PIPELINE
    PIPELINE = InferencePipeline.init_with_custom_logic(
        video_reference=video_source,
        on_video_frame=inference_callback )
    PIPELINE.start()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    print("Press Ctrl+C to terminate")
    main(
        # source=f"{config.INPUT_FOLDER}/{config.INPUT_VIDEO}",
        source='0',
        # output=f"{config.INPUT_FOLDER}/{Path(config.INPUT_VIDEO).stem}_output",
        output=f"{config.INPUT_FOLDER}/webcam_timer",
        weights=f"{config.YOLOV8_FOLDER}/{config.YOLOV8_WEIGHTS}.pt",
        zone_path=f"{config.INPUT_FOLDER}/{Path(config.INPUT_VIDEO).stem}_zones.json",
        class_filter=config.CLASS_FILTER,
        image_size=config.IMAGE_SIZE,
        confidence=config.CONFIDENCE,
        iou=config.IOU,
        track_length=config.TRACK_LENGTH,
        show_image=config.SHOW_IMAGE,
        save_csv=config.SAVE_CSV,
        save_video=config.SAVE_VIDEO
    )
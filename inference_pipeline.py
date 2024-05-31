from ultralytics import YOLO
import supervision as sv

import cv2
import sys
import torch
import signal
import itertools
import numpy as np
from typing import List
from pathlib import Path
from typing import Union, Optional, List
from functools import partial
from datetime import datetime

from sinks.model_sink import ModelSink
from sinks.track_sink import TrackSink
from sinks.annotate_sink import AnnotateSink
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.sinks import multi_sink

import config
from tools.general import find_in_list, load_zones
from tools.timers import ClockBasedTimer
from tools.print_info import print_video_info, print_progress, step_message
from tools.video_info import from_video_path, from_camera, VideoInfo
from tools.write_data import CSVSave

# For debugging
from icecream import ic


PIPELINE: Optional[InferencePipeline] = None
def signal_handler(sig, frame):
    print("Terminating")
    if PIPELINE is not None:
        PIPELINE.terminate()
        PIPELINE.join()
    sys.exit(0)


class ProcessSink:
    def __init__(
        self,
        source_info: VideoInfo,
        track_length: int,
        iou: float,
        zone_configuration_path: str,
        output: str,
    ) -> None:
        self.tracker = sv.ByteTrack(minimum_matching_threshold=iou)

        # Annotators
        line_thickness = int(sv.calculate_optimal_line_thickness(resolution_wh=(source_info.width, source_info.height)) * 0.5)
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=(source_info.width, source_info.height)) * 0.5
        
        self.COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
        self.label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_padding=2, text_position=sv.Position.TOP_LEFT, text_thickness=line_thickness, color=self.COLORS)
        self.bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=line_thickness, color=self.COLORS)
        self.trace_annotator = sv.TraceAnnotator(position=sv.Position.CENTER, trace_length=track_length, thickness=line_thickness, color=self.COLORS)
                
        self.fps_monitor = sv.FPSMonitor()
        self.polygons = load_zones(file_path=zone_configuration_path)
        self.timers = [
            ClockBasedTimer()
            for _ in self.polygons
        ]
        self.zones = [
            sv.PolygonZone(
                polygon=polygon,
                triggering_anchors=(sv.Position.CENTER,),
            )
            for polygon in self.polygons
        ]

        self.video_writer = cv2.VideoWriter(
            filename=f"{output}.mp4",
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=source_info.fps,
            frameSize=source_info.resolution_wh,
        )

        self.csv_writer = CSVSave(file_name=f"{output}.csv")

    def on_prediction(self, detections, frame: VideoFrame) -> None:
        self.fps_monitor.tick()
        fps = self.fps_monitor.fps

        frame_number = frame.frame_id

        detections = self.tracker.update_with_detections(detections)

        annotated_image = frame.image.copy()
        annotated_image = sv.draw_text(
            scene=annotated_image,
            text=f"{fps:.1f}",
            text_anchor=sv.Point(40, 30),
            background_color=sv.Color.from_hex("#A351FB"),
            text_color=sv.Color.from_hex("#000000"),
        )

        for idx, zone in enumerate(self.zones):
            annotated_image = sv.draw_polygon(
                scene=annotated_image,
                polygon=zone.polygon,
                color=self.COLORS.by_idx(idx)
            )

            if detections.tracker_id is not None:
                detections_in_zone = detections[zone.trigger(detections)]
                time_in_zone = self.timers[idx].tick(detections_in_zone)
                custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)

                object_labels = [
                    f"#{tracker_id} {int(time // 60):02d}:{int(time % 60):02d}"
                    for tracker_id, time in zip(detections_in_zone.tracker_id, time_in_zone)
                ]
                annotated_image = self.label_annotator.annotate(
                    scene=annotated_image,
                    detections=detections_in_zone,
                    labels=object_labels,
                    custom_color_lookup=custom_color_lookup
                )
                
                # Draw boxes
                annotated_image = self.bounding_box_annotator.annotate(
                    scene=annotated_image,
                    detections=detections_in_zone,
                    custom_color_lookup=custom_color_lookup
                )
                
                # Draw tracks
                annotated_image = self.trace_annotator.annotate(
                    scene=annotated_image,
                    detections=detections_in_zone,
                    custom_color_lookup=custom_color_lookup
                )

                custom_data = {
                    'frame_number': frame_number,
                    'time': datetime.now(),
                    'zone': idx
                }
                self.csv_writer.append(detections_in_zone, custom_data)
        self.video_writer.write(image=annotated_image)

        cv2.imshow("Processed Video", annotated_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Terminating")
            if PIPELINE is not None:
                PIPELINE.terminate()
                PIPELINE.join()
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

    # GPU availability
    step_message(next(step_count), f"Processor: {'GPU ✅' if torch.cuda.is_available() else 'CPU ⚠️'}")

    # Initialize model
    model_sink = ModelSink(
        weights_path=weights,
        image_size=image_size,
        confidence=confidence,
        class_filter=class_filter )
    step_message(next(step_count), f'{Path(weights).stem.upper()} Model Initialized ✅')

    # Start video processing pipeline
    process_sink = ProcessSink(
        source_info=source_info,
        track_length=track_length,
        iou=iou,
        zone_configuration_path=zone_path,
        output=output )
    step_message(next(step_count), 'Process Pipeline Initialized ✅')

    step_message(next(step_count), 'Start Video Processing')
    global PIPELINE
    PIPELINE = InferencePipeline.init_with_custom_logic(
        video_reference=source,
        on_video_frame=model_sink.detect,
        on_prediction=process_sink.on_prediction,
    )
    PIPELINE.start()


if __name__ == "__main__":
    signal.signal(signal. SIGINT, signal_handler)
    print("Press Ctrl+C to terminate")

    main(
        source=f"{config.INPUT_FOLDER}/{config.INPUT_VIDEO}",
        output=f"{config.INPUT_FOLDER}/{Path(config.INPUT_VIDEO).stem}_output",
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
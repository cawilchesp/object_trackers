from ultralytics import YOLO
import supervision as sv

from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.sinks import VideoFileSink

from typing import List

import cv2
import torch
import numpy as np
from pathlib import Path
import itertools
import signal
import sys

import config
from tools.general import find_in_list, load_zones
from tools.print_info import print_video_info, print_progress, step_message
from tools.video_info import VideoInfo, from_video_path
from tools.csv_sink import CSVSink
from tools.timers import ClockBasedTimer

# For debugging
from icecream import ic


COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)


class CustomSink:
    def __init__(
        self, zone_configuration_path: str,
        classes: List[int]
    ):
        self.classes = classes
        self.tracker = sv.ByteTrack(minimum_matching_threshold=0.8)
        self.fps_monitor = sv.FPSMonitor()
        self.polygons = load_zones(file_path=zone_configuration_path)
        self.timers = [ClockBasedTimer() for _ in self.polygons]
        self.zones = [
            sv.PolygonZone(
                polygon=polygon,
                triggering_anchors=(sv.Position.CENTER,),
            )
            for polygon in self.polygons
        ]
        
    def on_prediction(
        self,
        results,
        frame: VideoFrame,
        output_writer
    ) -> None:
        detections = sv.Detections.from_ultralytics(results).with_nms(threshold=0.7)
        
        self.fps_monitor.tick()
        fps = self.fps_monitor.fps

        detections = detections[find_in_list(detections.class_id, self.classes)]
        detections = self.tracker.update_with_detections(detections)

        annotated_frame = frame.image.copy()
        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=f"{fps:.1f}",
            text_anchor=sv.Point(40, 30),
            background_color=sv.Color.from_hex("#A351FB"),
            text_color=sv.Color.from_hex("#000000"),
        )

        for idx, zone in enumerate(self.zones):
            annotated_frame = sv.draw_polygon(
                scene=annotated_frame, polygon=zone.polygon, color=COLORS.by_idx(idx)
            )

            detections_in_zone = detections[zone.trigger(detections)]
            time_in_zone = self.timers[idx].tick(detections_in_zone)
            custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)

            annotated_frame = COLOR_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                custom_color_lookup=custom_color_lookup,
            )
            labels = [
                f"#{tracker_id} {int(time // 60):02d}:{int(time % 60):02d}"
                for tracker_id, time in zip(detections_in_zone.tracker_id, time_in_zone)
            ]
            annotated_frame = LABEL_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                labels=labels,
                custom_color_lookup=custom_color_lookup,
            )

        output_writer.write(annotated_frame)

        cv2.imshow("Processed Video", annotated_frame)
        cv2.waitKey(1)


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
    cap = cv2.VideoCapture(source)
    if not cap.isOpened(): quit()
    source_info = from_video_path(cap)
    cap.release()
    step_message(next(step_count), 'Video Source Initialized ✅')
    print_video_info(source, source_info)

    # GPU availability
    step_message(next(step_count), f"Processor: {'GPU ✅' if torch.cuda.is_available() else 'CPU ⚠️'}")

    # Initialize model
    model = YOLO(weights)
    step_message(next(step_count), f'{Path(weights).stem.upper()} Model Initialized ✅')

    output_writer = cv2.VideoWriter(f"{output}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), source_info.fps, (source_info.width, source_info.height))

    def signal_handler(sig, frame):
        output_writer.release()
        cv2.destroyAllWindows()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    def inference_callback(frame: VideoFrame) -> sv.Detections:
        results = model(
            source=frame[0].image,
            imgsz=image_size,
            conf=confidence,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=False,
        )[0]
        return results
        
    sink = CustomSink(
        zone_configuration_path=zone_path,
        classes=class_filter
    )

    pipeline = InferencePipeline.init_with_custom_logic(
        video_reference=source,
        on_video_frame=inference_callback,
        on_prediction=lambda results, frame:  sink.on_prediction(results, frame, output_writer),
    )

    pipeline.start()

    try:
        pipeline.join()
    except KeyboardInterrupt:
        pipeline.terminate()


if __name__ == "__main__":
    main(
        source=f"{config.INPUT_FOLDER}/{config.INPUT_VIDEO}",
        output=f"{config.INPUT_FOLDER}/{Path(config.INPUT_VIDEO).stem}_tracking",
        weights=f"{config.YOLOV9_FOLDER}/{config.YOLOV9_WEIGHTS}.pt",
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
   
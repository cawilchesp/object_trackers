from typing import List

import cv2
import numpy as np
from pathlib import Path
import torch

from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame

from ultralytics import YOLO
import supervision as sv

from tools.general import find_in_list, load_zones
from tools.timers import ClockBasedTimer


import config


from icecream import ic


COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)


class CustomSink:
    def __init__(self, zone_configuration_path: str, classes: List[int]):
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
        self.video_sink = sv.VideoSink(target_path="output.mp4", video_info=source_info)

    def on_prediction(self, detections: sv.Detections, frame: VideoFrame) -> None:
        
        detections = sv.Detections(
            xyxy = np.array([detections[0]]),
            mask = detections[1],
            confidence = np.array([detections[2]]),
            class_id = np.array([detections[3]]),
            tracker_id = detections[4],
            data = {'class_name': np.array([detections[5]['class_name']]) }
        )
        
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
        cv2.imshow("Processed Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return None


def main(
    rtsp_url: str,
    zone_configuration_path: str,
    weights: str,
    device: str,
    confidence: float,
    iou: float,
    classes: List[int],
) -> None:
    



    model = YOLO(weights)

    def inference_callback(frame: VideoFrame) -> sv.Detections:
        results = model(frame[0].image, verbose=False, conf=confidence, device=device)[0]
        detections = sv.Detections.from_ultralytics(results).with_nms(threshold=iou)
        return detections
        
    sink = CustomSink(zone_configuration_path=zone_configuration_path, classes=classes)

    pipeline = InferencePipeline.init_with_custom_logic(
        video_reference=rtsp_url,
        on_video_frame=inference_callback,
        on_prediction=sink.on_prediction,
    )

    pipeline.start()

    try:
        pipeline.join()
    except KeyboardInterrupt:
        pipeline.terminate()







if __name__ == "__main__":
    main(
        rtsp_url=f"{config.INPUT_FOLDER}/{config.INPUT_VIDEO}",
        zone_configuration_path=f"{config.INPUT_FOLDER}/{Path(config.INPUT_VIDEO).stem}_zones.json",
        weights=f"{config.YOLOV9_FOLDER}/{config.YOLOV9_WEIGHTS}.pt",
        device='cuda' if torch.cuda.is_available() else 'cpu',
        confidence=config.CONFIDENCE,
        iou=config.IOU,
        classes=config.CLASS_FILTER,
    )
   
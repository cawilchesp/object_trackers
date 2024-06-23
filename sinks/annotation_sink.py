import supervision as sv

import cv2
import numpy as np

from tools.video_info import VideoInfo


class AnnotationSink:
    def __init__(
        self,
        source_info: VideoInfo,
        track_length: int = 50,
        fps: bool = True,
        label: bool = True,
        box: bool = True,
        trace: bool = False
    ) -> None:
        self.fps = fps
        self.label = label
        self.box = box
        self.trace = trace
        
        # Annotators
        line_thickness = int(sv.calculate_optimal_line_thickness(resolution_wh=source_info.resolution_wh) * 0.5)
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=source_info.resolution_wh) * 0.5
        
        if self.fps: self.fps_monitor = sv.FPSMonitor()
        
        if self.label: self.label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_padding=2, text_position=sv.Position.TOP_LEFT, text_thickness=line_thickness)
        if self.box: self.bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=line_thickness)
        if self.trace: self.trace_annotator = sv.TraceAnnotator(position=sv.Position.CENTER, trace_length=track_length, thickness=line_thickness)
                

    def on_detections(self, detections: sv.Detections, image: np.array) -> np.array:
        annotated_image = image.copy()
        
        # Draw FPS box
        if self.fps:
            self.fps_monitor.tick()
            fps_value = self.fps_monitor.fps

            annotated_image = sv.draw_text(
                scene=annotated_image,
                text=f"{fps_value:.1f}",
                text_anchor=sv.Point(40, 30),
                background_color=sv.Color.from_hex("#A351FB"),
                text_color=sv.Color.from_hex("#000000"),
            )

        # Draw labels
        if self.label:
            if detections.tracker_id is None:
                object_labels = [
                    f"{data['class_name']} ({score:.2f})"
                    for _, _, score, _, _, data in detections
                ]
            else:
                object_labels = [
                    f"{data['class_name']} {tracker_id} ({score:.2f})"
                    for _, _, score, _, tracker_id, data in detections
                ]
            annotated_image = self.label_annotator.annotate(
                scene=annotated_image,
                detections=detections,
                labels=object_labels )
            
        # Draw boxes
        if self.box:
            annotated_image = self.bounding_box_annotator.annotate(
                scene=annotated_image,
                detections=detections )
            
        # Draw tracks
        if self.trace and detections.tracker_id is not None:
            annotated_image = self.trace_annotator.annotate(
                scene=annotated_image,
                detections=detections )
        
        return annotated_image
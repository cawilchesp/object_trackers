import supervision as sv

from inference.core.interfaces.camera.entities import VideoFrame

import cv2

from tools.video_info import VideoInfo
from tools.write_data import CSVSave




class TrackSink:
    def __init__(
        self,
        source_info: VideoInfo,
        track_length: int = 50,
        iou: float = 0.7,
    ) -> None:
        self.tracker = sv.ByteTrack(minimum_matching_threshold=iou)

        # Annotators
        line_thickness = int(sv.calculate_optimal_line_thickness(resolution_wh=(source_info.width, source_info.height)) * 0.5)
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=(source_info.width, source_info.height)) * 0.5
        
        self.label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_padding=2, text_position=sv.Position.TOP_LEFT, text_thickness=line_thickness)
        self.bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=line_thickness)
        self.trace_annotator = sv.TraceAnnotator(position=sv.Position.CENTER, trace_length=track_length, thickness=line_thickness)
                
        self.fps_monitor = sv.FPSMonitor()

    def on_prediction(self, detections, frame: VideoFrame) -> None:
        self.fps_monitor.tick()
        fps = self.fps_monitor.fps

        detections = self.tracker.update_with_detections(detections)

        annotated_image = frame[0].image.copy()
        annotated_image = sv.draw_text(
            scene=annotated_image,
            text=f"{fps:.1f}",
            text_anchor=sv.Point(40, 30),
            background_color=sv.Color.from_hex("#A351FB"),
            text_color=sv.Color.from_hex("#000000"),
        )

        if detections.tracker_id is not None:
            object_labels = [
                f"{data['class_name']} {tracker_id} ({score:.2f})"
                for _, _, score, _, tracker_id, data in detections
            ]
            annotated_image = self.label_annotator.annotate(
                scene=annotated_image,
                detections=detections,
                labels=object_labels )
            
            # Draw boxes
            annotated_image = self.bounding_box_annotator.annotate(
                scene=annotated_image,
                detections=detections )
            
            # Draw tracks
            annotated_image = self.trace_annotator.annotate(
                scene=annotated_image,
                detections=detections )
        
        return annotated_image, detections
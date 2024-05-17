from ultralytics import YOLO
import supervision as sv

import cv2
import yaml
import torch
import numpy as np
from pathlib import Path
import itertools

from imutils.video import WebcamVideoStream, FileVideoStream
from imutils.video import FPS

import config
from tools.print_info import print_video_info, print_progress, step_message
from tools.video_info import from_camera, from_video_path
from tools.csv_sink import CSVSink
from tools.timers import FPSBasedTimer, ClockBasedTimer
from tools.general import load_zones

# For debugging
from icecream import ic


def main(
    source: str,
    output: str,
    weights: str,
    class_filter: list[int],
    image_size: int,
    confidence: float,
    iou: float,
    show_image: bool,
    save_csv: bool,
    save_video: bool
) -> None:
    step_count = itertools.count(1)
    
    # Initialize model
    model = YOLO(weights)
    step_message(next(step_count), f'{Path(weights).stem.upper()} Model Initialized')

    # Inicializar captura de video
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

    if show_image:
        scaled_width = 1280 if source_info.width > 1280 else source_info.width
        scaled_height = int(scaled_width * source_info.height / source_info.width)
        scaled_height = scaled_height if source_info.height > scaled_height else source_info.height

    step_message(next(step_count), f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

    # Annotators
    fps_monitor = sv.FPSMonitor()
    COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
    line_thickness = int(sv.calculate_optimal_line_thickness(resolution_wh=(source_info.width, source_info.height)) * 0.5)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=(source_info.width, source_info.height)) * 0.5

    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_padding=2, text_position=sv.Position.TOP_LEFT, text_thickness=line_thickness, color=COLORS)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=line_thickness, color=COLORS)

    # Regions
    polygons = load_zones(file_path=f"{Path(source).parent}/{Path(source).stem}_zones.json")
    regions = [ sv.PolygonZone(
        polygon=polygon,
        triggering_anchors=(sv.Position.CENTER,)) for polygon in polygons ]
    # timers = [FPSBasedTimer() for _ in polygons]
    timers = [ClockBasedTimer() for _ in polygons]

    # Start video processing
    step_message(next(step_count), 'Start Video Processing')
    if source.isnumeric() or source.lower().startswith('rtsp://'):
        fvs = WebcamVideoStream(video_source)
    else:
        fvs = FileVideoStream(source)
    source_info.fps = 10.0
    video_sink = sv.VideoSink(target_path=f"{output}.mp4", video_info=source_info)
    
    frame_number = 0
    fvs.start()
    fps = FPS().start()
    with video_sink:
        # while fvs.more():
        while True:
            image = fvs.read()
            if image is None:
                print()
                break
            annotated_image = image.copy()
            
            # YOLOv8 inference
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
            detections = sv.Detections.from_ultralytics(results).with_nms(threshold=iou)

            fps_monitor.tick()
            fps_rt = fps_monitor.fps
            if show_image or save_video:
                annotated_image = sv.draw_text(
                    scene=annotated_image,
                    text=f"{fps_rt:.1f}",
                    text_anchor=sv.Point(40, 30),
                    background_color=sv.Color.from_hex("#A351FB"),
                    text_color=sv.Color.from_hex("#000000"),
                )


                # Draw regions
                for index, region in enumerate(regions):
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=region.polygon,
                        color=COLORS.by_idx(index)
                    )

                    if detections.tracker_id is not None:
                        detections_in_zone = detections[region.trigger(detections=detections)]
                        time_in_zone = timers[index].tick(detections_in_zone)
                        custom_color_lookup = np.full(detections_in_zone.class_id.shape, index)

                        # Draw labels
                        object_labels = [f"ID {tracker_id} {int(time // 60):02d}:{int(time % 60):02d}" for tracker_id, time in zip(detections_in_zone.tracker_id, time_in_zone)]
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
                
            if save_video: video_sink.write_frame(frame=annotated_image)

            # print_progress(frame_number, source_info.total_frames)
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
    fvs.stop()


if __name__ == "__main__":
    main(
        source=f"{config.INPUT_FOLDER}/{config.INPUT_VIDEO}",
        # source='0',
        output=f"{config.INPUT_FOLDER}/{Path(config.INPUT_VIDEO).stem}_timer",
        # output=f"{config.INPUT_FOLDER}/webcam_timer",
        weights=f"{config.YOLOV8_FOLDER}/{config.YOLOV8_WEIGHTS}.pt",
        class_filter=config.CLASS_FILTER,
        image_size=config.IMAGE_SIZE,
        confidence=config.CONFIDENCE,
        iou=config.IOU,
        show_image=config.SHOW_IMAGE,
        save_csv=config.SAVE_CSV,
        save_video=config.SAVE_VIDEO
    )

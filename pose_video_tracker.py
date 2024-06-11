from ultralytics import YOLO
import supervision as sv

import yaml
import torch
import cv2
from pathlib import Path
import itertools

from imutils.video import FileVideoStream
from imutils.video import FPS

import config
from tools.general import find_in_list, load_zones
from tools.print_info import print_video_info, print_progress, step_message
from tools.video_info import from_video_path
from tools.pose_annotator import PoseAnnotator
from tools.csv_sink import CSVSink

# For debugging
from icecream import ic


def main(
    source: str,
    output: str,
    weights: str,
    image_size: int,
    confidence: float,
    show_image: bool,
    save_video: bool
) -> None:
    step_count = itertools.count(1)

    # Initialize model
    ppe_model = YOLO(f"{config.YOLOV8_FOLDER}/{config.YOLOV8_WEIGHTS}m-ppe.pt")
    model = YOLO(weights)
    cel_model = YOLO(f"{config.YOLOV8_FOLDER}/{config.YOLOV8_WEIGHTS}m.pt")
    step_message(next(step_count), f'{Path(weights).stem.upper()} Model Initialized')

    # Initialize zones
    # polygons = load_zones(file_path="D:/Data/Harry/16_region.json")
    # zones = [
    #     sv.PolygonZone(
    #         polygon=polygon,
    #         triggering_anchors=(sv.Position.BOTTOM_CENTER,),
    #     )
    #     for polygon in polygons
    # ]
    # COLORS = sv.ColorPalette.from_hex(["#3CB44B", "#FFE119", "#3C76D1"])

    # Initialize video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened(): quit()
    source_info = from_video_path(cap)
    cap.release()
    step_message(next(step_count), 'Video Source Initialized')
    print_video_info(source, source_info)

    if show_image:
        scaled_width = 1280 if source_info.width > 1280 else source_info.width
        k = int(scaled_width * source_info.height / source_info.width)
        scaled_height = k if source_info.height > k else source_info.height

    # Annotators
    pose_annotator = PoseAnnotator(thickness=2, radius=4)
    color_annotator = sv.ColorAnnotator(color=sv.Color(r=0, g=0, b=200), opacity=0.25)
    ppe_color_annotator = sv.ColorAnnotator(color=sv.Color(r=0, g=255, b=0), opacity=0.5)
    cel_color_annotator = sv.ColorAnnotator(color=sv.Color(r=255, g=255, b=0), opacity=0.5)

    # Start video processing
    step_message(next(step_count), 'Start Video Processing')
    fvs = FileVideoStream(source)
    video_sink = sv.VideoSink(target_path=f"{output}.mp4", video_info=source_info)
    # csv_sink = CSVSink(file_name=f"{output}.csv")

    frame_number = 0
    fvs.start()
    # time.sleep(1.0)
    fps = FPS().start()
    with video_sink:
        while fvs.more():
            image = fvs.read()
            if image is None:
                print()
                break
            annotated_image = image.copy()
            
            # YOLOv8 inference
            results = model(
                source=image,
                imgsz=image_size,
                conf=confidence,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                verbose=False
            )[0]

            ppe_results = ppe_model(
                source=image,
                imgsz=image_size,
                conf=confidence,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                verbose=False
            )[0]

            cel_results = cel_model(
                source=image,
                imgsz=image_size,
                conf=confidence,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                classes=[67],
                verbose=False
            )[0]

            detections = sv.Detections.from_ultralytics(results)
            ppe_detections = sv.Detections.from_ultralytics(ppe_results)
            cel_detections = sv.Detections.from_ultralytics(cel_results)

            if show_image or save_video:
                # for idx, zone in enumerate(zones):
                #     annotated_image = sv.draw_polygon(
                #         scene=annotated_image,
                #         polygon=zone.polygon,
                #         color=COLORS.by_idx(idx)
                #     )
                # Draw poses
                annotated_image = pose_annotator.annotate(
                    scene=annotated_image,
                    ultralytics_results=results,
                    color=(0,255,255)
                )

                annotated_image = color_annotator.annotate(
                    scene=annotated_image,
                    detections=detections
                )

                annotated_image = ppe_color_annotator.annotate(
                    scene=annotated_image,
                    detections=ppe_detections
                )

                annotated_image = cel_color_annotator.annotate(
                    scene=annotated_image,
                    detections=cel_detections
                )
            
            if save_video: video_sink.write_frame(frame=annotated_image)
            
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
        output=f"{config.INPUT_FOLDER}/{Path(config.INPUT_VIDEO).stem}_output",
        weights=f"{config.YOLOV8_FOLDER}/{config.YOLOV8_WEIGHTS}x-pose.pt",
        image_size=config.IMAGE_SIZE,
        confidence=config.CONFIDENCE,
        show_image=True,
        save_video=True
    )

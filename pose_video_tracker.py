import supervision as sv

import cv2
import torch
import datetime
import itertools
from pathlib import Path

from imutils.video import FileVideoStream, WebcamVideoStream

from sinks.model_sink import ModelSink
from sinks.annotation_sink import AnnotationSink

import config
from tools.video_info import VideoInfo
from tools.messages import source_message, progress_message, step_message
from tools.general import load_zones


# For debugging
from icecream import ic


def main(
    source: str,
    output: str,
    weights: str,
    image_size: int,
    confidence: float
) -> None:
    # Initialize video source
    source_info, source_flag = VideoInfo.get_source_info(source)
    step_message(next(step_count), 'Video Source Initialized ✅')
    source_message(source, source_info)

    # Check GPU availability
    step_message(next(step_count), f"Processor: {'GPU ✅' if torch.cuda.is_available() else 'CPU ⚠️'}")

    # Initialize YOLOv8 model
    person_sink = ModelSink(weights_path=weights)
    ppe_sink = ModelSink(weights_path=f"{config.YOLOV8_FOLDER}/{config.YOLOV8_WEIGHTS}m-ppe.pt")
    # celphone_sink = ModelSink(
    #     weights_path=f"{config.YOLOV8_FOLDER}/{config.YOLOV8_WEIGHTS}m.pt",
    #     class_filter=[67]
    # )
    step_message(next(step_count), f"{Path(weights).stem.upper()} Model Initialized ✅")

    scaled_width = 1280 if source_info.width > 1280 else source_info.width
    scaled_height = int(scaled_width * source_info.height / source_info.width)
    scaled_height = scaled_height if source_info.height > scaled_height else source_info.height

    # Annotators
    person_annotation_sink = AnnotationSink(
        source_info=source_info,
        fps=False,
        label=False,
        box=False,
        colorbox=True,
        vertex=True,
        edge=True,
        color_bg=sv.Color(r=0, g=0, b=200),
        color_opacity=0.25,
    )

    ppe_annotation_sink = AnnotationSink(
        source_info=source_info,
        fps=False,
        label=False,
        box=False,
        colorbox=True,
        vertex=False,
        edge=False,
        color_bg=sv.Color(r=0, g=255, b=0),
        color_opacity=0.5,
    )

    # celphone_annotation_sink = AnnotationSink(
    #     source_info=source_info,
    #     fps=False,
    #     label=False,
    #     box=False,
    #     colorbox=True,
    #     vertex=False,
    #     edge=False,
    #     color_bg=sv.Color(r=255, g=255, b=0),
    #     color_opacity=0.5,
    # )

    # Initialize zones
    polygons = load_zones(file_path="D:/Data/Industria/4_region.json")
    zones = [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=(sv.Position.BOTTOM_CENTER,),
        )
        for polygon in polygons
    ]
    COLORS = sv.ColorPalette.from_hex(["#FFE119", "#3CB44B", "#3C76D1"])

    # Start video detection processing
    step_message(next(step_count), 'Video Detection Started ✅')

    if source_flag == 'stream':
        video_stream = WebcamVideoStream(src=eval(source) if source.isnumeric() else source)
        source_writer = cv2.VideoWriter(f"{output}_source.mp4", cv2.VideoWriter_fourcc(*'mp4v'), source_info.fps, (source_info.width, source_info.height))
    elif source_flag == 'video':
        video_stream = FileVideoStream(source)
    output_writer = cv2.VideoWriter(f"{output}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), source_info.fps, (source_info.width, source_info.height))

    frame_number = 0
    output_data = []
    video_stream.start()
    time_start = datetime.datetime.now()
    fps_monitor = sv.FPSMonitor()
    try:
        while video_stream.more() if source_flag == 'video' else True:
            fps_monitor.tick()
            fps_value = fps_monitor.fps

            image = video_stream.read()
            if image is None:
                print()
                break
            annotated_image = image.copy()
            mask_image = image.copy()

            # Inference
            person_results = person_sink.detect(image=image)
            ppe_results = ppe_sink.detect(image=image)
            # cel_results = celphone_sink.detect(image=image)
            
            # Convert results to Supervision format
            person_keypoints = sv.KeyPoints.from_ultralytics(person_results)
            person_detections = sv.Detections.from_ultralytics(person_results)
            ppe_detections = sv.Detections.from_ultralytics(ppe_results)
            # cel_detections = sv.Detections.from_ultralytics(cel_results)

            for idx, zone in enumerate(zones):
                # annotated_image = sv.draw_polygon(
                #     scene=annotated_image,
                #     polygon=zone.polygon,
                #     color=COLORS.by_idx(idx)
                # )
                annotated_image = cv2.fillPoly(
                    annotated_image,
                    [zone.polygon],
                    # isClosed=True,
                    color=COLORS.by_idx(idx).as_bgr()
                )
            annotated_image = cv2.addWeighted(
                annotated_image, 0.5, mask_image, 1 - 0.5, gamma=0
            )
            
            # Draw annotations
            annotated_image = person_annotation_sink.on_detections(
                detections=person_detections,
                scene=annotated_image
            )
            annotated_image = person_annotation_sink.on_keypoints(
                key_points=person_keypoints,
                scene=annotated_image
            )
            annotated_image = ppe_annotation_sink.on_detections(
                detections=ppe_detections,
                scene=annotated_image
            )
            # annotated_image = celphone_annotation_sink.on_detections(
            #     detections=cel_detections,
            #     scene=annotated_image
            # )
            
            # Save results
            output_writer.write(annotated_image)
            if source_flag == 'stream': source_writer.write(image)
            
            # Print progress
            progress_message(frame_number, source_info.total_frames, fps_value)
            frame_number += 1

            cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Output', int(scaled_width), int(scaled_height))
            cv2.imshow("Output", annotated_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n")
                break

    except KeyboardInterrupt:
        step_message(next(step_count), 'End of Video ✅')


    step_message(next(step_count), f"Elapsed Time: {(datetime.datetime.now() - time_start).total_seconds():.2f} s")
    output_writer.release()
    if source_flag == 'stream': source_writer.release()

    cv2.destroyAllWindows()
    video_stream.stop()


if __name__ == "__main__":
    step_count = itertools.count(1)
    main(
        source=f"{config.INPUT_FOLDER}/{config.INPUT_VIDEO}",
        output=f"{config.INPUT_FOLDER}/{Path(config.INPUT_VIDEO).stem}_output_2",
        weights=f"{config.YOLOV8_FOLDER}/{config.YOLOV8_WEIGHTS}x-pose.pt",
        image_size=config.IMAGE_SIZE,
        confidence=config.CONFIDENCE
    )

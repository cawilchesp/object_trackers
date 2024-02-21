from ultralytics import YOLO
from super_gradients.training import models
import supervision as sv

import sys
import argparse
import torch
import cv2
import time
import numpy as np
from pathlib import Path
from collections import defaultdict, deque

from tools.print_info import print_video_info, print_progress, step_message
from tools.write_csv import output_data_list, write_csv
from tools.video_info import from_camera, from_video_path

# For debugging
from icecream import ic


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


def time_synchronized():
    """ PyTorch accurate time """
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return time.time()


def process_step() -> str:
    """ Simple step counter """
    global step_count
    step_count = str(int(step_count) + 1)
    return step_count


def main():
    # Obtain parent folder to start app from any folder
    source_path = Path(__file__).resolve()
    source_dir = source_path.parent

    # Get arguments from command line
    SOURCE = option.source
    OUTPUT = option.output
    WEIGHTS = option.weights
    DEVICE = option.device
    CLASS_FILTER = option.class_filter
    IMAGE_SIZE = option.image_size
    CONFIDENCE = option.confidence
    CLIP_LENGTH = option.clip_length
    TRACK_LENGTH = option.track_length
    SHOW_IMAGE = option.show_image
    SAVE_CSV = option.save_csv
    SAVE_VIDEO = option.save_video
    SAVE_SOURCE = option.save_source
    CODEC = option.codec
    TIME_MEASURE = option.time_measure
    
    # Initialize YOLO Model
    if 'v8' in WEIGHTS:
        model = YOLO(f"{source_dir}/weights/{WEIGHTS}.pt")
        step_message(process_step(), 'YOLOv8 Model Initialized')
    elif 'nas' in WEIGHTS:
        model = models.get(model_name=WEIGHTS, pretrained_weights='coco').cuda()
        step_message(process_step(), 'YOLO-NAS Model Initialized')
    
    # Initialize Byte Tracker
    byte_tracker = sv.ByteTrack()
    step_message(process_step(), 'ByteTrack Tracker Initialized')

    # Initialize video capture
    step_message(process_step(), 'Initializing Video Source')

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened(): sys.exit()
    if SOURCE.lower().startswith('rtsp://'):
        source_info = from_camera(cap)
    else:
        source_info = from_video_path(cap)
    print_video_info(SOURCE, source_info)

    # Start video processing
    step_message(process_step(), 'Video Processing Started')

    # Annotators
    line_thickness = sv.calculate_dynamic_line_thickness(resolution_wh=(source_info.width,source_info.height))
    text_scale = sv.calculate_dynamic_text_scale(resolution_wh=(source_info.width,source_info.height))

    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_padding=2, text_position=sv.Position.TOP_LEFT, text_thickness=line_thickness)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=line_thickness)
    mask_annotator = sv.MaskAnnotator()
    trace_annotator = sv.TraceAnnotator(position=sv.Position.CENTER, trace_length=TRACK_LENGTH, thickness=line_thickness)

    polygon_zone = sv.PolygonZone(polygon=ZONE_ANALYSIS, frame_resolution_wh=(source_info.width,source_info.height))
    view_transformer = ViewTransformer(source=ZONE_ANALYSIS, target=TARGET)
    coordinates = defaultdict(lambda: deque(maxlen=int(source_info.fps)))

    t_process_start = time.time()

    progress_times = {}

    video_path = None
    video_writer = None
    video_source_writer = None
    csv_path = None
    results_data = []
    time_data = []
    frame_number = 0
    try:
        while cap.isOpened():
            t_frame_start = time_synchronized()

            # Image capture
            success, image = cap.read()
            if not success: break

            t_capture = time_synchronized()

            annotated_image = image.copy()
            annotated_image = sv.draw_polygon(scene=annotated_image, polygon=ZONE_ANALYSIS, color=sv.Color.RED)

            # YOLO inference
            if 'v8' in WEIGHTS:
                results = model(
                    source=image,
                    imgsz=IMAGE_SIZE,
                    conf=CONFIDENCE,
                    device=DEVICE,
                    agnostic_nms=True,
                    classes=CLASS_FILTER,
                    retina_masks=True,
                    verbose=False
                )
            elif 'nas' in WEIGHTS:
                results = [model.predict(
                    image,
                    conf=CONFIDENCE,
                    fuse_model=False)]

            t_inference = time_synchronized()

            # Processing inference results
            for result in results:
                if 'v8' in WEIGHTS:
                    detections = sv.Detections.from_ultralytics(result)
                elif 'nas' in WEIGHTS:
                    detections = sv.Detections.from_yolo_nas(result)
                    
                t_detections = time_synchronized()

                # Polygon zone filter
                # detections = detections[polygon_zone.trigger(detections)]

                # Update tracks
                tracks = byte_tracker.update_with_detections(detections)

                t_tracks = time_synchronized()

                points = tracks.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                points = view_transformer.transform_points(points=points).astype(int)

                # Save object data in list
                if 'v8' in WEIGHTS:
                    results_data = output_data_list(results_data, frame_number, tracks, result.names)
                elif 'nas' in WEIGHTS:
                    results_data = output_data_list(results_data, frame_number, tracks, result.class_names)

                t_savings = time_synchronized()

                if SAVE_VIDEO or SHOW_IMAGE:
                    # Draw labels
                    # if 'v8' in WEIGHTS:
                    #     object_labels = [f"{result.names[class_id]} - {tracker_id} - {score:.2f}" for _, _, score, class_id, tracker_id, _ in tracks]
                    # elif 'nas' in WEIGHTS:
                    #     object_labels = [f"{result.class_names[class_id]} - {tracker_id} - {score:.2f}" for _, _, score, class_id, tracker_id, _ in tracks]

                    object_labels =[]
                    for tracker_id, [_, y] in zip(tracks.tracker_id, points):
                        coordinates[tracker_id].append(y)
                        if len(coordinates[tracker_id]) < source_info.fps / 2:
                            object_labels.append(f"#{tracker_id}")
                        else:
                            coordinate_start = coordinates[tracker_id][-1]
                            coordinate_end = coordinates[tracker_id][0]
                            distance = abs(coordinate_start - coordinate_end)
                            time_diff = len(coordinates[tracker_id]) / source_info.fps
                            speed = distance / time_diff * 3.6
                            object_labels.append(f"#{tracker_id} {int(speed)} Km/h")

                    # object_labels = [f"x: {x}, y: {y}" for [x,y] in points]

                    annotated_image = label_annotator.annotate(
                        scene=annotated_image,
                        detections=tracks,
                        labels=object_labels )

                    # Draw boxes
                    annotated_image = bounding_box_annotator.annotate(
                        scene=annotated_image,
                        detections=tracks )
                    
                    # Draw masks
                    if tracks.mask is not None:
                        annotated_image = mask_annotator.annotate(
                        scene=annotated_image,
                        detections=detections )

                    # Draw tracks
                    annotated_image = trace_annotator.annotate(
                        scene=annotated_image,
                        detections=tracks )

            t_drawings = time_synchronized()
            
            # Save results
            if SAVE_CSV or SAVE_VIDEO or SAVE_SOURCE:
                # Output file name from source type
                if CLIP_LENGTH == 0:
                    save_path = f'{Path(OUTPUT)}'
                elif frame_number % CLIP_LENGTH == 0:
                    timestamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
                    save_path = f'{Path(OUTPUT)}_{timestamp}'
                
                # New video
                if video_path != save_path:
                    video_path = save_path
                    if SAVE_CSV:
                        if csv_path is not None and CLIP_LENGTH > 0:
                            step_message(process_step(), 'Saving Results in CSV file')
                            write_csv(f"{csv_path}.csv", results_data)
                            results_data = []
                        csv_path = save_path
                    if SAVE_VIDEO:
                        if isinstance(video_writer, cv2.VideoWriter):
                            video_writer.release()
                        video_writer = cv2.VideoWriter(f"{video_path}.avi", cv2.VideoWriter_fourcc(*CODEC), source_info.fps, (source_info.width, source_info.height))
                    if SAVE_SOURCE:
                        if isinstance(video_source_writer, cv2.VideoWriter):
                            video_source_writer.release()
                        video_source_writer = cv2.VideoWriter(f"{video_path}_source.avi", cv2.VideoWriter_fourcc(*CODEC), source_info.fps, (source_info.width, source_info.height))
                if SAVE_VIDEO: video_writer.write(annotated_image)
                if SAVE_SOURCE: video_source_writer.write(image)

            t_frame_end = time_synchronized()

            progress_times['capture_time'] = t_capture - t_frame_start
            progress_times['inference_time'] = t_inference - t_capture
            progress_times['detections_time'] = t_detections - t_inference
            progress_times['tracks_time'] = t_tracks - t_detections
            progress_times['saving_time'] = t_savings - t_tracks
            progress_times['drawings_time'] = t_drawings - t_savings
            progress_times['files_time'] = t_frame_end - t_drawings
            progress_times['frame_time'] = t_frame_end - t_frame_start

            if TIME_MEASURE:
                time_data.append([frame_number, t_capture - t_frame_start, t_inference - t_capture, t_detections - t_inference, t_tracks - t_detections, t_savings - t_tracks, t_drawings - t_savings, t_frame_end - t_drawings, t_frame_end - t_frame_start])

            # Print time (inference + NMS)
            print_progress(frame_number, source_info, progress_times)
            frame_number += 1

            # View live results
            if SHOW_IMAGE:
                cv2.namedWindow('Output', cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow("Output", 1280, 720)
                cv2.imshow('Output', annotated_image)

                # Stop if Esc key is pressed
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    except KeyboardInterrupt:
        pass

    if TIME_MEASURE:
        step_message(process_step(), 'Saving time data in CSV file')
        write_csv(f'{Path(OUTPUT)}_time_data.csv', time_data)

    # Saving data in CSV
    if SAVE_CSV:
        step_message(process_step(), 'Saving final CSV file')
        write_csv(f"{csv_path}.csv", results_data)
    
    # Print total time elapsed
    step_message('Final', f"Total Time: {(time.time() - t_process_start):.2f} s")


if __name__ == "__main__":
    # Initialize input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='video source')
    parser.add_argument('--output', type=str, required=True, help='output folder')  # output folder
    parser.add_argument('--weights', type=str, default='yolov8m', help='model.pt path(s)')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--class-filter', nargs='+', type=int, help='filter by class: --class-filter 0, or --class-filter 0 2 3')
    parser.add_argument('--image-size', type=int, default=640, help='inference size in pixels')
    parser.add_argument('--confidence', type=float, default=0.5, help='inference confidence threshold')
    parser.add_argument('--clip-length', type=int, default=0, help='clip length in frames')
    parser.add_argument('--track-length', type=int, default=64, help='track length')
    parser.add_argument('--show-image', action='store_true', help='display results')
    parser.add_argument('--save-csv', action='store_true', help='save result csv')
    parser.add_argument('--save-video', action='store_true', help='save result video')
    parser.add_argument('--save-source', action='store_true', help='save source video')
    parser.add_argument('--codec', type=str, default='mp4v', help='video codec')
    parser.add_argument('--time-measure', action='store_true', help='measure subprocess times in miliseconds')
    option = parser.parse_args()

    step_count = '0'

    # ZONE_ANALYSIS = np.array([[475,128], [859,128], [1255,673], [0,673]])
    ZONE_ANALYSIS = np.array([[530,128], [800,128], [1255,673], [0,673]])
    TARGET_WIDTH = 25
    TARGET_HEIGHT = 75
    TARGET = np.array( [ [0, 0], [TARGET_WIDTH - 1, 0], [TARGET_WIDTH - 1, TARGET_HEIGHT - 1], [0, TARGET_HEIGHT - 1] ] )
    
    main()

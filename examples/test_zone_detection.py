from ultralytics import YOLO
import supervision as sv

import cv2
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path

import numpy as np

from tools.print_info import print_video_info


from icecream import ic


COLORS = [
    '#ff2d55',
    '#0f7f07',
    '#0095ff',
    '#ffcc00',
    '#46f0f0',
    '#ff9500',
    '#d2f53c',
    '#cf52de',
]

colors_idx = sv.ColorPalette.default()

polygons = [
    np.array([
        [24, 642],
        [141, 642],
        [371, 338],
        [297, 338]
    ], np.int32), 
    np.array([
        [141, 642],
        [319, 642],
        [480, 338],
        [371, 338]
    ], np.int32), 
    np.array([
        [319, 642],
        [523, 642],
        [596, 338],
        [480, 338]
    ], np.int32),
    np.array([
        [776, 642],
        [972, 642],
        [843, 338],
        [736, 338]
    ], np.int32),
    np.array([
        [972, 642],
        [1116, 642],
        [940, 338],
        [843, 338]
    ], np.int32),
    np.array([
        [1116, 642],
        [1223, 642],
        [994, 338],
        [940, 338]
    ], np.int32)
]




def main():
    # Obtain parent folder to start app from any folder
    source_path = Path(__file__).resolve()
    source_dir = source_path.parent

    # Get arguments from command line
    weights = f'{source_dir}/weights/yolov8/yolov8m.pt'
    SOURCE_VIDEO = "C:/Users/cwilches/OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P/Dev/ruta_costera/C3_detenido.mp4"
    # SOURCE_VIDEO = "C:/Users/cwilches/OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P/Videos/Proyectos/Mexico/103 - CCTV-22 KM 83+500 - 2023-08-02 11-29-59-859.mp4"
    # SOURCE_VIDEO = "rtsp://inteia:inteiaCostera@172.17.11.2:554/trackID=1"
    source_type = 'camera' if SOURCE_VIDEO.lower().startswith('rtsp://') else 'video'
    video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO)
    print_video_info(SOURCE_VIDEO, source_type, video_info)
    



    # Zone analysis
    zones = [sv.PolygonZone(
                polygon=polygon, 
                frame_resolution_wh=video_info.resolution_wh
            ) for polygon in polygons
    ]

    zone_annotators = [
        sv.PolygonZoneAnnotator(
            zone=zone, 
            color=colors_idx.by_idx(index), 
            thickness=1,
            text_thickness=1,
            text_scale=0.3
        ) for index, zone in enumerate(zones)
    ]

    box_annotators = [sv.BoxAnnotator(
            color=colors_idx.by_idx(index), 
            thickness=1, 
            text_thickness=1, 
            text_scale=0.3,
            text_padding=5
        ) for index, _ in enumerate(polygons)
    ]

    # Initialize YOLOv8 Model
    model = YOLO(weights)

    # Initialize Byte Tracker
    byte_tracker = sv.ByteTrack()

    # Initialize CuDNN
    cudnn.benchmark = True


    class_filter = [0,1,2,3,5,7]
    total_counts = {}
    for index, zone in enumerate(zones):
        total_counts[index] = 0
    # extract video frame
    for frame in sv.get_video_frames_generator(source_path=SOURCE_VIDEO):
        # detect
        results = model(
            source=frame,
            stream=True,
            imgsz=640,
            conf=0.5,
            device=0,
            agnostic_nms=True,
            classes=class_filter,
            # retina_masks=True,
            verbose=False
        )

        for result in results:
            # Process YOLOv8 detections
            detections = sv.Detections.from_ultralytics(result)

            # Update tracks
            tracks = byte_tracker.update_with_detections(detections)

            # annotate
            labels = [f"{result.names[class_id]} - {tracker_id}" for _, _, _, class_id, tracker_id in tracks]
            
            count_index = 0
            for zone, zone_annotator, box_annotator in zip(zones, zone_annotators, box_annotators):
                # total_counts[count_index] = total_counts[count_index] + zone.current_count
                mask = zone.trigger(detections=detections)
                detections_filtered = detections[mask]
                frame = zone_annotator.annotate(scene=frame)#, label=str(total_counts[count_index]))
                frame = box_annotator.annotate(scene=frame, detections=detections_filtered, labels=labels)
                # count_index += 1


        cv2.imshow('Output', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            quit()


if __name__ == "__main__":
    with torch.no_grad():
        main()

import csv
from supervision.detection.core import Detections



def csv_detections_list(data: list, frame_number: int, detections: Detections, class_names) -> list:
    for xyxy, _, confidence, class_id, _ in detections:
        x = int(xyxy[0])
        y = int(xyxy[1])
        w = int(xyxy[2]-xyxy[0])
        h = int(xyxy[3]-xyxy[1])
        if frame_number is not None:
            data.append([frame_number, class_names[class_id], x, y, w, h, confidence])
        else:
            data.append([class_names[class_id], x, y, w, h, confidence])

    return data


def csv_tracks_list(data: list, frame_number: int, tracks, class_names) -> list:
    for xyxy, _, _, class_id, tracker_id in tracks:
        x = int(xyxy[0])
        y = int(xyxy[1])
        w = int(xyxy[2]-xyxy[0])
        h = int(xyxy[3]-xyxy[1])
        data.append([frame_number, tracker_id, class_names[class_id], x, y, w, h, None])

    return data


def write_csv(save_path: str, data: list) -> None:
    """
    Write object detection results in csv file
    """
    with open(save_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(data)
        
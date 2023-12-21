import csv
from supervision.detection.core import Detections


def output_data_list(output: list, frame_number: int, data: Detections, class_names: dict) -> list:
    for xyxy, _, confidence, class_id, tracker_id in data:
        x = int(xyxy[0])
        y = int(xyxy[1])
        w = int(xyxy[2]-xyxy[0])
        h = int(xyxy[3]-xyxy[1])
        if tracker_id is None:
            output.append([frame_number, class_names[class_id], x, y, w, h, confidence])
        else:
            output.append([frame_number, tracker_id, class_names[class_id], x, y, w, h, confidence])

    return output


def write_csv(save_path: str, data: list) -> None:
    """
    Write object detection results in csv file
    """
    with open(save_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(data)
        
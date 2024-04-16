from ultralytics import YOLO
import supervision as sv

import sys
import argparse
import torch
import cv2
import time
from pathlib import Path
import numpy as np

from tools.print_info import print_video_info, print_progress, step_message
from tools.write_csv import output_data_list, write_csv
from tools.video_info import from_camera, from_video_path
from tools.general import get_stream_frames_generator
from tools.pose_annotator import PoseAnnotator
from tools.drawings import draw_info_rectangle, line_cross, draw_line_cross, region_cross, draw_region_cross, draw_alert, draw_time_rectangle

from collections import defaultdict, deque
import pandas as pd

# from shapely.geometry import LineString, Point, Polygon


# For debugging
from icecream import ic


def time_synchronized():
    """ PyTorch accurate time """
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return time.time()


def process_step() -> str:
    """ Simple step counter """
    global step_count
    step_count = str(int(step_count) + 1)
    return step_count


def main(
        source: str,
        output: str,
        id: int,
        hand: str,
        k: float,
        direction: str
    ) -> None:
    
    # Carpeta raíz del proyecto
    root_path = Path(__file__).resolve().parent

    # Inicializar modelo YOLO Pose
    model = YOLO(f"{root_path}/weights/yolov8m-pose.pt")
    step_message(process_step(), f'Modelo YOLOv8 M Pose inicializado')

    # Inicializar captura de video
    video_source = eval(source) if source.isnumeric() else source
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened(): sys.exit()
    if source.isnumeric() or source.lower().startswith('rtsp://'):
        source_info = from_camera(cap)
        frames_generator = get_stream_frames_generator(stream_source=video_source)
    else:
        source_info = from_video_path(cap)
        frames_generator = sv.get_video_frames_generator(source_path=video_source)
    cap.release()
    step_message(process_step(), 'Origen del video inicializado')
    print_video_info(source, source_info)

    # Anotadores
    pose_annotator = PoseAnnotator(thickness=2, radius=4)
    hand_track = defaultdict(lambda: deque(maxlen=50))
    vector_track = defaultdict(lambda: deque())
    user_crossed_flag = False
    user_zone_flag = False

    # Guardar
    video_sink = sv.VideoSink(target_path=f"{output}/{id}_output.mp4", video_info=source_info, codec="mp4v")
    alert_sink = sv.VideoSink(target_path=f"{output}/{id}_alert.mp4", video_info=source_info, codec="mp4v")
    
    # Guardar origen si es streaming
    # source_sink = sv.VideoSink(target_path=f"{output}/{id}_source.mp4", video_info=source_info, codec="mp4v")



    # Iniciar procesamiento de video
    step_message(process_step(), 'Procesamiento de video iniciado')
    with video_sink, alert_sink:
        for frame_number, image in enumerate(frames_generator):
            print(f"Frame: {frame_number}")

            annotated_image = image.copy()
            image_alert = image.copy()
            image_fall = image.copy()
            image_hands_up = image.copy()
            
            annotated_image = draw_time_rectangle(
                scene=annotated_image,
                time_text=f"T: {frame_number / 10} s" )

            # YOLO inference
            results = model.track(
                source=image,
                imgsz=640,
                conf=0.5,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                agnostic_nms=True,
                retina_masks=True,
                verbose=False,
                tracker='bytetrack.yaml',
                persist=True
            )[0]

            # Draw poses
            annotated_image = pose_annotator.annotate(
                scene=annotated_image,
                ultralytics_results=results,
                # labels=True
            )
            
            # Draw hand track
            annotated_image = pose_annotator.track_point(
                scene=annotated_image,
                ultralytics_results=results,
                frame_number=frame_number,
                keypoint_track=hand_track,
                keypoint=hand )

            # Estimate speed and draw speed vector
            annotated_image = pose_annotator.speed_vector(
                scene=annotated_image,
                keypoint_track=hand_track,
                vector_track=vector_track )

            index = -1
            for track_id, vector in vector_track.items():
                if track_id == 1:
                    current_speed = vector[-1][1]
                    index = np.argmax(np.array(vector)[:,1])
                    launch_speed = vector[index][1]
                    launch_angle = vector[index][2]

                    text_speed = f"Vel: {(current_speed * k):.2f} m/s"
                    text_top_speed = f"Max: {(launch_speed * k):.2f} m/s"
                    text_angle = f"Ang: {int(launch_angle)} deg"

                    annotated_image = draw_info_rectangle(
                        scene=annotated_image,
                        texts=[text_angle, text_top_speed, text_speed, "Datos del jugador"] )
                    
            # Count points crossing line
            line_points = (270, 0, 270, 480)
            interest_keypoints = [11, 12, 13, 14, 15, 16]
            
            crossed_flag = line_cross(
                ultralytics_results=results,
                line=line_points,
                interest_keypoints=interest_keypoints,
                direction=direction )
            
            if crossed_flag: user_crossed_flag = True
            
            annotated_image = draw_line_cross(
                scene=annotated_image,
                line=line_points,
                crossed_flag=user_crossed_flag )

            # Count objects inside region
            region_points = [(390, 0), (640, 0), (640, 480), (390, 480)]
            zone_flag = region_cross(
                ultralytics_results=results,
                region=region_points )
            
            if zone_flag: user_zone_flag = True

            annotated_image = draw_region_cross(
                scene=annotated_image,
                region=region_points,
                zone_flag=zone_flag )
            
            if zone_flag: image_alert = draw_alert(image)

            if index == frame_number - 1:
                cv2.imwrite(
                    filename=f"{Path(output)}/{id}_launch.png",
                    img=annotated_image )

            # Fall detection
            image_fall, fall_flag = pose_annotator.fall_detection(
                scene=image_fall,
                ultralytics_results=results
            )

            # Hands up detection
            image_hands_up, hands_up_flag = pose_annotator.hands_up(
                scene=image_hands_up,
                ultralytics_results=results
            )

            # Guardar imagen en video
            video_sink.write_frame(frame=annotated_image)
            alert_sink.write_frame(frame=image_alert)
            
            # View live results - stop if Esc key is pressed
            cv2.imshow('Output', annotated_image)
            cv2.imshow('Alert', image_alert)
            cv2.imshow('Fall Detection', image_fall)
            cv2.imshow('Hands Up Detection', image_hands_up)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            frame_number += 1

    step_message(process_step(), 'Saving final CSV file')
    write_csv(f"{output}/{id}.csv", [[id, hand, launch_speed, launch_angle, user_crossed_flag, user_zone_flag]])
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Initialize input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--source'   , type=str, required=True, help='Origen del video')
    parser.add_argument('--output'   , type=str, required=True, help='Carpeta de resultados')
    parser.add_argument('--id'       , type=int, required=True, help='Número de ID de la persona')
    parser.add_argument('--hand'     , type=str, default='right_wrist', help='Mano de la persona: right_wrist / left_wrist')
    parser.add_argument('--k'        , type=float, default=1.0, help='Constante para estimación de la velocidad')
    parser.add_argument('--direction', type=str, default='right', help='Dirección de lanzamiento: right / left')
    option = parser.parse_args()

    step_count = '0'

    main(
        source=option.source,
        output=option.output,
        id=option.id,
        hand=option.hand,
        k=option.k,
        direction=option.direction
    )

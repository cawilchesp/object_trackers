from ultralytics import YOLO
import supervision as sv

import sys
import argparse
import torch
import cv2
import time
from pathlib import Path

from tools.print_info import print_video_info, print_progress, step_message
from tools.write_csv import output_data_list, write_csv
from tools.video_info import from_camera
from tools.general import get_stream_frames_generator
from tools.timers import ClockBasedTimer


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
        weights: str,
        class_filter: list[int],
        image_size: int,
        confidence: float,
        clip_length: int,
        track_length: int,
        show_image: bool,
        save_csv: bool,
        save_video: bool,
        save_source: bool,
        codec: str,
        time_measure: bool
    ) -> None:
    
    # Carpeta raíz del proyecto
    root_path = Path(__file__).resolve().parent

    # Inicializar modelo YOLO
    model = YOLO(f"{root_path}/weights/{weights}.pt")
    step_message(process_step(), f'Modelo {weights.upper()} inicializado')

    # Inicializar captura de video
    stream_source = eval(source) if source.isnumeric() else source
    cap = cv2.VideoCapture(stream_source)
    if not cap.isOpened(): sys.exit()
    source_info = from_camera(cap)
    step_message(process_step(), 'Origen del video inicializado')
    print_video_info(source, source_info)

    # Anotatores
    line_thickness = int(sv.calculate_dynamic_line_thickness(resolution_wh=(source_info.width, source_info.height)) * 0.5)
    text_scale = sv.calculate_dynamic_text_scale(resolution_wh=(source_info.width, source_info.height)) * 0.5

    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_padding=2, text_position=sv.Position.TOP_LEFT, text_thickness=line_thickness)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=line_thickness)
    mask_annotator = sv.MaskAnnotator()
    trace_annotator = sv.TraceAnnotator(position=sv.Position.CENTER, trace_length=track_length, thickness=line_thickness)

    # Inicializar variables
    video_path = None
    video_writer = None
    video_source_writer = None
    csv_path = None
    
    time_data = []

    # Iniciar procesamiento de video
    step_message(process_step(), 'Procesamiento de video iniciado')
    t_process_start = time.time()

    frames_generator = get_stream_frames_generator(stream_source=stream_source)
    fps_monitor = sv.FPSMonitor()

    for frame_number, image in enumerate(frames_generator):
        fps_monitor.tick()
        fps = fps_monitor.fps

        annotated_image = image.copy()

        # Inferencia de YOLO con seguidor
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
        detections = sv.Detections.from_ultralytics(results)
        detections = detections.with_nms()

        # Dibujar FPS
        annotated_image = sv.draw_text(
            scene=annotated_image,
            text=f"FPS: {fps:.1f}",
            text_anchor=sv.Point(40, 30),
            background_color=sv.Color.from_hex("#FFFFFF"),
            text_color=sv.Color.from_hex("#000000"),
        )
            
        results_data = [[
            frame_number,
            tracker_id,
            data['class_name'],
            int(xyxy[0]),
            int(xyxy[1]),
            int(xyxy[2]-xyxy[0]),
            int(xyxy[3]-xyxy[1]),
            score ] for xyxy, _, score, _, tracker_id, data in detections]
        
        if show_image or save_video:
            # Dibujar etiquetas
            object_labels = [f"{data['class_name']} {tracker_id} ({score:.2f})" for _, _, score, _, tracker_id, data in detections]
            annotated_image = label_annotator.annotate(
                scene=annotated_image,
                detections=detections,
                labels=object_labels )

            # Draw boxes
            annotated_image = bounding_box_annotator.annotate(
                scene=annotated_image,
                detections=detections )
            
            # Draw masks
            annotated_image = mask_annotator.annotate(
                scene=annotated_image,
                detections=detections )
            
            # Draw tracks
            annotated_image = trace_annotator.annotate(
                scene=annotated_image,
                detections=detections )
            
        # Save results
        if save_csv or save_video or save_source:
            # Output file name from source type
            if clip_length == 0:
                save_path = f'{Path(output)}'
            elif frame_number % clip_length == 0:
                timestamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
                save_path = f'{Path(output)}_{timestamp}'
            
            # New video
            if video_path != save_path:
                
                
                video_path = save_path
                
                
                if save_csv:
                    if csv_path is not None and clip_length > 0:
                        step_message(process_step(), 'Saving Results in CSV file')
                        write_csv(f"{csv_path}.csv", results_data)
                        results_data = []
                    csv_path = save_path
                

                if save_video:
                    if isinstance(video_writer, cv2.VideoWriter):
                        video_writer.release()
                    video_writer = cv2.VideoWriter(f"{video_path}.avi", cv2.VideoWriter_fourcc(*codec), source_info.fps, (source_info.width, source_info.height))
                

                if save_source:
                    if isinstance(video_source_writer, cv2.VideoWriter):
                        video_source_writer.release()
                    video_source_writer = cv2.VideoWriter(f"{video_path}_source.avi", cv2.VideoWriter_fourcc(*codec), source_info.fps, (source_info.width, source_info.height))
            
            
            
            if save_video: video_writer.write(annotated_image)
            if save_source: video_source_writer.write(image)

        if show_image:
            cv2.imshow("Resultado", annotated_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


    # try:
    #     while cap.isOpened():
    #         t_frame_start = time_synchronized()

    #         # Image capture
    #         success, image = cap.read()
    #         if not success: break

    #         t_capture = time_synchronized()

    #         annotated_image = image.copy()

    #         # YOLO inference
    #         results = model(
    #             source=image,
    #             imgsz=IMAGE_SIZE,
    #             conf=CONFIDENCE,
    #             device=DEVICE,
    #             agnostic_nms=True,
    #             classes=CLASS_FILTER,
    #             retina_masks=True,
    #             verbose=False
    #         )[0]

    #         t_inference = time_synchronized()

    #         # Processing inference results
    #         for result in results:
    #             detections = sv.Detections.from_ultralytics(result)
                    
    #             t_detections = time_synchronized()

    #             # Update tracks
    #             tracks = byte_tracker.update_with_detections(detections)

    #             t_tracks = time_synchronized()

    #             # Save object data in list
    #             if 'v8' in WEIGHTS or 'v9' in WEIGHTS:
    #                 results_data = output_data_list(results_data, frame_number, tracks, result.names)
    #             elif 'nas' in WEIGHTS:
    #                 results_data = output_data_list(results_data, frame_number, tracks, result.class_names)

    #             t_savings = time_synchronized()

    #         t_drawings = time_synchronized()
            
    #         # Save results
    #         if SAVE_CSV or SAVE_VIDEO or SAVE_SOURCE:
    #             # Output file name from source type
    #             if CLIP_LENGTH == 0:
    #                 save_path = f'{Path(OUTPUT)}'
    #             elif frame_number % CLIP_LENGTH == 0:
    #                 timestamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    #                 save_path = f'{Path(OUTPUT)}_{timestamp}'
                
    #             # New video
    #             if video_path != save_path:
    #                 video_path = save_path
    #                 if SAVE_CSV:
    #                     if csv_path is not None and CLIP_LENGTH > 0:
    #                         step_message(process_step(), 'Saving Results in CSV file')
    #                         write_csv(f"{csv_path}.csv", results_data)
    #                         results_data = []
    #                     csv_path = save_path
    #                 if SAVE_VIDEO:
    #                     if isinstance(video_writer, cv2.VideoWriter):
    #                         video_writer.release()
    #                     video_writer = cv2.VideoWriter(f"{video_path}.avi", cv2.VideoWriter_fourcc(*CODEC), source_info.fps, (source_info.width, source_info.height))
    #                 if SAVE_SOURCE:
    #                     if isinstance(video_source_writer, cv2.VideoWriter):
    #                         video_source_writer.release()
    #                     video_source_writer = cv2.VideoWriter(f"{video_path}_source.avi", cv2.VideoWriter_fourcc(*CODEC), source_info.fps, (source_info.width, source_info.height))
    #             if SAVE_VIDEO: video_writer.write(annotated_image)
    #             if SAVE_SOURCE: video_source_writer.write(image)

    #         t_frame_end = time_synchronized()

    #         progress_times = {
    #             'capture_time': t_capture - t_frame_start,
    #             'inference_time': t_inference - t_capture,
    #             'detections_time': t_detections - t_inference,
    #             'tracks_time': t_tracks - t_detections,
    #             'saving_time': t_savings - t_tracks,
    #             'drawings_time': t_drawings - t_savings,
    #             'files_time': t_frame_end - t_drawings,
    #             'frame_time': t_frame_end - t_frame_start }

    #         if TIME_MEASURE:
    #             time_data.append([frame_number, t_capture - t_frame_start, t_inference - t_capture, t_detections - t_inference, t_tracks - t_detections, t_savings - t_tracks, t_drawings - t_savings, t_frame_end - t_drawings, t_frame_end - t_frame_start])

    #         # Print time (inference + NMS)
    #         print_progress(frame_number, source_info, progress_times)
    #         frame_number += 1

    #         # View live results
    #         if SHOW_IMAGE:
    #             cv2.namedWindow('Output', cv2.WINDOW_KEEPRATIO)
    #             cv2.resizeWindow("Output", 1280, 720)
    #             cv2.imshow('Output', annotated_image)

    #             # Stop if Esc key is pressed
    #             if cv2.waitKey(1) & 0xFF == 27:
    #                 break

    # except KeyboardInterrupt:
    #     pass

    # if TIME_MEASURE:
    #     step_message(process_step(), 'Saving time data in CSV file')
    #     write_csv(f'{Path(OUTPUT)}_time_data.csv', time_data)

    # # Saving data in CSV
    # if SAVE_CSV:
    #     step_message(process_step(), 'Saving final CSV file')
    #     write_csv(f"{csv_path}.csv", results_data)
    
    # Print total time elapsed
    step_message('Final', f"Total Time: {(time.time() - t_process_start):.2f} s")


if __name__ == "__main__":
    # Initialize input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Origen del video: cámara web o transmisión RTSP')
    parser.add_argument('--output', type=str, required=True, help='Carpeta de resultados')
    parser.add_argument('--weights', type=str, default='yolov8m', help='Nombre del modelo YOLOv8 o YOLOv9')
    parser.add_argument('--class-filter', nargs='+', type=int, help='Filtra detecciones por clase (i.e. 0 o 0 2 3')
    parser.add_argument('--image-size', type=int, default=640, help='Tamaño de imagen de inferencia')
    parser.add_argument('--confidence', type=float, default=0.5, help='Umbral de confianza de inferencia')
    parser.add_argument('--clip-length', type=int, default=0, help='Número de cuadros por clip de video')
    parser.add_argument('--track-length', type=int, default=50, help='Número de puntos de trayectoria')
    parser.add_argument('--show-image', action='store_true', help='Mostrar resultados')
    parser.add_argument('--save-csv', action='store_true', help='Guardar resultados en CSV')
    parser.add_argument('--save-video', action='store_true', help='Guardar video de resultados')
    parser.add_argument('--save-source', action='store_true', help='Guardar video de origen')
    parser.add_argument('--time-measure', action='store_true', help='Medir tiempos de subprocesos en milisegundos')
    options = parser.parse_args()

    step_count = '0'

    main(
        source = options.source,
        output = options.output,
        weights = options.weights,
        class_filter = options.class_filter,
        image_size = options.image_size,
        confidence = options.confidence,
        clip_length = options.clip_length,
        track_length = options.track_length,
        show_image = options.show_image,
        save_csv = options.save_csv,
        save_video = options.save_video,
        save_source = options.save_source,
        codec = options.codec,
        time_measure = options.time_measure
    )

import supervision as sv

from pathlib import Path
import pandas as pd
import numpy as np
import cv2


from tools.speed import ViewTransformer

from collections import defaultdict, deque

from icecream import ic
COLORS = {
    'person': (255,0,0),
    'car': (0,170,0),
    'bicycle': (0,0,255),
    'motorbike': (0,255,255),
    'bus': (255,0,255),
    'truck': (255,255,255)
}

def combine_csv_files(camera_id, day, hour):
    csv_path = f"D:/Data/Piloto_EDGE/CCT_{camera_id}_{day}{hour}"
    csv_file_list = list(sorted(Path(csv_path).glob(f"C{camera_id}*.csv")))

    print('Combining CSV')
    combined_csv = pd.concat( [ pd.read_csv(csv_file, header=None) for csv_file in csv_file_list ] )

    print('Saving output file')
    combined_csv.to_csv(f"{csv_path}/{camera_id}.csv", index=False)

    print('End')


def process_csv_file(camera_id, day, hour):
    # Speed estimation
    ZONE_ANALYSIS = np.array([[279,64], [406,64], [635,338], [0,338]])
    TARGET_WIDTH = 25
    TARGET_HEIGHT = 120
    TARGET = np.array( [ [0, 0], [TARGET_WIDTH - 1, 0], [TARGET_WIDTH - 1, TARGET_HEIGHT - 1], [0, TARGET_HEIGHT - 1] ] )
    view_transformer = ViewTransformer(source=ZONE_ANALYSIS, target=TARGET)
    
    
    # Anotatores
    line_thickness = int(sv.calculate_dynamic_line_thickness(resolution_wh=(704, 480)) * 0.5)
    text_scale = sv.calculate_dynamic_text_scale(resolution_wh=(704, 480)) * 0.5

    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_padding=2, text_position=sv.Position.TOP_LEFT, text_thickness=line_thickness)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=line_thickness)
    trace_annotator = sv.TraceAnnotator(position=sv.Position.CENTER, trace_length=50, thickness=line_thickness)

    # Line counters
    line_zone_1 = sv.LineZone(start=sv.Point(x=65, y=220), end=sv.Point(x=118, y=220))
    line_zone_2 = sv.LineZone(start=sv.Point(x=118, y=220), end=sv.Point(x=173, y=220))
    
    id_classes = {
        'bicycle': 0,
        'bus': 1,
        'car': 2,
        'motorbike': 3,
        'person': 4,
        'truck': 5 
    }


    # Load CSV data
    csv_file = f"D:/Data/Piloto_EDGE/CCT_{camera_id}_{day}{hour}/{camera_id}.csv"
    csv_data = pd.read_csv(csv_file, sep=',', names=['frame', 'id', 'class', 'x', 'y', 'w', 'h', 'score'], header=None, index_col=False)
    csv_data['x2'] = csv_data['x'] + csv_data['w']
    csv_data['y2'] = csv_data['y'] + csv_data['h']
    csv_data['class_id'] = csv_data['class'].replace(id_classes)
    
    max_frame = csv_data['frame'].max()
    
    # Variables
    time_step = 1.0 / 10.0

    frame_number = 0
    while frame_number < max_frame:
        annotated_image = np.zeros([480, 704, 3], np.uint8)
        annotated_image = sv.draw_polygon(scene=annotated_image, polygon=ZONE_ANALYSIS, color=sv.Color.RED)
        
        detections_dataframe = csv_data[csv_data['frame'] == frame_number].copy()
        
        detections = sv.Detections(
            xyxy=detections_dataframe[['x', 'y', 'x2', 'y2']].to_numpy().astype(np.float32),
            mask=None,
            confidence=detections_dataframe[['score']].to_numpy().ravel(),
            class_id=detections_dataframe['class_id'].to_numpy().ravel(),
            tracker_id=detections_dataframe['id'].to_numpy().ravel(),
            data={'class_name': detections_dataframe['class'].to_numpy().astype('<U10')}
        )

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
        
        # Draw tracks
        if detections.tracker_id is not None:
            annotated_image = trace_annotator.annotate(
                scene=annotated_image,
                detections=detections )
        
        frame_number += 1

        annotated_image = sv.draw_line(scene=annotated_image, start=sv.Point(x=65, y=220), end=sv.Point(x=118, y=220), color=sv.Color.BLUE)
        annotated_image = sv.draw_line(scene=annotated_image, start=sv.Point(x=118, y=220), end=sv.Point(x=173, y=220), color=sv.Color.GREEN)
        
        cv2.imshow("Resultado", annotated_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


def main():
    camera_id = 7402
    day = 'M'
    hour = 14
    
    # combine_csv_files(camera_id, day, hour)
    process_csv_file(camera_id, day, hour)


if __name__ == "__main__":
    main()
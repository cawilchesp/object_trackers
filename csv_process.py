import supervision as sv

from pathlib import Path
import pandas as pd
import numpy as np
import cv2

from tools.speed import ViewTransformer

from icecream import ic
COLORS = {
    'person': (255,0,0),
    'car': (0,170,0),
    'bicycle': (0,0,255),
    'motorbike': (0,255,255),
    'bus': (255,0,255),
    'truck': (255,255,255)
}

ID_CLASSES = {
    'bicycle': 0,
    'bus': 1,
    'car': 2,
    'motorbike': 3,
    'person': 4,
    'truck': 5 
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
    # Load CSV data
    csv_file = f"D:/Data/Piloto_EDGE/CCT_{camera_id}_{day}{hour}/{camera_id}.csv"
    csv_data = pd.read_csv(csv_file, sep=',', names=['frame', 'id', 'class', 'x', 'y', 'w', 'h', 'score'], header=None, index_col=False)
    object_id_list = csv_data['id'].unique().astype(int)
    object_dict = {
        'id': [],
        'class_id': [],
        'class': [],
        'speed': [],
        'lane': [],
        'trajectory': []
    }

    # Processing data
    csv_data['class_id'] = csv_data['class'].replace(ID_CLASSES)
    csv_data['x2'] = csv_data['x'] + csv_data['w']
    csv_data['y2'] = csv_data['y'] + csv_data['h']
    csv_data['time_frame'] = csv_data['frame'] * (1.0 / 10.0)
    csv_data['center_x'] = csv_data['x'] + (csv_data['w'] / 2)
    csv_data['center_y'] = csv_data['y'] + csv_data['h']

    # Space transformation
    ZONE_ANALYSIS = np.array([[279,64], [406,64], [635,338], [0,338]])
    TARGET_WIDTH = 2500
    TARGET_HEIGHT = 9000
    TARGET = np.array( [ [0, 0], [TARGET_WIDTH - 1, 0], [TARGET_WIDTH - 1, TARGET_HEIGHT - 1], [0, TARGET_HEIGHT - 1] ] )
    view_transformer = ViewTransformer(source=ZONE_ANALYSIS, target=TARGET)

    points = csv_data[['center_x', 'center_y']].to_numpy()
    points_transformed = view_transformer.transform_points(points=points).astype(int)
    points_transformed = pd.DataFrame(points_transformed, columns=['x_transformed', 'y_transformed'])
    csv_data = pd.concat([csv_data, points_transformed], axis = 1)

    # Lane
    csv_data.loc[(csv_data['x_transformed'] > -331) & (csv_data['x_transformed'] < 0) & (csv_data['y_transformed'] > 3000) & (csv_data['y_transformed'] < 8500), 'lane'] = 1
    csv_data.loc[(csv_data['x_transformed'] > 0) & (csv_data['x_transformed'] < 317) & (csv_data['y_transformed'] > 3000) & (csv_data['y_transformed'] < 8500), 'lane'] = 2
    csv_data.loc[(csv_data['x_transformed'] > 317) & (csv_data['x_transformed'] < 647) & (csv_data['y_transformed'] > 3000) & (csv_data['y_transformed'] < 8500), 'lane'] = 3
    csv_data.loc[(csv_data['x_transformed'] > 647) & (csv_data['x_transformed'] < 987) & (csv_data['y_transformed'] > 3000) & (csv_data['y_transformed'] < 8500), 'lane'] = 4
    csv_data.loc[(csv_data['x_transformed'] > 1367) & (csv_data['x_transformed'] < 1734) & (csv_data['y_transformed'] > 3000) & (csv_data['y_transformed'] < 8500), 'lane'] = 5
    csv_data.loc[(csv_data['x_transformed'] > 1734) & (csv_data['x_transformed'] < 2118) & (csv_data['y_transformed'] > 3000) & (csv_data['y_transformed'] < 8500), 'lane'] = 6
    csv_data.loc[(csv_data['x_transformed'] > 2118) & (csv_data['x_transformed'] < 2496) & (csv_data['y_transformed'] > 3000) & (csv_data['y_transformed'] < 8500), 'lane'] = 7

    for object_number, object in enumerate(object_id_list):
        print(f"{object_number}: {object}")

        object_data = csv_data[csv_data['id'] == object].copy()
        
        # Speed Estimation
        average_speed = np.nan
        if len(object_data) > 1:
            previous_time = 0.0
            previous_x = 0
            previous_y = 0
            speed_column = []
            for current_time, current_x, current_y in zip(object_data['time_frame'], object_data['x_transformed'], object_data['y_transformed']):
                distance = abs(np.sqrt((current_y-previous_y)**2 + (current_x-previous_x)**2)) / 100
                time_diff = current_time - previous_time
                speed = distance / time_diff * 3.6 if time_diff != 0.0 else 0.0
                speed_column.append(speed)
                previous_time = current_time
                previous_x = current_x
                previous_y = current_y
            object_data['speed'] = speed_column
            average_speed = object_data['speed'][1:].mean()
        
        object_class_id = object_data['class_id'].unique()
        object_class = object_data['class'].unique()
        object_lanes = object_data['lane'].unique().astype(int)
        object_trajectory = object_data[['center_x', 'center_y']].to_numpy()

        if not np.isnan(average_speed) and object_lanes[-1] > 0:
            object_dict['id'].append(object)
            object_dict['class_id'].append(object_class_id[-1])
            object_dict['class'].append(object_class[-1])
            object_dict['speed'].append(average_speed)
            object_dict['lane'].append(object_lanes[-1])
            object_dict['trajectory'].append(object_trajectory)

    pd_object_dict = pd.DataFrame(object_dict)
    pd_object_dict.to_csv(f"D:/Data/Piloto_EDGE/CCT_{camera_id}_{day}{hour}/{camera_id}_processed.csv", index=False)
    pd_object_dict.to_json(f"D:/Data/Piloto_EDGE/CCT_{camera_id}_{day}{hour}/{camera_id}_processed.json")
    
def analysis(camera_id, day, hour):
    # Load CSV data
    json_file = f"D:/Data/Piloto_EDGE/CCT_{camera_id}_{day}{hour}/{camera_id}_processed.json"
    json_data = pd.read_json(json_file)
    
    # Processing for counting
    annotated_image = np.zeros([480, 704, 3], np.uint8)

    for index, object in json_data.iterrows():
        print(f"Objeto: {index}")
        
        object_id = object['id']
        object_class_id = object['class_id']
        object_class = object['class']
        object_speed = object['speed']
        object_lane = object['lane']

        if object_class == 'truck':
            object_trajectory = np.array(object['trajectory'], np.int32)
            object_trajectory = object_trajectory.reshape((-1, 1, 2))
            cv2.polylines(
                img=annotated_image,
                pts=[object_trajectory],
                isClosed=False,
                color=COLORS[object_class],
                thickness=1,
                lineType=cv2.LINE_AA )
        
    ZONE_ANALYSIS = np.array([[279,64], [406,64], [635,338], [0,338]])
    annotated_image = sv.draw_polygon(scene=annotated_image, polygon=ZONE_ANALYSIS, color=sv.Color.RED)

    cv2.imshow("Resultado", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def video_reconstruction():
    # Anotatores
    line_thickness = int(sv.calculate_dynamic_line_thickness(resolution_wh=(704, 480)) * 0.5)
    text_scale = sv.calculate_dynamic_text_scale(resolution_wh=(704, 480)) * 0.5

    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_padding=2, text_position=sv.Position.TOP_LEFT, text_thickness=line_thickness)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=line_thickness)
    trace_annotator = sv.TraceAnnotator(position=sv.Position.CENTER, trace_length=50, thickness=line_thickness)
    
    max_frame = csv_data['frame'].max()
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


    cv2.imshow("Resultado", annotated_image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


def main():
    camera_id = 7402
    day = 'J'
    hour = 10
    
    # combine_csv_files(camera_id, day, hour)
    process_csv_file(camera_id, day, hour)
    analysis(camera_id, day, hour)


if __name__ == "__main__":
    main()
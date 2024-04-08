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
    ZONE_ANALYSIS = np.array([[279,64], [406,64], [635,338], [0,338]])
    TARGET_WIDTH = 25
    TARGET_HEIGHT = 100
    TARGET = np.array( [ [0, 0], [TARGET_WIDTH - 1, 0], [TARGET_WIDTH - 1, TARGET_HEIGHT - 1], [0, TARGET_HEIGHT - 1] ] )

    view_transformer = ViewTransformer(source=ZONE_ANALYSIS, target=TARGET)

    csv_file = f"D:/Data/Piloto_EDGE/CCT_{camera_id}_{day}{hour}/{camera_id}.csv"
    time_step = 1.0 / 10.0

    image = np.zeros([480, 704, 3], np.uint8)
    image = sv.draw_polygon(scene=image, polygon=ZONE_ANALYSIS, color=sv.Color.RED)

    data = pd.read_csv(csv_file, sep=',', names=['frame', 'id', 'class', 'x', 'y', 'w', 'h', 'score'], header=None, index_col=False)
    
    object_id_list = data['id'].unique().astype(int)

    object_number = 0
    for object in object_id_list:
        print(object_number)
        object_data = data[data['id'] == object].copy()
        object_data.reset_index(drop=True, inplace=True)
        object_data['time_frame'] = object_data['frame'] * time_step
        object_data['center_x'] = object_data['x'] + (object_data['w'] / 2)
        object_data['center_y'] = object_data['y'] + object_data['h']

        points = object_data[['center_x', 'center_y']].to_numpy()
        points = view_transformer.transform_points(points=points).astype(int)
        points = pd.DataFrame(points, columns=['x_transformed', 'y_transformed'])

        object_data = pd.concat([object_data, points], axis = 1)

        previous_time = 0.0
        previous_x = 0
        previous_y = 0
        speed_column = []
        for current_time, current_x, current_y in zip(object_data['time_frame'], object_data['x_transformed'], object_data['y_transformed']):
            distance = abs(np.sqrt((current_y-previous_y)**2 + (current_x-previous_x)**2))
            time_diff = current_time - previous_time

            speed = distance / time_diff * 3.6 if time_diff != 0.0 else 0.0
            speed_column.append(speed)
            previous_time = current_time
            previous_x = current_x
            previous_y = current_y

        object_data['speed'] = speed_column

        object_class = object_data['class'][0]
        
        object_x = object_data['center_x'].to_list()
        object_y = object_data['center_y'].to_list()
        object_speed = object_data['speed'].to_list()[1:]

        point_list = [(int(x), int(y)) for x, y in zip(object_x, object_y)]
        point_array = np.array(point_list)
        point_array = point_array.reshape((-1, 1, 2))
        
        image = cv2.polylines(
            img=image,
            pts=[point_array],
            isClosed=False,
            color=COLORS[object_class],
            thickness=1 )
        
        object_number += 1
        
    cv2.imshow("Resultado", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    


def main():
    camera_id = 7402
    day = 'M'
    hour = 14
    
    # combine_csv_files(camera_id, day, hour)
    process_csv_file(camera_id, day, hour)


if __name__ == "__main__":
    main()
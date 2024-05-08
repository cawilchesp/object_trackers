
from subprocess import PIPE, Popen, STDOUT
from pathlib import Path
import pandas as pd
import numpy as np
import shlex
import json
import cv2
import re
import io

def video_metadata(video_path: str):
    # Se usa `ffprobe` del proyecto *ffmpeg* para la consulta de metadatos del
    # video.
    ffprobe_process = Popen(
        shlex.split("ffprobe -select_streams v -show_streams '{}'".format(video_path)),
        stdout=PIPE, stderr=PIPE, encoding="utf-8")
    # En la salida estándar se generan los metadatos de la señal de video.
    # En el error estándar se generan los metadatos globales.
    msg_out, msg_err = ffprobe_process.communicate()
    # Se extraen metadatos de la señal de video
    width = re.findall("\nwidth=(\d+)", msg_out)[0]
    height = re.findall("\nheight=(\d+)", msg_out)[0]
    frames = re.findall("nb_frames=(\d+)", msg_out)[0]
    duration = re.findall("duration=(\d+\.\d+)", msg_out)[0]
    fps = re.findall(", (\d+(\.\d+)?) fps,", msg_err)[0][0]
    original = True if re.findall("DJI", msg_out) \
        else False  # Si es el video generado directamente por el dron
    meta = {'width': int(width), 'height': int(height), 'frames': int(frames),
            'fps': float(fps), 'duration': float(duration)}
    if original:
        # El formato de las coordenadas en los metadatos globales:
        # `location        : +6.255565-75.613288+59.500`
        format_coordinate = "[\+-]\d+\.\d+"
        format_location = "location\s+:\s+({0})({0})({0})".format(format_coordinate)
        lat, lon, alt = re.findall(format_location, msg_err)[0]
        meta.update({'lat': float(lat), 'lon': float(lon), 'alt': float(alt)})
    return meta


def get_px_size(h: float,
                res_x: int,
                res_y: int,
                d_x: float = 13.2,
                d_y: float = 8.8,
                f: float = 8.8):
    """Determina el tamaño en metros de un pixel.

    Longitud focal de la cámara.
    Parámetros intrínsecos para la cámara de un Phantom DJI 4 Pro basados en el
    modelo lineal de agujero de cámara (*camera pinhole model*).

    Corresponden respectivamente al delta X, delta Y y longitud focal de la cámara.

    .. warning::
        Se debe validar el cálculo de los parámetros y reportarlo.

    .. note::
        Para saber más:

        + `Camera models and parameters <http://ftp.cs.toronto.edu/pub/psala/VM/camera-parameters.pdf>`_.
        + `Pinhole camera model <https://en.wikipedia.org/wiki/Pinhole_camera_model>`_.
        + `Camera resectioning <https://en.wikipedia.org/wiki/Camera_resectioning>`_.
        + `Intrinsec camera parameters <https://www.coursera.org/lecture/robotics-perception/intrinsic-camera-parameter-6IISZ>`_.

    D_X: float = 13.2  #: float: Delta X de la cámara. Ver :const:`F`.
    D_Y: float = 8.8  #: float: Delta Y de la cámara. Ver :const:`F`.
    F: float = 8.8

    Parameters
    ----------
    h : float
        Altura de vuelo.
    res_x : int
    res_y : int
        Resolución en Y.
    d_x : float
        Factor de escala intrínseco en X.
    d_y : float
        Factor de escala intrínseco en Y.
    f : float
        Longitud focal.
    Return
    -------
    Tuple[float, float]
        Tamaño en metros de un pixel en dirección X y Y respectivamente.
    """
    # Determining the angle of view
    alpha_x = 2 * np.arctan(d_x / (2 * f))
    alpha_y = 2 * np.arctan(d_y / (2 * f))
    # Determining distances
    x = 2 * h * np.tan(alpha_x / 2)
    y = 2 * h * np.tan(alpha_y / 2)
    # Determining the MMP
    mu_x = x / res_x
    mu_y = y / res_y
    return mu_x, mu_y

def read_d33p(csv_path, video_path, h=100) -> pd.DataFrame:
    # Nombre del video debe ser 01SAB_CARRERA43_CALLE48Y49_20201016_1226.MP4
    meta_video = video_metadata(video_path)
    fps = meta_video['fps']
    duration = meta_video['duration']
    duration = f"00:{int(duration/60):02}:{int(duration%60):02}"
    period, intersection, date, time = "","","",""
    # Obtener period, intersection, date, time del nombre del video
    video_info = re.match(
        "\d{2,}([A-Z]{2,3})_(\w+)_(\d{8,})_(\d{4,})", 
        Path(video_path).name)
    if video_info:
        period, intersection, date, time = video_info.groups()

    date = f"{date[:4]}-{date[4:6]}-{date[-2:]}" #'%Y-%m-%d'
    time = f"{time[:2]}:{time[-2:]}:00"
    h = meta_video['alt'] if 'alt' in meta_video.keys() else h

    print('altura', h, 'alt' in meta_video.keys())

    x, y = get_px_size(h, meta_video["width"], meta_video["height"])
    df = pd.read_csv(csv_path, index_col=False, header=None)
    df.columns = ['t', 'id', 'type', 'x', 'y', 'w', 'h', 'p', '_']
    # scale poitns
    df['x']=df['x']*x
    df['y']=df['y']*y
    df['w']=df['w']*x
    df['h']=df['h']*y
    # x2 point
    df['x2']=df['x'] + df['w']
    # y2 point
    df['y2']=df['y'] + df['h']
    # Center of the bounding box in x
    df['cx']= df['x'] + df['w'] * 0.5
    # Center of the bounding box in y
    df['cy']= df['y'] + df['h'] * 0.5
    # get time
    df['t']=df['t']/fps

    names_metvial= {
        'car': 'Car', 
        'motorcycle': 'Motorbike', 
        'bus': 'Bus', 
        'person': 'Person',
        'truck': 'Truck',
        'bike': 'Bicycle'
    }

    trajectories = []
    for id_traj in df['id'].unique():
        traj = df[df['id'] == id_traj]
        traj_data = {
            "id": id_traj,
            "category": names_metvial[traj['type'].unique()[0]],
            "trajectory": traj[['cx', 'cy', 't']].round(2).to_numpy(),
            "bbox": traj[['x', 'y', 'x2', 'y2']].round(2).to_numpy(),
        }
        trajectories.append(traj_data)

    return pd.Series({
        'trajectory': pd.DataFrame(trajectories),
        'fligth_meta': {
            "pixel_size": [x, y],
            "duration": duration,
            "time": time,
            "date": date,
            "period": period
        }
    })

def assing_key(dict_out, key_mov, value):
    dict_len=dict() # sufijo: [prefijos]
    # revisa cual es el prefijo en dict_out 
    for val in dict_out.keys():
        key_len, count = map(int, val.split('_'))
        if key_len in dict_len:
            dict_len[key_len].append(count)
        else:
            dict_len[key_len]=[count]
            
    # Agrega nueva key con prefijo incremental
    if key_mov in dict_len:
        dict_out[f'{key_mov}_{str(max(dict_len[key_mov]) + 1)}'] = value
    else:
        dict_out[f'{key_mov}_0'] = value
    return dict_out


def create_gzip_(params):
    """Actualiza el archivo gzip tal cómo lo recibe metvial para
    procesar metricas"""

    # path video
    video_path = Path(params['video_path'])
    # path gzip de salida
    gzip_path = Path(str(video_path).replace(video_path.suffix, '.gzip'))
    # trayectorias
    csv_path = video_path.parents[0].joinpath('out_video.csv')
    # Imagen de pantallazo
    jpg_path = Path(str(video_path).replace(video_path.suffix, '.jpg'))
    serie = read_d33p(csv_path, video_path, params['h'])
    
    vidcap = cv2.VideoCapture(str(video_path))
    success,image = vidcap.read()
    cv2.imwrite(str(jpg_path), image)
    vidcap.release()

    data = json.load(open(params['json_network']))
    json_network_data = json.dumps(data, indent=2).encode('utf-8')


    mov_traffic={}
    for mov_tr in params['mov_traffic']:
        mov_traffic = assing_key(mov_traffic, *mov_tr) # 1, [[17, 18],['N', 'S']])


    byte_img = io.open(jpg_path, "rb", buffering = 0).read()
    serie['geom_bytes'] = json_network_data
    serie['img_veh_bytes'] = byte_img
    serie['img_ped_bytes'] = byte_img


    period_keys = {'AM':'Mañana','PM':'Tarde','VA':'Periodo valle','SAB':'Sábado'}
    network = {'passage': '', 
        'intersection': '', 
        'img_mov_veh': jpg_path.as_posix(), 
        'img_mov_ped': jpg_path.as_posix(), 
        'geom_file': params['json_network'], 
        'equivalence_vehicle': {
            'car': 1.0, 
            'motor': 0.3, 
            'bus': 2.2, 
            'truck': 2.5
            }, 
        'img_cod': '', # se obtiene de metvial
        'mov_pedestrian': params['no_vh'], 
        'mov_bike': params['no_vh'], 
        'mov_traffic': mov_traffic, 
    }
    fligth={
            serie.fligth_meta['period']: {
                'date': serie.fligth_meta['date'], 
                'time': serie.fligth_meta['time'], 
                'period': period_keys[serie.fligth_meta['period']],
                'duration': serie.fligth_meta['duration'],
                'pixel_size': serie.fligth_meta['pixel_size'], 
                'traj_file': video_path.as_posix() # carperta de destino
                }
            }

    serie['metvial_dict']={'network':network, 'fligth':fligth}
    serie.to_pickle(gzip_path.as_posix(), compression='gzip')
    return gzip_path.as_posix()


# moves = [

# [4, [['1','2'],['E', 'O']]],
# [4, [['2','3'],['E', 'O']]],
# [4, [['3','4'],['E', 'O']]],
# [3, [['5','6'],['O', 'E']]],
# [3, [['6','7'],['O', 'E']]],
# [3, [['7','8'],['O', 'E']]],

# ]

# files={
#     '093930':'01SAB_CARRERA45_CALLE88_20221201_1445',
#     '542075':'02SAB_CARRERA45_CALLE88_20221201_1445'
# }

# for i in ['093930', '542075']:
#     print('procesando : ', i)

# moves = [
# [2, [['1', '2'], ['S', 'N']]],
# [2, [['2', '3'], ['S', 'N']]],
# [2, [['3', '4'], ['S', 'N']]],
# [1, [['4', '3'], ['N', 'S']]],
# [1, [['3', '2'], ['N', 'S']]],
# [1, [['2', '1'], ['N', 'S']]],]


# params={
#     'video_path': fr'D:\resultados analitica de incidentes\Avenida  33 carrera 63b\01SAB_AVENIDA_CARRERA63B_20201016_1226.mp4',
#     'json_network': fr'D:\resultados analitica de incidentes\Avenida  33 carrera 63b\01SAB_AVENIDA_CARRERA63B_20201016_1226.json',
#     'mov_traffic': moves,
#     'no_vh': [],
# }

# # FRE ENG
# moves = [
# [1, [['17', '18'],['N', 'S']]],
# [1, [['17', '19'],['N', 'S']]],
# [2, [['4', '6'] ,['S', 'N']]],
# [2, [['4', '7'] ,['S', 'N']]],
# [2, [['4', '8'] ,['S', 'N']]],
# [2, [['4', '9'] ,['S', 'N']]],
# [2, [['3', '6'] ,['S', 'N']]],
# [2, [['3', '7'] ,['S', 'N']]],
# [2, [['3', '8'] ,['S', 'N']]],
# [2, [['3', '9'] ,['S', 'N']]],
# [2, [['8', '14'] ,['S', 'N']]],
# [2, [['8', '13'] ,['S', 'N']]],
# [2, [['22', '23'] ,['S', 'N']]],
# [4, [['11', '12'], ['E', 'O']]],
# [4, [['12', '15'], ['E', 'O']]],
# [6, [['10', '12'], ['S', 'O']]],
# [8, [['15', '18'], ['E', 'S']]],
# [8, [['15', '19'], ['E', 'S']]],
# [92, [['4', '5'], ['S', 'E']]],
# [93, [['1', '2'], ['O', 'S']]],
# [94, [['12', '13'], ['E', 'N']]],
# [94, [['15', '16'], ['E', 'N']]],
# ]

moves = [

[4, [['1','2'],['E', 'O']]],
[4, [['3','2'],['E', 'O']]],
[4, [['4','5'],['E', 'O']]]
# [3, [['5','6'],['O', 'E']]],
# [3, [['6','7'],['O', 'E']]],
# [3, [['7','8'],['O', 'E']]],

]

FOLDER_YOLOR = 'D:/GitHub/INTEIA/INTEIA-D33P-Detectores' # cambiar para linux 
FOLDER_METVIAL = 'D:/GitHub/INTEIA/INTEIA-D33P-Metvial' # Cambiar para linux
ENV_METVIAL = 'metvial'

# files={
#     139169:'01SAB_CARRERA45_CALLE88_20221201_1445',
#     340876:'02SAB_CARRERA45_CALLE88_20221201_1445'
# }
#for i in [139169, 340876]:

# files={
#     '139169':'01SAB_AVENIDA33_CARRERA63B_20221201_1445',
#     '340876':'02SAB_AVENIDA33_CARRERA63B_20221201_1445'
# }

files = {
    '139169': '01SAB_AVENIDA33_CARRERA63B_20221201_1445'
}

for key, value in files.items():
    print(f'Procesando : {key}')

    folder = f'D:/INTEIA/Videos/Proyectos/Costa_Rica/{key}/{value}'
    params={
        'video_path': folder + '.mp4',
        'json_network': folder + '.json',
        'mov_traffic': moves,
        'no_vh': ['21', '20'],
        'h': 100
    }

    gzip_path = create_gzip_(params)

    try:
        cmd = (f'conda activate {ENV_METVIAL} && '
            'd: && '
            f'cd {FOLDER_METVIAL} && '
            f'python metvial_compute.py --compute_gzip "{gzip_path}"')
        
        print(" SE ESTÁ EJECUTANDO METVIAL ")

        with Popen(cmd, stderr=PIPE, stdout=PIPE, shell=True) as p:
                    stdout, stderr = p.communicate()
            # print(stdout,stderr,sep='\n\n')
    except Exception as e:
        print(e)

    print('stdout', stdout)
    print('stderr', stderr)
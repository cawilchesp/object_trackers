from tools.video_info import VideoInfo
from pathlib import Path

from icecream import ic

# Constants
# ---------
FG_RED = '\033[31m'
FG_GREEN = '\033[32m'
FG_YELLOW = '\033[33m'
FG_BLUE = '\033[34m'
FG_WHITE = '\033[37m'
FG_BOLD = '\033[01m'
FG_RESET = '\033[0m'

# Funciones de color
# ------------------
def bold(text: str) -> str:
    return f"{FG_BOLD}{text}{FG_RESET}" if text is not None else ''

def red(text: str) -> str:
    return f"{FG_RED}{text}{FG_RESET}" if text is not None else ''

def green(text: str) -> str:
    return f"{FG_GREEN}{text}{FG_RESET}" if text is not None else ''

def yellow(text: str) -> str:
    return f"{FG_YELLOW}{text}{FG_RESET}" if text is not None else ''

def blue(text: str) -> str:
    return f"{FG_BLUE}{text}{FG_RESET}" if text is not None else ''

def white(text: str) -> str:
    return f"{FG_WHITE}{text}{FG_RESET}" if text is not None else ''

# Funciones
# ---------
def print_video_info(source: str, video_info: VideoInfo):
    if source.isnumeric():
        source_name = "Webcam"
    elif source.lower().startswith('rtsp://'):
        source_name = "RSTP Stream"
    else:
        source_name = Path(source).name
    
    text_length = 20 + max(len(source_name) , len(f"{video_info.width} x {video_info.height}"))
    
    # Print video information
    print(f"\n{red('*'*text_length)}")
    print(f"{green('Source Information'):^{text_length+9}}")
    print(f"{bold('Source'):<29}{source_name}")
    print(f"{bold('Size'):<29}{video_info.width} x {video_info.height}")
    print(f"{bold('Total Frames'):<29}{video_info.total_frames}") if video_info.total_frames is not None else None
    print(f"{bold('Frames Per Second'):<29}{video_info.fps:.2f}")
    print(f"\n{red('*'*text_length)}\n")
    

def print_progress(frame_number: int, total_frames: int):
    if total_frames is not None:
        percentage = f"[ {frame_number/total_frames:6.1%} ] "
        frame_progress = f"{frame_number} / {total_frames}"
        percentage_title = f"{'':11}"
    else:
        percentage = ''
        frame_progress = f"{frame_number}"
        percentage_title = ''
    
    frame_text_length = (2 * len(str(total_frames))) + 3
    if frame_number == 0:
        print(f"{percentage_title}{bold('Frame'):>{frame_text_length+9}}")
    print(f"\r{green(percentage)}{frame_progress:>{frame_text_length}}  ", end="", flush=True)


def print_times(frame_number: int, source_info: VideoInfo, progress_times: dict):
    total_frames = source_info.total_frames
    capture_time = progress_times['capture_time']
    inference_time = progress_times['inference_time']
    detections_time = progress_times['detections_time']
    tracks_time = progress_times['tracks_time']
    saving_time = progress_times['saving_time']
    drawings_time = progress_times['drawings_time']
    files_time = progress_times['files_time']
    frame_time = progress_times['frame_time']
    
    if total_frames is not None:
        percentage = f"[ {frame_number/total_frames:6.1%} ] "
        frame_progress = f"{frame_number} / {total_frames}"
        percentage_title = f"{'':11}"
    else:
        percentage = ''
        frame_progress = f"{frame_number}"
        percentage_title = ''
        
    frame_text_length = (2 * len(str(total_frames))) + 3
    if frame_number == 0:
        print(f"{percentage_title}{bold('Frame'):>{frame_text_length+9}}{bold('Capture'):>22}{bold('Inference'):>22}{bold('Detections'):>22}{bold('Tracks'):>22}{bold('Saving'):>22}{bold('Drawings'):>22}{bold('Files'):>22}{bold('Total'):>22}")

    print(
        f"{green(percentage)}"
        f"{frame_progress:>{frame_text_length}}  "
        f"{1000*(capture_time):8.2f} ms  "
        f"{1000*(inference_time):8.2f} ms  "
        f"{1000*(detections_time):8.2f} ms  "
        f"{1000*(tracks_time):8.2f} ms  "
        f"{1000*(saving_time):8.2f} ms  "
        f"{1000*(drawings_time):8.2f} ms  "
        f"{1000*(files_time):8.2f} ms  "
        f"{1000*(frame_time):8.2f} ms"
    )
    

def step_message(step: str = None, message: str = None):
    step_text = green(f"[{step}]") if step != "Error" else red(f"[{step}]")
    print(f"\n{step_text} {message} ✅")

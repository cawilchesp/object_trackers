from tools.video_info import VideoInfo
from pathlib import Path


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
    return f"{FG_BOLD}{text}{FG_RESET}"

def red(text: str) -> str:
    return f"{FG_RED}{text}{FG_RESET}"

def green(text: str) -> str:
    return f"{FG_GREEN}{text}{FG_RESET}"

def yellow(text: str) -> str:
    return f"{FG_YELLOW}{text}{FG_RESET}"

def blue(text: str) -> str:
    return f"{FG_BLUE}{text}{FG_RESET}"

def white(text: str) -> str:
    return f"{FG_WHITE}{text}{FG_RESET}"

# Funciones
# ---------
def print_video_info(source: str, video_info: VideoInfo):
    text_length = 20 + max(len(Path(source).name) , len(f"{video_info.width} x {video_info.height}"))
    
    # Print video information
    print(f"\n{red('*'*text_length)}")
    print(f"{green('Source Information'): ^{text_length+9}}")
    print(f"{bold('Source')}              {Path(source).name}") if not source.lower().startswith('rtsp://') else None
    print(f"{bold('Size')}                {video_info.width} x {video_info.height}")
    print(f"{bold('Total Frames')}        {video_info.total_frames}") if video_info.total_frames is not None else None
    print(f"{bold('Frames Per Second')}   {video_info.fps:.2f}")
    print(f"\n{red('*'*text_length)}\n")
    

def print_progress(frame_number: int, source_info: VideoInfo, progress_times: dict):
    total_frames = source_info.total_frames
    frame_time = progress_times['frame_time']
    capture_time = progress_times['capture_time']
    inference_time = progress_times['inference_time']
    annotations_time = progress_times['annotations_time']
    
    percentage = f"[ {100*frame_number/total_frames:.1f} % ] " if total_frames is not None else None
    frame_progress = f"{frame_number} / {total_frames}" if total_frames is not None else f"{frame_number}"

    print(
        f"{green(str(percentage))}"
        f"{bold('Frame:')} {frame_progress}  |  "
        f"{bold('Capture Time:')} {1000*(capture_time):.2f} ms  |  "
        f"{bold('Inference Time:')} {1000*(inference_time):.2f} ms  |  "
        f"{bold('Annotations Time:')} {1000*(annotations_time):.2f} ms  |  "
        f"{bold('Time per Frame:')} {1000*(frame_time):.2f} ms"
    )


def step_message(step: str = None, message: str = None):
    step_text = green(f"[{step}]") if step != "Error" else red(f"[{step}]")
    print(f"{step_text} {message}")

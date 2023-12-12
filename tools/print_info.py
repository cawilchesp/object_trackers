from supervision import VideoInfo
from pathlib import Path
import numpy as np

# Constants
FG_RED = '\033[31m'
FG_GREEN = '\033[32m'
FG_YELLOW = '\033[33m'
FG_BLUE = '\033[34m'
FG_WHITE = '\033[37m'
FG_BOLD = '\033[01m'
FG_RESET = '\033[0m'


def print_video_info(source: str, video_info: VideoInfo):
    text_length = 20 + max(len(Path(source).name) , len(f"{video_info.width} x {video_info.height}"))

    # Print Information
    print(f"\n{FG_RED}{'*':*^{text_length}}")
    print(f"{FG_GREEN}{'Source Information': ^{text_length}}{FG_WHITE}")
    print(f"{FG_BOLD}Source              {FG_RESET}{Path(source).name}") if not source.lower().startswith('rtsp://') else None
    print(f"{FG_BOLD}Size                {FG_RESET}{video_info.width} x {video_info.height}")
    print(f"{FG_BOLD}Total Frames        {FG_RESET}{video_info.total_frames}") if video_info.total_frames > 0 else None
    print(f"{FG_BOLD}Frames Per Second   {FG_RESET}{video_info.fps}")
    print(f"\n{FG_RED}{'*':*^{text_length}}\n{FG_RESET}")


def print_image_info(source: str, image: np.array):
    text_length = 20 + max(len(Path(source).name) , len(f"{image.shape[0]} x {image.shape[1]}"))

    # Print Information
    print(f"\n{FG_RED}{'*':*^{text_length}}")
    print(f"{FG_GREEN}{'Source Information': ^{text_length}}{FG_WHITE}")
    print(f"{FG_BOLD}Source              {FG_RESET}{Path(source).name}")
    print(f"{FG_BOLD}Size                {FG_RESET}{image.shape[0]} x {image.shape[1]}")
    print(f"\n{FG_RED}{'*':*^{text_length}}\n{FG_RESET}")


def print_progress(frame_number: int, source_info: VideoInfo, progress_times: dict):
    total_frames = source_info.total_frames
    frame_time = progress_times['frame_time']
    capture_time = progress_times['capture_time']
    inference_time = progress_times['inference_time']
    annotations_time = progress_times['annotations_time']
    
    percentage = f"[ {100*frame_number/total_frames:.1f} % ] " if total_frames > 0 else ""
    frame_progress = f"{frame_number} / {total_frames}" if total_frames > 0 else f"{frame_number}"

    print(
        f'{FG_GREEN}{percentage}'
        f'{FG_WHITE}{FG_BOLD}Frame: {FG_RESET}{frame_progress}  |  '
        f'{FG_BOLD}Capture Time: {FG_RESET}{1000*(capture_time):.2f} ms  |  '
        f'{FG_BOLD}Inference Time: {FG_RESET}{1000*(inference_time):.2f} ms  |  '
        f'{FG_BOLD}Annotations Time: {FG_RESET}{1000*(annotations_time):.2f} ms  |  '
        f'{FG_BOLD}Time per Frame: {FG_RESET}{1000*(frame_time):.2f} ms'
    )


def step_message(step: str = None, message: str = None):
    print(f"{FG_GREEN}[{step}] {FG_RESET}{message}")
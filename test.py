from detector import ObjectDetection

def main():
    detector = ObjectDetection(
        source='D:\Data\C3_detenido.mp4',
        # source=0,
        weights='yolov8m-seg',
        # weights='yolov9c',
        mode='track',
        labels=True,
        boxes=True,
        tracks=True,
        masks=True,
        heatmap=True,
        show_fps=True,
        # class_filter=[2],
        output='D:\Data\C3_detenido_output',
        show_output=False,
        
          )
    detector()

if __name__ == "__main__":
    main()
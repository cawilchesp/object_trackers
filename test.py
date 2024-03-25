from detector import ObjectDetection

detector = ObjectDetection(
    source="D:\Data\C3_detenido.mp4",
    weights='yolov9c.pt')
detector()
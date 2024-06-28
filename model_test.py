from ultralytics import YOLO

# Load a pretrained YOLOv10n model
root_model_path = 'D:/Data/models'

v8_model = YOLO(f"{root_model_path}/yolov8/yolov8x.pt")
v9_model = YOLO(f"{root_model_path}/yolov9/yolov9e.pt")
v10_model = YOLO(f"{root_model_path}/yolov10/yolov10x.pt")



# Perform object detection on an image
test_image = 'D:/Data/test_image.jpg'
results_v8 = v8_model(test_image)
results_v9 = v9_model(test_image)
results_v10 = v10_model(test_image)

# Display the results
# results_v8[0].show()
# results_v9[0].show()
results_v10[0].show()
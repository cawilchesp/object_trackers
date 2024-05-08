from roboflow import Roboflow
from icecream import ic

import roboflow_config

# Initialize the Roboflow object with your API key
rf = Roboflow(api_key=roboflow_config.API_KEY)

# Retrieve your current workspace and project name
# print(rf.workspace())
# quit()





# Specify the project for upload
# let's you have a project at https://app.roboflow.com/my-workspace/my-project
# project_id = 'pruebas-iuhpl'
# project = rf.workspace(roboflow_config.workspace_id).project(project_id)

# nombre_conjunto = 'P2_50_100_CR'

# # Upload the image to your project
# image_file = 'D:/Data/CCT_Drones/P2_50/P2_50_100_CR.jpg'
# annotation_file = 'D:/Data/CCT_Drones/P2_50/P2_50_100_CR.txt'
# labelmap_file = 'D:/Data/CCT_Drones/P2_50/labelmap.txt'

# """
# Optional Parameters:
# - num_retry_uploads: Number of retries for uploading the image in case of failure.
# - batch_name: Upload the image to a specific batch. (conjunto de imágenes)
# - split: Upload the image to a specific split. (train, val, test)
# - tag: Store metadata as a tag on the image.
# - sequence_number: [Optional] If you want to keep the order of your images in the dataset, pass sequence_number and sequence_size..
# - sequence_size: [Optional] The total number of images in the sequence. Defaults to 100,000 if not set.
# """

# project.upload(
#     num_retry_uploads=3,
#     batch_name=nombre_conjunto,
#     image_path=image_file,
#     annotation_path=annotation_file,
#     annotation_labelmap=labelmap_file,
#     split="train",
#     sequence_number=85,
#     sequence_size=100
# )



### Prueba de Generar version
# project_id = 'tejo-tracking'
# project = rf.workspace(roboflow_config.workspace_id).project(project_id)
# project.generate_version(
#     settings={
#         "preprocessing": {
#             "auto-orient": True,
#             # "contrast": { "type": "Contrast Stretching" },
#             # "filter-null": { "percent": 50 },
#             # "grayscale": True,
#             # "isolate": True,
#             # "remap": { "original_class_name": "new_class_name" },
#             "resize": { "width": 640, "height": 640, "format": "Stretch to" },
#             # "static-crop": { "x_min": 10, "x_max": 90, "y_min": 10, "y_max": 90 },
#             # "tile": { "rows": 2, "columns": 2 }
#         },
#         "augmentation": {
#             # "bbblur": { "pixels": 1.5 },
#             # "bbbrightness": { "brighten": True, "darken": False, "percent": 25 },
#             # "bbcrop": { "min": 12, "max": 71 },
#             # "bbexposure": { "percent": 30 },
#             # "bbflip": { "horizontal": True, "vertical": True },
#             # "bbnoise": { "percent": 50 },
#             # "bbninety": { "clockwise": True, "counter-clockwise": True, "upside-down": False },
#             # "bbrotate": { "degrees": 45 },
#             # "bbshear": { "horizontal": 45, "vertical": 45 },
#             # "blur": { "pixels": 1.5 },
#             "brightness": { "brighten": True, "darken": True, "percent": 25 },
#             # "crop": { "min": 12, "max": 71 },
#             # "cutout": { "count": 26, "percent": 71 },
#             "exposure": { "percent": 30 },
#             "flip": { "horizontal": True, "vertical": True },
#             # "hue": { "degrees": 180 },
#             "image": { "versions": 32 },
#             # "mosaic": True,
#             "ninety": { "clockwise": True, "counter-clockwise": True, "upside-down": False },
#             # "noise": { "percent": 50 },
#             # "rgrayscale": { "percent": 50 },
#             "rotate": { "degrees": 15 },
#             # "saturation": { "percent": 50 },
#             # "shear": { "horizontal": 45, "vertical": 45 }
#         }
#     }
# )


### Prueba de Descargar version
# project_id = 'tejo-tracking'
# project = rf.workspace(roboflow_config.workspace_id).project(project_id)
# # print(project.list_versions())
# version = project.version(1)
# dataset = version.download(
#     model_format='yolov9',
#     # location='D:/Data/Tejo'
# )
# ic(dataset.name)
# ic(dataset.version)
# ic(dataset.model_format)
# ic(dataset.location)
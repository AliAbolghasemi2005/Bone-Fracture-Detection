import keras
import keras_hub

from data import CLASSES


# Load the pretrained RetinaNet backbone

backbone = keras_hub.models.Backbone.from_preset(
    "hf://keras/retinanet_resnet50_fpn_coco"
)

preprocessor = keras_hub.models.RetinaNetObjectDetectorPreprocessor.from_preset(
    "hf://keras/retinanet_resnet50_fpn_coco"
)

model = keras_hub.models.RetinaNetObjectDetector(
    backbone=backbone,
    num_classes=len(CLASSES),
    preprocessor=preprocessor
)


# Check the preprocessing configuration.

print("Image converter:")
print(preprocessor.image_converter)

print("\nImage size:")
print(preprocessor.image_converter.image_size)

print("\nBounding box format:")
print(preprocessor.image_converter.bounding_box_format)


# Configure the model for training

model.compile(
    box_loss=keras.losses.MeanAbsoluteError(
        reduction="sum"
    )
)
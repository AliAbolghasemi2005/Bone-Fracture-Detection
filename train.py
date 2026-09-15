import pathlib
import tensorflow as tf
import keras
import keras_hub

# Dataset paths and class names

dataset_dir = pathlib.Path("Bone-Fractures-Detection-Dataset")

CLASSES = {
    0: "Comminuted",
    1: "Greenstick",
    2: "Healthy",
    3: "Linear",
    4: "Oblique Displaced",
    5: "Oblique",
    6: "Segmental",
    7: "Spiral",
    8: "Transverse Displaced",
    9: "Transverse",
}


# Read one YOLO annotation file
#
# YOLO format:
# class_id x_center y_center width height

def read_annotation_data(annotation_file_path):

    with open(annotation_file_path, "r", encoding="utf-8") as f:
        text = f.read()

    labels = []
    boxes = []

    for line in text.splitlines():

        values = line.split()

        labels.append(int(values[0]))

        boxes.append(
            [float(value) for value in values[1:]]
        )

    return {
        "labels": labels,
        "boxes": boxes
    }


# Build metadata for a dataset split
#
# Creates a Python dictionary containing:
# - class labels
# - bounding boxes
# - paths to the corresponding images

def build_metadata(split):

    labels_dir = dataset_dir / split / "labels"
    images_dir = dataset_dir / split / "images"

    metadata = {
        "labels": [],
        "boxes": [],
        "image/file_path": []
    }

    for annotation_file_path in labels_dir.glob("*.txt"):

        annotation = read_annotation_data(
            annotation_file_path
        )

        image_file_name = annotation_file_path.stem + ".jpg"
        image_file_path = images_dir / image_file_name

        metadata["labels"].append(
            annotation["labels"]
        )

        metadata["boxes"].append(
            annotation["boxes"]
        )

        metadata["image/file_path"].append(
            str(image_file_path)
        )

    return metadata


# Load an image from its path

def load_image(sample):

    image_path = sample.pop("image/file_path")

    image = tf.io.read_file(
        image_path
    )

    image = tf.image.decode_jpeg(
        image
    )

    sample["image"] = image

    return sample


# Convert YOLO boxes to yxyx pixel coordinates that RetinaNet needs

def convert_boxes(sample):

    height = tf.shape(sample["image"])[0]
    width = tf.shape(sample["image"])[1]

    sample["boxes"] = keras.utils.bounding_boxes.convert_format(
        sample["boxes"],
        source="rel_center_xywh",
        target="yxyx",
        height=height,
        width=width,
    )

    return sample


# Put the annotations into the format expected by Keras

def decode_dataset(sample):

    return {
        "images": sample["image"],
        "bounding_boxes": {
            "boxes": sample["boxes"],
            "labels": sample["labels"],
        }
    }


# Convert the Keras bounding-box structure into
# the (images, targets) format expected by model.fit()

def convert_to_tuple(record):

    return record["images"], {
        "boxes": record["bounding_boxes"]["boxes"],
        "labels": record["bounding_boxes"]["labels"],
    }


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


# Pad the annotations so every sample has the same number of boxes

max_box_layer = keras.layers.MaxNumBoundingBoxes(
    max_number=100,
    bounding_box_format="yxyx"
)


# This function applies the entire input pipeline:
#
# metadata
#     ↓
# RaggedTensors
#     ↓
# TensorFlow Dataset
#     ↓
# load images
#     ↓
# convert bounding boxes
#     ↓
# Keras bounding-box structure
#     ↓
# pad boxes to 100
#     ↓
# batch
#     ↓
# (images, targets)
#     ↓
# prefetch

def build_dataset(split):

    # Build metadata from the selected dataset split.
    metadata = build_metadata(split)

    # Convert variable-length annotations into RaggedTensors.
    metadata["labels"] = tf.ragged.constant(
        metadata["labels"]
    )

    metadata["boxes"] = tf.ragged.constant(
        metadata["boxes"],
        ragged_rank=1
    )

    # Create a TensorFlow Dataset from the metadata.
    dataset = tf.data.Dataset.from_tensor_slices(
        metadata
    )

    # Load the actual image files.
    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Convert YOLO bounding boxes to yxyx format.
    dataset = dataset.map(
        convert_boxes,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Rearrange the sample into Keras bounding-box format.
    dataset = dataset.map(
        decode_dataset,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Pad annotations to a fixed maximum number of boxes.
    dataset = dataset.map(
        max_box_layer,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Group samples into batches.
    dataset = dataset.batch(
        4,
        drop_remainder=True
    )

    # Convert each batch into the format expected by model.fit().
    dataset = dataset.map(
        convert_to_tuple,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Prepare future batches while the current batch is being processed.
    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# Build training and validation and test datasets

train_dataset = build_dataset("train")
valid_dataset = build_dataset("valid")
test_dataset = build_dataset("test")
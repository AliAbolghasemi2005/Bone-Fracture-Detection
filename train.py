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

metadata = build_metadata("train")

print("Number of images:", len(metadata["image/file_path"]))

print("\nFirst image path:")
print(metadata["image/file_path"][0])

print("\nFirst labels:")
print(metadata["labels"][0])

print("\nFirst boxes:")
print(metadata["boxes"][0])
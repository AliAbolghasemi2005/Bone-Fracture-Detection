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
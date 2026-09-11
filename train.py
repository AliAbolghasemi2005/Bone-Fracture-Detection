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
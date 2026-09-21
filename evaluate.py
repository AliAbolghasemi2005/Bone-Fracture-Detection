import pathlib
import keras
import matplotlib.pyplot as plt

from data import build_dataset, CLASSES

# Path to the trained model

model_path = pathlib.Path(
    "outputs/retinanet_best.keras"
)


# Check if the trained model exists

if not model_path.exists():

    print(
        "Trained model not found."
        "\nRun train.py first."
    )

    raise SystemExit


# Build the test dataset

test_dataset = build_dataset("test")


# Load the trained model

model = keras.models.load_model(
    model_path
)


# Evaluate the model

model.evaluate(test_dataset)


# Visualizing the predictions

images, y_true = next(iter(test_dataset))

y_pred = model.predict(
    images
)

keras.visualization.plot_bounding_box_gallery(
    images=images,
    bounding_box_format="yxyx",
    y_true=y_true,
    y_pred=y_pred,
    rows=2,
    cols=2,
    scale=4,
    class_mapping=CLASSES,
    legend=True,
)
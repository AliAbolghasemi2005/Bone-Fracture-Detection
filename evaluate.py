import pathlib
import keras

from data import build_dataset


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
import keras

from data import build_dataset
from model import model


# Build training and validation datasets

train_dataset = build_dataset("train")
valid_dataset = build_dataset("valid")


# Save the model whenever validation loss improves

checkpoint = keras.callbacks.ModelCheckpoint(
    "outputs/retinanet_best.keras",
    monitor="val_loss",
    save_best_only=True,
)


# Train the model

model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=5,
    callbacks=[checkpoint]
)
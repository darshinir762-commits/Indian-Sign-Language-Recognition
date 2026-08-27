from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)


def build_cnn(num_classes):
    model = Sequential(name="ISL_CNN")

    # Block 1
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=(224, 224, 3),
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 2
    model.add(
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu",
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 3
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation="relu",
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Classifier
    model.add(Flatten())

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation="softmax"))

    return model
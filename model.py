import tensorflow as tf
import numpy as np
import cv2 as cv
from numpy.typing import NDArray
from augment_utils import CircularMask
from typing import Callable


class MobileNetV3Regression:

    def __init__(self, shape: tuple[int, int], channels: int, num_outputs: int):
        self._shape = (*shape, channels)
        self._num_outputs = num_outputs

        self._cnn = tf.keras.applications.MobileNetV3Large(
            input_shape=self._shape,
            include_top=False,
            include_preprocessing=False,
            pooling='max'
        )
        self._cnn.trainable = True

        x = inputs = tf.keras.Input(shape=self._shape)
        x = self._cnn(x)
        x = tf.keras.layers.Dense(64, activation="leaky_relu", name='dense_1')(x)
        x = tf.keras.layers.Dropout(.25)(x)
        x = tf.keras.layers.Dense(32, activation="leaky_relu", name='dense_2')(x)
        x = tf.keras.layers.Dropout(.25)(x)
        x = tf.keras.layers.Dense(self._num_outputs, activation="linear", name='output')(x)
        print(f"{x.shape}")

        self._model = tf.keras.Model(inputs, x)

    def model(self):
        return self._model

    @staticmethod
    def preprocess_input() -> Callable:
        return tf.keras.layers.Rescaling(scale=1./127.5, offset=-1.)

    @staticmethod
    def preprocessed_range() -> tuple[float, float]:
        return -1., 1.

    @staticmethod
    def restore_output_to_cv(x):
        x = x.numpy()
        x = cv.cvtColor((x + 1.) * 127.5, cv.COLOR_RGB2BGR).astype(np.uint8)
        return x

    @staticmethod
    def input_cv_to_tf(x):
        x = cv.cvtColor(x, cv.COLOR_BGR2RGB)
        x = x.astype(np.float32) / 127.5 - 1.
        return x


class OrientationNet:

    def __init__(self, shape: tuple[int, int], channels: int, num_outputs: int):

        self._shape = (*shape, channels)
        self._num_outputs = num_outputs

        self._model = tf.keras.Sequential([

            tf.keras.layers.Input(shape=self._shape),
            CircularMask(self._shape),

            tf.keras.layers.Conv2D(32, 3, padding='same', activation='leaky_relu'),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='leaky_relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Conv2D(64, 3, padding='same', activation='leaky_relu'),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='leaky_relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Conv2D(128, 3, padding='same', activation='leaky_relu'),
            tf.keras.layers.Conv2D(128, 3, padding='same', activation='leaky_relu'),
            tf.keras.layers.Conv2D(128, 3, padding='same', activation='leaky_relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Conv2D(256, 3, padding='same', activation='leaky_relu'),
            tf.keras.layers.Conv2D(256, 3, padding='same', activation='leaky_relu'),
            tf.keras.layers.Conv2D(256, 3, padding='same', activation='leaky_relu'),
            tf.keras.layers.Conv2D(256, 3, padding='same', activation='leaky_relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(512, activation='leaky_relu'),
            tf.keras.layers.Dropout(.50),
            tf.keras.layers.Dense(512, activation='leaky_relu'),
            tf.keras.layers.Dropout(.50),

            tf.keras.layers.Dense(self._num_outputs, activation='linear')
        ])

    def model(self) -> tf.keras.Model:
        return self._model

    @staticmethod
    def preprocess_input() -> Callable:
        return tf.keras.layers.Rescaling(1. / 255.)

    @staticmethod
    def preprocessed_range() -> tuple[float, float]:
        return 0., 1.

    @staticmethod
    def restore_output_to_cv(x):
        x = x.numpy()
        x = cv.cvtColor(x * 255, cv.COLOR_BGR2RGB).astype(np.uint8)
        return x

    @staticmethod
    def input_cv_to_tf(x):
        x = cv.cvtColor(x, cv.COLOR_BGR2RGB)
        x = x.astype(np.float32) / 255.
        return x



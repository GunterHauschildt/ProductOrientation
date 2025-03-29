import os
from time import strftime, localtime
import tensorflow as tf
from tf_records import load_tfrecord_dataset
from model import OrientationNet, MobileNetV3Regression
from utils import restore_angle, markup_and_show
from augment_utils import RandomColorize, CircularMask, CropAndResize
import argparse
import numpy as np
import math
from typing import Any
import numpy as np


class CircularMask(tf.keras.layers.Layer):
    def __init__(self, shape: tuple[int, int, int]):
        super().__init__()
        self._shape = shape
        self._clr = (0, 0, 0) if self._shape[2] == 3 else 0
        self._set = (255, 255, 255) if self._shape[2] == 3 else (255,)
        rad = min(self._shape[0], self._shape[1]) // 2 - 1
        center = (self._shape[0] // 2, self._shape[1] // 2)
        mask = np.zeros((self._shape[0], self._shape[1], self._shape[2])).astype(np.uint8)
        mask = cv.circle(mask, center, rad, self._set, cv.FILLED)
        self._mask = tf.convert_to_tensor(mask)

    def call(self, s, *args, **kwargs):
        return tf.where(self._mask != 0, s, self._clr)


class RandomColorize(tf.keras.layers.Layer):

    def __init__(self, min: float, max: float):
        super().__init__()
        assert math.isclose(-max, min) or math.isclose(min, 0.), \
            "Only -a - -a or +a - 0 style ranges is accepted."
        self._min = min
        self._max = max

    def call(self, s, *args, **kwargs):
        s = tf.image.random_brightness(s, .1)
        s = tf.image.random_contrast(s, .75, 1.0)
        s = tf.image.random_hue(s, .05)
        s = tf.clip_by_value(s, self._min, self._max)
        return s


class CropAndResize(tf.keras.layers.Layer):
    def __init__(self,
                 shape: tuple[int, int, int],
                 do_factor: float = .125,
                 dont_restore_factor: float = .50):
        super().__init__()
        self._shape = shape
        _do_factor = max(1, round(1. / do_factor)) - 1
        self._do = tf.convert_to_tensor(np.array([True] + [False] * _do_factor))
        dont_restore = max(1, round(1. / dont_restore_factor)) - 1
        self._dont_restore = tf.convert_to_tensor(np.array([True] + [False] * dont_restore))

    def call(self, s, *args, **kwargs):

        s = tf.image.crop_to_bounding_box(s,
               tf.random.uniform([], 0, self._shape[0] // 8, dtype=tf.int32),
               tf.random.uniform([], 0, self._shape[1] // 8, dtype=tf.int32),
               tf.random.uniform([], 6 * self._shape[0] // 8, 7 * self._shape[0] // 8, dtype=tf.int32),
               tf.random.uniform([], 6 * self._shape[1] // 8, 7 * self._shape[1] // 8, dtype=tf.int32))
        s = tf.expand_dims(s, 0)

        if tf.random.shuffle(self._dont_restore)[0]:
            s = tf.image.resize_with_crop_or_pad(s, *self._shape[:2])
        else:
            s = tf.image.resize(s, self._shape[:2])

        s = tf.squeeze(s, 0)
        return s
        
        
def main():
    print("GPU Devices: ", tf.config.list_physical_devices('GPU'))

    parser = argparse.ArgumentParser()
    parser.add_argument("--root-folder",
                        help="Root folder.",
                        required=True,
                        type=str)
    parser.add_argument("--trainvalid-folder",
                        help="Train/Valid folder.",
                        required=False,
                        default="TrainValid",
                        type=str)
    parser.add_argument("--shape",
                        help=(
                            "Shape of the data in the model. Not necessarily the size used for the tfrecords."
                            "(But should be the same or smaller.)"
                            "Used to keep the tfrecords files 'reasonably' sized. rows x cols"),
                        type=tuple[int, int],
                        default=(256, 256),
                        nargs=2)
    parser.add_argument("--epochs",
                        help="Number of epochs",
                        default=64,
                        type=int)
    parser.add_argument("--batch-size",
                        help="Batch size used while training. If you get an out-of-memory error, decrease.",
                        default=16,
                        type=int)
    parser.add_argument("--buffer-size",
                        help="Buffer size size used while training. If you get an out-of-memory error, decrease.",
                        default=64,
                        type=int)
    parser.add_argument("--show-augmented",
                        help="Shows the augmented data (as augmented during training).",
                        default=False,
                        type=bool)
    parser.add_argument(
        "--model-weights",
        help=("The model name to be used."
              "Expected to be in root/trained_models/checkpoints"
              "Include the .tf but not .index ie '2024-10-25-20-21-0.02.tf'"
              "Must match the model architecture in model.py"),
        default=None,
        type=str)

    args = parser.parse_args()
    shape = (int(args.shape[0]), int(args.shape[1]), 3)

    tf_records_file_path_train = os.path.join(args.root_folder, "train.tf_record")
    tf_records_file_path_valid = os.path.join(args.root_folder, "valid.tf_record")

    epochs = args.epochs
    print(f"Loading model ...")
    nn = OrientationNet(args.shape, 3, 4)
    # nn = MobileNetV3Regression(args.shape, 3, 4)

    weights_file_name = None
    if args.model_weights is not None:
        weights_file_name = os.path.join(args.root_folder,
                                         "trained_models",
                                         "checkpoints",
                                         args.model_weights)
    if weights_file_name is not None and os.path.isfile(weights_file_name + ".index"):
        try:
            nn.model().load_weights(weights_file_name).expect_partial()
            print(f"Successfully loaded weights from: {weights_file_name}")
        except (Exception,) as e:
            print(f"Error loading weights from: {weights_file_name}: {e}\nContinuing ...")

    def cos_sin_cos_sin_loss(y_true, y_pred):
        cos1_sin1_true = y_true[:, 0:2]
        cos2_sin2_true = y_true[:, 2:4]
        cos1_sin1_pred = y_pred[:, 0:2]
        cos2_sin2_pred = y_pred[:, 2:4]
        cos1_sin1_true = (cos1_sin1_true * 2.) - 1.
        cos2_sin2_true = (cos2_sin2_true * 2.) - 1.
        cos1_sin1_pred = (cos1_sin1_pred * 2.) - 1.
        cos2_sin2_pred = (cos2_sin2_pred * 2.) - 1.

        cos1_sin1_true_sum = tf.reduce_sum(tf.abs(cos1_sin1_true), axis=1, keepdims=True)
        cos2_sin2_true_sum = tf.reduce_sum(tf.abs(cos2_sin2_true), axis=1, keepdims=True)

        diff = tf.where(cos1_sin1_true_sum >= cos2_sin2_true_sum,
                        cos1_sin1_true - cos1_sin1_pred,
                        cos2_sin2_true - cos2_sin2_pred
                        )
        return tf.reduce_mean(tf.square(diff), axis=-1)

    optimizer = tf.keras.optimizers.SGD(learning_rate=.0001)
    nn.model().compile(optimizer=optimizer, loss=cos_sin_cos_sin_loss)
    nn.model().summary()

    # resizing and augmenting
    resize_and_rescale = tf.keras.Sequential([
        nn.preprocess_input(),
        tf.keras.layers.Resizing(*shape[0:2]),
    ])

    resize_and_rescale_and_augment = tf.keras.Sequential([
        resize_and_rescale,
        RandomColorize(*nn.preprocessed_range()),
        CropAndResize(shape),
        # ... we don't need here unless testing augmentation. It doesn't hurt.
        CircularMask(shape)
    ])

    train_dataset = load_tfrecord_dataset(tf_records_file_path_train)
    train_dataset = train_dataset.shuffle(buffer_size=args.buffer_size)
    train_dataset = train_dataset.map(lambda x, y: (resize_and_rescale_and_augment(x), y))
    train_dataset = train_dataset.batch(args.batch_size)

    done_show_augmented = False
    if args.show_augmented:
        for Xb, yb in train_dataset:
            for b in range(Xb.shape[0]):
                image = nn.restore_output_to_cv(Xb[b])
                y = yb[b].numpy()
                angle = restore_angle(*y)
                _, ch = markup_and_show(image, angle)
                if ch in [ord('x'), ord('q'), ord('Q')]:
                    done_show_augmented = True
                    break
            if done_show_augmented:
                break

    valid_dataset = load_tfrecord_dataset(tf_records_file_path_valid)
    valid_dataset = valid_dataset.shuffle(buffer_size=args.buffer_size)
    valid_dataset = valid_dataset.map(lambda x, y: (resize_and_rescale(x), y))
    valid_dataset = valid_dataset.batch(args.batch_size)

    checkpoint_dir = os.path.join(args.root_folder,
                                  "trained_models",
                                  "checkpoints",
                                  f"{strftime('%Y-%m-%d-%H-%M', localtime())}" + "-{val_loss:.4f}.tf")
    keyhit_file = os.path.join(checkpoint_dir, "saved_weights_keyhit")

    if epochs:
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(checkpoint_dir,
                                               monitor='val_loss',
                                               save_best_only=True,
                                               save_weights_only=True)
        ]
        try:
            nn.model().fit(train_dataset, epochs=epochs, validation_data=valid_dataset,
                           callbacks=callbacks)
        except KeyboardInterrupt:
            nn.model().save(keyhit_file)


if __name__ == '__main__':
    main()

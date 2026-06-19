from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import numpy as np


IMAGENET_TRAIN_EXAMPLES = 1_281_167
IMAGENET_VALIDATION_EXAMPLES = 50_000


@dataclass(frozen=True)
class ImageNetStreamInfo:
    input_shape: tuple[int, int, int, int]
    train_examples: int
    hyperval_examples: int
    validation_examples: int
    train_split: str
    hyperval_split: str
    validation_split: str


def _require_tensorflow():
    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ImageNet streaming requires optional TensorFlow support. "
            "Install TensorFlow in an environment that supports it, then make sure "
            "the TFDS imagenet2012 dataset is prepared under data.data_dir."
        ) from exc
    return tf, tfds


def imagenet_stream_info(config: Dict) -> ImageNetStreamInfo:
    image_size = int(config.get("image_size", 224))
    hyperval_examples = int(config.get("hyperval_size", 50_000))
    train_examples = int(
        config.get("train_examples", IMAGENET_TRAIN_EXAMPLES - hyperval_examples)
    )
    validation_examples = int(config.get("validation_examples", IMAGENET_VALIDATION_EXAMPLES))
    train_split = config.get("train_split", f"train[{hyperval_examples}:]")
    hyperval_split = config.get("hyperval_split", f"train[:{hyperval_examples}]")
    validation_split = config.get("validation_split", "validation")
    return ImageNetStreamInfo(
        input_shape=(1, image_size, image_size, 3),
        train_examples=train_examples,
        hyperval_examples=hyperval_examples,
        validation_examples=validation_examples,
        train_split=train_split,
        hyperval_split=hyperval_split,
        validation_split=validation_split,
    )


def _center_crop_numpy(image: np.ndarray, crop_height: int, crop_width: int) -> np.ndarray:
    height, width = image.shape[:2]
    top = max(0, (height - crop_height) // 2)
    left = max(0, (width - crop_width) // 2)
    return image[top : top + crop_height, left : left + crop_width]


def _resize_short_side_for_eval(image_size: int) -> int:
    return int(round(image_size * 256 / 224))


def _stateless_seed(tf, base_seed: int, index, salt: int):
    return tf.stack(
        [
            tf.cast(base_seed, tf.int32),
            tf.cast(index + salt, tf.int32),
        ]
    )


def _distort_color(tf, image, base_seed: int, index):
    image = tf.image.stateless_random_brightness(
        image,
        max_delta=32.0 / 255.0,
        seed=_stateless_seed(tf, base_seed, index, 11),
    )
    image = tf.image.stateless_random_saturation(
        image,
        lower=0.5,
        upper=1.5,
        seed=_stateless_seed(tf, base_seed, index, 12),
    )
    image = tf.image.stateless_random_hue(
        image,
        max_delta=0.2,
        seed=_stateless_seed(tf, base_seed, index, 13),
    )
    image = tf.image.stateless_random_contrast(
        image,
        lower=0.5,
        upper=1.5,
        seed=_stateless_seed(tf, base_seed, index, 14),
    )
    return tf.clip_by_value(image, 0.0, 1.0)


def _preprocess_imagenet_record(tf, record, image_size: int, training: bool, base_seed: int, index):
    image = tf.image.convert_image_dtype(record["image"], tf.float32)
    label = tf.cast(record["label"], tf.int32)

    if training:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
        begin, size, _ = tf.image.stateless_sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            seed=_stateless_seed(tf, base_seed, index, 1),
            min_object_covered=0.1,
            aspect_ratio_range=(0.75, 4.0 / 3.0),
            area_range=(0.08, 1.0),
            max_attempts=100,
            use_image_if_no_bounding_boxes=True,
        )
        image = tf.slice(image, begin, size)
        image = tf.image.resize(
            image,
            (image_size, image_size),
            method="bilinear",
            antialias=True,
        )
        image = tf.image.stateless_random_flip_left_right(
            image,
            seed=_stateless_seed(tf, base_seed, index, 2),
        )
        image = _distort_color(tf, image, base_seed, index)
    else:
        resize_side = _resize_short_side_for_eval(image_size)
        shape = tf.shape(image)
        height = tf.cast(shape[0], tf.float32)
        width = tf.cast(shape[1], tf.float32)
        scale = tf.cast(resize_side, tf.float32) / tf.minimum(height, width)
        new_height = tf.cast(tf.round(height * scale), tf.int32)
        new_width = tf.cast(tf.round(width * scale), tf.int32)
        image = tf.image.resize(
            image,
            (new_height, new_width),
            method="bilinear",
            antialias=True,
        )
        image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)

    image = tf.clip_by_value(image, 0.0, 1.0)
    image.set_shape((image_size, image_size, 3))
    return {"image": image, "label": label}


class TfdsImageNetIterator(Iterator[Dict[str, np.ndarray]]):
    def __init__(
        self,
        split: str,
        batch_size: int,
        data_dir: Optional[str] = None,
        image_size: int = 224,
        training: bool = True,
        seed: int = 0,
        shuffle_buffer: int = 8192,
        repeat: bool = True,
        drop_remainder: bool = True,
        skip_batches: int = 0,
    ) -> None:
        tf, tfds = _require_tensorflow()
        dataset = tfds.load(
            "imagenet2012",
            split=split,
            data_dir=data_dir,
            shuffle_files=training,
        )
        if training and shuffle_buffer > 0:
            dataset = dataset.shuffle(
                shuffle_buffer,
                seed=seed,
                reshuffle_each_iteration=True,
            )
        if repeat:
            dataset = dataset.repeat()

        dataset = dataset.enumerate()
        dataset = dataset.map(
            lambda index, record: _preprocess_imagenet_record(
                tf,
                record,
                image_size=image_size,
                training=training,
                base_seed=seed,
                index=index,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        if skip_batches > 0:
            dataset = dataset.skip(skip_batches)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        self._iterator = iter(tfds.as_numpy(dataset))

    def __iter__(self) -> "TfdsImageNetIterator":
        return self

    def __next__(self) -> Dict[str, np.ndarray]:
        batch = next(self._iterator)
        return {
            "image": batch["image"].astype(np.float32),
            "label": batch["label"].astype(np.int32),
        }


def make_imagenet_iterator(
    config: Dict,
    split: str,
    batch_size: int,
    training: bool,
    seed: int,
    repeat: bool = True,
    drop_remainder: bool = True,
    skip_batches: int = 0,
) -> TfdsImageNetIterator:
    return TfdsImageNetIterator(
        split=split,
        batch_size=batch_size,
        data_dir=config.get("data_dir"),
        image_size=int(config.get("image_size", 224)),
        training=training,
        seed=seed,
        shuffle_buffer=int(config.get("shuffle_buffer", 8192)),
        repeat=repeat,
        drop_remainder=drop_remainder,
        skip_batches=skip_batches,
    )

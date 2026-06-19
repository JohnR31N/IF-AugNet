from dataclasses import dataclass
import gzip
import hashlib
from pathlib import Path
import pickle
import struct
import tarfile
import time
from typing import Dict, Iterator, Optional, Sequence

import numpy as np
import requests


@dataclass
class DatasetSplits:
    train_images: np.ndarray
    train_labels: np.ndarray
    hyperval_images: np.ndarray
    hyperval_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray


class NumpyBatchIterator(Iterator[Dict[str, np.ndarray]]):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        seed: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> None:
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(images))
        self.position = len(images)

    def __iter__(self) -> "NumpyBatchIterator":
        return self

    def state_dict(self) -> Dict:
        return {
            "position": self.position,
            "indices": self.indices,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: Dict) -> None:
        self.position = state["position"]
        self.indices = state["indices"]
        self.rng.bit_generator.state = state["rng_state"]

    def __next__(self) -> Dict[str, np.ndarray]:
        if self.position >= len(self.indices):
            self.position = 0
            if self.shuffle:
                self.rng.shuffle(self.indices)

        end = self.position + self.batch_size
        if end > len(self.indices) and self.drop_last:
            if self.batch_size > len(self.indices):
                raise StopIteration
            self.position = 0
            if self.shuffle:
                self.rng.shuffle(self.indices)
            end = self.batch_size

        batch_indices = self.indices[self.position : min(end, len(self.indices))]
        if len(batch_indices) == 0:
            raise StopIteration

        self.position += len(batch_indices)
        return {
            "image": self.images[batch_indices],
            "label": self.labels[batch_indices],
        }


def _as_float_images(images: np.ndarray) -> np.ndarray:
    return images.astype(np.float32) / 255.0


def _split_train_hyperval(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    hyperval_size: int,
    seed: int,
) -> DatasetSplits:
    if hyperval_size <= 0 or hyperval_size >= len(train_images):
        raise ValueError(
            f"hyperval_size must be in [1, {len(train_images) - 1}], got {hyperval_size}."
        )

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(train_images))
    hyperval_indices = indices[:hyperval_size]
    train_indices = indices[hyperval_size:]

    return DatasetSplits(
        train_images=train_images[train_indices],
        train_labels=train_labels[train_indices],
        hyperval_images=train_images[hyperval_indices],
        hyperval_labels=train_labels[hyperval_indices],
        test_images=test_images,
        test_labels=test_labels,
    )


def load_tfds_image_dataset(
    name: str,
    data_dir: Optional[str] = None,
    hyperval_size: int = 5000,
    seed: int = 0,
) -> DatasetSplits:
    import tensorflow_datasets as tfds

    train = tfds.as_numpy(tfds.load(name, split="train", batch_size=-1, data_dir=data_dir))
    test = tfds.as_numpy(tfds.load(name, split="test", batch_size=-1, data_dir=data_dir))

    train_images = _as_float_images(train["image"])
    train_labels = train["label"].astype(np.int32)
    test_images = _as_float_images(test["image"])
    test_labels = test["label"].astype(np.int32)

    return _split_train_hyperval(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        hyperval_size=hyperval_size,
        seed=seed,
    )


def load_cifar10(
    data_dir: Optional[str] = None,
    hyperval_size: int = 5000,
    seed: int = 0,
) -> DatasetSplits:
    return load_tfds_image_dataset(
        "cifar10",
        data_dir=data_dir,
        hyperval_size=hyperval_size,
        seed=seed,
    )


def load_cifar100(
    data_dir: Optional[str] = None,
    hyperval_size: int = 5000,
    seed: int = 0,
) -> DatasetSplits:
    return load_tfds_image_dataset(
        "cifar100",
        data_dir=data_dir,
        hyperval_size=hyperval_size,
        seed=seed,
    )


def load_mnist(
    data_dir: Optional[str] = None,
    hyperval_size: int = 5000,
    seed: int = 0,
) -> DatasetSplits:
    return load_tfds_image_dataset(
        "mnist",
        data_dir=data_dir,
        hyperval_size=hyperval_size,
        seed=seed,
    )


def _mnist_urls(filename: str, mirrors: Sequence[str]) -> tuple[str, ...]:
    return tuple(f"{mirror.rstrip('/')}/{filename}" for mirror in mirrors)


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected MNIST image magic {magic} in {path}.")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    images = data.reshape(count, rows, cols, 1)
    return _as_float_images(images)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected MNIST label magic {magic} in {path}.")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    if labels.shape[0] != count:
        raise ValueError(
            f"MNIST label count mismatch in {path}: expected {count}, got {labels.shape[0]}."
        )
    return labels.astype(np.int32)


def load_mnist_direct(
    data_dir: str = ".data/mnist_raw",
    download_mirrors: Optional[Sequence[str]] = None,
    download_timeout_seconds: int = 60,
    resource_md5s: Optional[Dict[str, str]] = None,
    hyperval_size: int = 5000,
    seed: int = 0,
) -> DatasetSplits:
    root = Path(data_dir)
    paths = {}
    resources = resource_md5s or _MNIST_RESOURCES
    mirrors = download_mirrors or _MNIST_MIRRORS
    for filename, md5 in resources.items():
        destination = root / filename
        _download_if_missing(
            _mnist_urls(filename, mirrors),
            destination,
            timeout_seconds=download_timeout_seconds,
            expected_md5=md5,
        )
        paths[filename] = destination

    return _split_train_hyperval(
        train_images=_read_idx_images(paths["train-images-idx3-ubyte.gz"]),
        train_labels=_read_idx_labels(paths["train-labels-idx1-ubyte.gz"]),
        test_images=_read_idx_images(paths["t10k-images-idx3-ubyte.gz"]),
        test_labels=_read_idx_labels(paths["t10k-labels-idx1-ubyte.gz"]),
        hyperval_size=hyperval_size,
        seed=seed,
    )


_CIFAR10_URLS = (
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
)
_CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"
_CIFAR100_URLS = (
    "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
    "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
)
_CIFAR100_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"
_MNIST_MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
)
_MNIST_RESOURCES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def file_md5(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()  # noqa: S324 - dataset integrity checksum, not security.
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_md5(path: Path, expected_md5: Optional[str]) -> None:
    if expected_md5 is None:
        return
    actual_md5 = file_md5(path)
    if actual_md5.lower() != expected_md5.lower():
        raise RuntimeError(
            f"Archive checksum mismatch for {path}: expected md5={expected_md5}, got {actual_md5}."
        )


def _download_if_missing(
    urls: Sequence[str],
    destination: Path,
    timeout_seconds: int = 60,
    expected_md5: Optional[str] = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _verify_md5(destination, expected_md5)
        return

    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    errors = []
    for url in urls:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
            started_at = time.monotonic()
            with requests.get(
                url,
                stream=True,
                timeout=(min(15, timeout_seconds), min(15, timeout_seconds)),
            ) as response:
                response.raise_for_status()
                with tmp_path.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if time.monotonic() - started_at > timeout_seconds:
                            raise TimeoutError(
                                f"Download exceeded {timeout_seconds} seconds."
                            )
                        if chunk:
                            f.write(chunk)
            tmp_path.replace(destination)
            _verify_md5(destination, expected_md5)
            return
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url}: {exc}")

    if tmp_path.exists():
        tmp_path.unlink()
    raise RuntimeError(
        "Could not download CIFAR archive. "
        f"Place it manually at {destination} or configure data.archive_path. "
        f"Tried: {'; '.join(errors)}"
    )


def _load_pickle_from_tar(archive: Path, member_name: str) -> Dict:
    with tarfile.open(archive, "r:gz") as tar:
        member = tar.extractfile(member_name)
        if member is None:
            raise FileNotFoundError(f"{member_name} not found in {archive}.")
        return pickle.load(member, encoding="bytes")


def _cifar_array(data: np.ndarray) -> np.ndarray:
    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return _as_float_images(images)


def load_cifar10_direct(
    data_dir: str = ".data/cifar_raw",
    archive_path: Optional[str] = None,
    download_urls: Sequence[str] = _CIFAR10_URLS,
    download_timeout_seconds: int = 60,
    archive_md5: Optional[str] = None,
    hyperval_size: int = 5000,
    seed: int = 0,
) -> DatasetSplits:
    archive = Path(archive_path) if archive_path else Path(data_dir) / "cifar-10-python.tar.gz"
    _download_if_missing(
        download_urls,
        archive,
        timeout_seconds=download_timeout_seconds,
        expected_md5=archive_md5,
    )

    train_images = []
    train_labels = []
    for i in range(1, 6):
        batch = _load_pickle_from_tar(archive, f"cifar-10-batches-py/data_batch_{i}")
        train_images.append(_cifar_array(batch[b"data"]))
        train_labels.append(np.asarray(batch[b"labels"], dtype=np.int32))

    test = _load_pickle_from_tar(archive, "cifar-10-batches-py/test_batch")
    return _split_train_hyperval(
        train_images=np.concatenate(train_images, axis=0),
        train_labels=np.concatenate(train_labels, axis=0),
        test_images=_cifar_array(test[b"data"]),
        test_labels=np.asarray(test[b"labels"], dtype=np.int32),
        hyperval_size=hyperval_size,
        seed=seed,
    )


def load_cifar100_direct(
    data_dir: str = ".data/cifar_raw",
    archive_path: Optional[str] = None,
    download_urls: Sequence[str] = _CIFAR100_URLS,
    download_timeout_seconds: int = 60,
    archive_md5: Optional[str] = None,
    hyperval_size: int = 5000,
    seed: int = 0,
) -> DatasetSplits:
    archive = Path(archive_path) if archive_path else Path(data_dir) / "cifar-100-python.tar.gz"
    _download_if_missing(
        download_urls,
        archive,
        timeout_seconds=download_timeout_seconds,
        expected_md5=archive_md5,
    )

    train = _load_pickle_from_tar(archive, "cifar-100-python/train")
    test = _load_pickle_from_tar(archive, "cifar-100-python/test")
    return _split_train_hyperval(
        train_images=_cifar_array(train[b"data"]),
        train_labels=np.asarray(train[b"fine_labels"], dtype=np.int32),
        test_images=_cifar_array(test[b"data"]),
        test_labels=np.asarray(test[b"fine_labels"], dtype=np.int32),
        hyperval_size=hyperval_size,
        seed=seed,
    )


def load_synthetic_cifar(
    train_size: int = 64,
    hyperval_size: int = 16,
    test_size: int = 16,
    num_classes: int = 10,
    seed: int = 0,
) -> DatasetSplits:
    rng = np.random.default_rng(seed)
    prototypes = rng.uniform(0.1, 0.9, size=(num_classes, 1, 1, 3)).astype(np.float32)

    def make_split(size: int) -> tuple[np.ndarray, np.ndarray]:
        labels = rng.integers(0, num_classes, size=(size,), dtype=np.int32)
        noise = rng.normal(0.0, 0.08, size=(size, 32, 32, 3)).astype(np.float32)
        images = np.clip(prototypes[labels] + noise, 0.0, 1.0)
        return images, labels

    train_images, train_labels = make_split(train_size + hyperval_size)
    test_images, test_labels = make_split(test_size)
    return _split_train_hyperval(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        hyperval_size=hyperval_size,
        seed=seed,
    )


def load_synthetic_mnist(
    train_size: int = 64,
    hyperval_size: int = 16,
    test_size: int = 16,
    num_classes: int = 10,
    seed: int = 0,
) -> DatasetSplits:
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, 28, dtype=np.float32),
        np.linspace(-1.0, 1.0, 28, dtype=np.float32),
        indexing="ij",
    )
    centers = rng.uniform(-0.65, 0.65, size=(num_classes, 2)).astype(np.float32)
    prototypes = []
    for class_id, (center_y, center_x) in enumerate(centers):
        blob = np.exp(-((yy - center_y) ** 2 + (xx - center_x) ** 2) / (2.0 * 0.18**2))
        stripe = ((xx + 1.0) * (class_id + 1) * 0.07) % 0.25
        prototypes.append(np.clip(blob + stripe, 0.0, 1.0))
    prototypes_array = np.asarray(prototypes, dtype=np.float32)[..., None]

    def make_split(size: int) -> tuple[np.ndarray, np.ndarray]:
        labels = rng.integers(0, num_classes, size=(size,), dtype=np.int32)
        noise = rng.normal(0.0, 0.06, size=(size, 28, 28, 1)).astype(np.float32)
        images = np.clip(prototypes_array[labels] + noise, 0.0, 1.0)
        return images, labels

    train_images, train_labels = make_split(train_size + hyperval_size)
    test_images, test_labels = make_split(test_size)
    return _split_train_hyperval(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        hyperval_size=hyperval_size,
        seed=seed,
    )


def load_imagenet_subset(
    data_dir: Optional[str] = None,
    hyperval_size: int = 50000,
    max_train_size: Optional[int] = None,
    max_test_size: Optional[int] = None,
    image_size: int = 224,
    seed: int = 0,
) -> DatasetSplits:
    if max_train_size is None:
        raise ValueError(
            "The in-memory ImageNet loader requires data.max_train_size. "
            "Use a subset for debugging; paper-scale ImageNet needs a streaming pipeline."
        )

    import tensorflow as tf
    import tensorflow_datasets as tfds

    def preprocess(record):
        image = tf.image.convert_image_dtype(record["image"], tf.float32)
        image = tf.image.resize(image, (image_size, image_size), antialias=True)
        return {
            "image": image,
            "label": tf.cast(record["label"], tf.int32),
        }

    def load_split(split: str, count: int) -> tuple[np.ndarray, np.ndarray]:
        dataset = tfds.load(
            "imagenet2012",
            split=split,
            data_dir=data_dir,
            shuffle_files=False,
        )
        dataset = dataset.take(count).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(count)
        batch = next(iter(tfds.as_numpy(dataset)))
        return batch["image"].astype(np.float32), batch["label"].astype(np.int32)

    train_take = max_train_size + hyperval_size
    train_images, train_labels = load_split("train", train_take)
    test_take = max_test_size if max_test_size is not None else 50000
    test_images, test_labels = load_split("validation", test_take)
    return _split_train_hyperval(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        hyperval_size=hyperval_size,
        seed=seed,
    )


def load_synthetic_imagenet(
    train_size: int = 8,
    hyperval_size: int = 4,
    test_size: int = 4,
    num_classes: int = 1000,
    image_size: int = 64,
    seed: int = 0,
) -> DatasetSplits:
    rng = np.random.default_rng(seed)
    prototypes = rng.uniform(0.1, 0.9, size=(num_classes, 1, 1, 3)).astype(np.float32)

    def make_split(size: int) -> tuple[np.ndarray, np.ndarray]:
        labels = rng.integers(0, num_classes, size=(size,), dtype=np.int32)
        noise = rng.normal(
            0.0,
            0.08,
            size=(size, image_size, image_size, 3),
        ).astype(np.float32)
        images = np.clip(prototypes[labels] + noise, 0.0, 1.0)
        return images, labels

    train_images, train_labels = make_split(train_size + hyperval_size)
    test_images, test_labels = make_split(test_size)
    return _split_train_hyperval(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        hyperval_size=hyperval_size,
        seed=seed,
    )


def _limit_splits(splits: DatasetSplits, config: Dict, seed: int) -> DatasetSplits:
    rng = np.random.default_rng(seed)

    def subset(images: np.ndarray, labels: np.ndarray, max_size: Optional[int]):
        if max_size is None or max_size >= len(images):
            return images, labels
        if max_size <= 0:
            raise ValueError(f"Subset size must be positive, got {max_size}.")
        indices = rng.permutation(len(images))[:max_size]
        return images[indices], labels[indices]

    def subset_per_class(images: np.ndarray, labels: np.ndarray, count: Optional[int]):
        if count is None:
            return images, labels
        if count <= 0:
            raise ValueError(f"train_labels_per_class must be positive, got {count}.")

        class_ids = np.unique(np.concatenate([splits.train_labels, splits.hyperval_labels, splits.test_labels]))
        selected = []
        for class_id in class_ids:
            class_indices = np.flatnonzero(labels == class_id)
            if len(class_indices) < count:
                raise ValueError(
                    f"Class {int(class_id)} only has {len(class_indices)} training samples, "
                    f"cannot select train_labels_per_class={count}."
                )
            selected.append(rng.permutation(class_indices)[:count])

        indices = rng.permutation(np.concatenate(selected))
        return images[indices], labels[indices]

    if config.get("train_labels_per_class") is not None and config.get("max_train_size") is not None:
        raise ValueError("Use either train_labels_per_class or max_train_size, not both.")

    train_images, train_labels = subset_per_class(
        splits.train_images,
        splits.train_labels,
        config.get("train_labels_per_class"),
    )
    train_images, train_labels = subset(train_images, train_labels, config.get("max_train_size"))
    hyperval_images, hyperval_labels = subset(
        splits.hyperval_images,
        splits.hyperval_labels,
        config.get("max_hyperval_size"),
    )
    test_images, test_labels = subset(
        splits.test_images,
        splits.test_labels,
        config.get("max_test_size"),
    )
    return DatasetSplits(
        train_images=train_images,
        train_labels=train_labels,
        hyperval_images=hyperval_images,
        hyperval_labels=hyperval_labels,
        test_images=test_images,
        test_labels=test_labels,
    )


def load_dataset(config: Dict, seed: int = 0) -> DatasetSplits:
    dataset_name = config.get("name", "cifar10")
    default_source = "tfds" if dataset_name == "imagenet" else "direct"
    source = config.get("source", default_source)
    if dataset_name == "cifar10":
        if source == "tfds":
            splits = load_cifar10(
                data_dir=config.get("data_dir"),
                hyperval_size=config["hyperval_size"],
                seed=seed,
            )
        elif source == "direct":
            splits = load_cifar10_direct(
                data_dir=config.get("raw_data_dir", ".data/cifar_raw"),
                archive_path=config.get("archive_path"),
                download_urls=config.get("download_urls", _CIFAR10_URLS),
                download_timeout_seconds=config.get("download_timeout_seconds", 60),
                archive_md5=config.get("archive_md5"),
                hyperval_size=config["hyperval_size"],
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown CIFAR-10 source: {source}")
        return _limit_splits(splits, config, seed)
    if dataset_name == "cifar100":
        if source == "tfds":
            splits = load_cifar100(
                data_dir=config.get("data_dir"),
                hyperval_size=config["hyperval_size"],
                seed=seed,
            )
        elif source == "direct":
            splits = load_cifar100_direct(
                data_dir=config.get("raw_data_dir", ".data/cifar_raw"),
                archive_path=config.get("archive_path"),
                download_urls=config.get("download_urls", _CIFAR100_URLS),
                download_timeout_seconds=config.get("download_timeout_seconds", 60),
                archive_md5=config.get("archive_md5"),
                hyperval_size=config["hyperval_size"],
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown CIFAR-100 source: {source}")
        return _limit_splits(splits, config, seed)
    if dataset_name == "mnist":
        if source == "tfds":
            splits = load_mnist(
                data_dir=config.get("data_dir"),
                hyperval_size=config["hyperval_size"],
                seed=seed,
            )
        elif source == "direct":
            splits = load_mnist_direct(
                data_dir=config.get("raw_data_dir", ".data/mnist_raw"),
                download_mirrors=config.get("download_mirrors", _MNIST_MIRRORS),
                download_timeout_seconds=config.get("download_timeout_seconds", 60),
                resource_md5s=config.get("resource_md5s"),
                hyperval_size=config["hyperval_size"],
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown MNIST source: {source}")
        return _limit_splits(splits, config, seed)
    if dataset_name == "imagenet":
        if source != "tfds":
            raise ValueError(f"Unknown ImageNet source: {source}")
        splits = load_imagenet_subset(
            data_dir=config.get("data_dir"),
            hyperval_size=config["hyperval_size"],
            max_train_size=config.get("max_train_size"),
            max_test_size=config.get("max_test_size"),
            image_size=config.get("image_size", 224),
            seed=seed,
        )
        return _limit_splits(splits, config, seed)
    if dataset_name == "synthetic_cifar":
        splits = load_synthetic_cifar(
            train_size=config.get("train_size", 64),
            hyperval_size=config.get("hyperval_size", 16),
            test_size=config.get("test_size", 16),
            num_classes=config.get("num_classes", 10),
            seed=seed,
        )
        return _limit_splits(splits, config, seed)
    if dataset_name == "synthetic_mnist":
        splits = load_synthetic_mnist(
            train_size=config.get("train_size", 64),
            hyperval_size=config.get("hyperval_size", 16),
            test_size=config.get("test_size", 16),
            num_classes=config.get("num_classes", 10),
            seed=seed,
        )
        return _limit_splits(splits, config, seed)
    if dataset_name == "synthetic_imagenet":
        splits = load_synthetic_imagenet(
            train_size=config.get("train_size", 8),
            hyperval_size=config.get("hyperval_size", 4),
            test_size=config.get("test_size", 4),
            num_classes=config.get("num_classes", 1000),
            image_size=config.get("image_size", 64),
            seed=seed,
        )
        return _limit_splits(splits, config, seed)
    raise ValueError(f"Unknown dataset: {dataset_name}")

from pathlib import Path
import copy
import gzip
import io
import json
import pickle
import struct
import subprocess
import sys
import tarfile
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from jax.flatten_util import ravel_pytree

from classification_network import (
    ClassifierTrainState,
    ImageNetResNet,
    MnistConvNet,
    PreActResNet,
    PreActResNet18,
    PyramidNetShakeDrop,
    ShakeShakeResNet,
    WideResNet,
    classifier_eval_step,
    classifier_train_step,
    classifier_train_step_with_augnet,
    create_classifier_state,
    extract_classifier_features,
)
from data import (
    NumpyBatchIterator,
    imagenet_stream_info,
    load_cifar10_direct,
    load_cifar100_direct,
    load_dataset,
    load_mnist_direct,
    load_synthetic_imagenet,
    load_synthetic_mnist,
)
from data.cifar import file_md5
from data.imagenet import _center_crop_numpy, _resize_short_side_for_eval
from paramyield_network import (
    augnet_influence_train_step,
    compute_batch_s_test,
    compute_batch_s_test_residual,
)
from paramyield_network.influence import (
    classifier_grad,
    classifier_logits,
    classifier_loss,
    compute_s_test,
    influence_up_loss,
    last_layer_grad_per_example,
    s_test_residual_norm,
)
from transformation_network import (
    CIFARAugmentationNetwork,
    FeatureDiscriminator,
    ImageDiscriminator,
    augnet_pretrain_step,
    apply_spatial_transform,
    average_pool_same,
    create_augnet_state,
    create_discriminator_state,
)
from classification_network.engine import _kernel_decay_mask
from scripts.train_cifar10 import (
    build_learning_rate,
    evaluate,
    progressive_image_size_for_step,
    steps_per_epoch,
    stop_step_for_stage,
)
from scripts.collect_results import collect_config_result, output_dir_from_arg
from scripts.compare_to_paper import (
    compare_suite_targets,
    compare_targets,
    filter_paper_targets,
    load_paper_targets,
    output_dir_from_arg as compare_output_dir_from_arg,
)
from scripts.preflight import preflight_configs
from scripts.run_paper_suite import command_for_run, materialize_suite_configs
from scripts.sweep_retrain_probability import _probability_label, _summarize_results
from scripts.suite_status import collect_suite_status, summarize as summarize_suite_status
from utils import load_config, restore_state, save_state, validate_config


def _add_pickle_member(tar: tarfile.TarFile, name: str, payload: dict) -> None:
    data = pickle.dumps(payload)
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _flat_cifar_images(count: int, offset: int = 0) -> np.ndarray:
    values = np.arange(count * 3 * 32 * 32, dtype=np.uint32)
    values = (values + offset) % 256
    return values.astype(np.uint8).reshape(count, 3 * 32 * 32)


def _write_fake_cifar10(path: Path, train_batches: int = 5, batch_size: int = 3) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for batch_idx in range(1, train_batches + 1):
            labels = [(batch_idx + i) % 10 for i in range(batch_size)]
            _add_pickle_member(
                tar,
                f"cifar-10-batches-py/data_batch_{batch_idx}",
                {
                    b"data": _flat_cifar_images(batch_size, offset=batch_idx),
                    b"labels": labels,
                },
            )
        _add_pickle_member(
            tar,
            "cifar-10-batches-py/test_batch",
            {
                b"data": _flat_cifar_images(4, offset=99),
                b"labels": [0, 1, 2, 3],
            },
        )


def _write_fake_cifar100(path: Path, train_size: int = 12, test_size: int = 5) -> None:
    with tarfile.open(path, "w:gz") as tar:
        _add_pickle_member(
            tar,
            "cifar-100-python/train",
            {
                b"data": _flat_cifar_images(train_size, offset=7),
                b"fine_labels": [i % 100 for i in range(train_size)],
            },
        )
        _add_pickle_member(
            tar,
            "cifar-100-python/test",
            {
                b"data": _flat_cifar_images(test_size, offset=11),
                b"fine_labels": [i % 100 for i in range(test_size)],
            },
        )


def _write_fake_mnist_images(path: Path, count: int, offset: int = 0) -> None:
    values = np.arange(count * 28 * 28, dtype=np.uint32)
    images = ((values + offset) % 256).astype(np.uint8)
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">IIII", 2051, count, 28, 28))
        f.write(images.tobytes())


def _write_fake_mnist_labels(path: Path, count: int, offset: int = 0) -> None:
    labels = ((np.arange(count, dtype=np.uint32) + offset) % 10).astype(np.uint8)
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">II", 2049, count))
        f.write(labels.tobytes())


def _write_fake_mnist_raw(path: Path, train_size: int = 20, test_size: int = 6) -> dict:
    path.mkdir(parents=True, exist_ok=True)
    files = {
        "train-images-idx3-ubyte.gz": lambda p: _write_fake_mnist_images(p, train_size, offset=1),
        "train-labels-idx1-ubyte.gz": lambda p: _write_fake_mnist_labels(p, train_size, offset=0),
        "t10k-images-idx3-ubyte.gz": lambda p: _write_fake_mnist_images(p, test_size, offset=99),
        "t10k-labels-idx1-ubyte.gz": lambda p: _write_fake_mnist_labels(p, test_size, offset=3),
    }
    for filename, writer in files.items():
        writer(path / filename)
    return {filename: file_md5(path / filename) for filename in files}


def _assert_cifar_direct_loaders() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cifar10_archive = tmp_path / "cifar-10-python.tar.gz"
        cifar100_archive = tmp_path / "cifar-100-python.tar.gz"
        _write_fake_cifar10(cifar10_archive)
        _write_fake_cifar100(cifar100_archive)

        cifar10 = load_cifar10_direct(
            archive_path=str(cifar10_archive),
            archive_md5=file_md5(cifar10_archive),
            hyperval_size=5,
            seed=123,
        )
        assert cifar10.train_images.shape == (10, 32, 32, 3)
        assert cifar10.hyperval_images.shape == (5, 32, 32, 3)
        assert cifar10.test_images.shape == (4, 32, 32, 3)
        assert cifar10.train_images.dtype == np.float32
        assert 0.0 <= float(cifar10.train_images.min()) <= 1.0
        assert 0.0 <= float(cifar10.train_images.max()) <= 1.0

        cifar100 = load_cifar100_direct(
            archive_path=str(cifar100_archive),
            archive_md5=file_md5(cifar100_archive),
            hyperval_size=4,
            seed=123,
        )
        assert cifar100.train_images.shape == (8, 32, 32, 3)
        assert cifar100.hyperval_images.shape == (4, 32, 32, 3)
        assert cifar100.test_labels.shape == (5,)

        try:
            load_cifar10_direct(
                archive_path=str(cifar10_archive),
                archive_md5="0" * 32,
                hyperval_size=5,
                seed=123,
            )
        except RuntimeError as exc:
            assert "checksum mismatch" in str(exc)
        else:
            raise AssertionError("load_cifar10_direct accepted a bad archive checksum.")


def _assert_balanced_train_subset() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "cifar-10-python.tar.gz"
        _write_fake_cifar10(archive, batch_size=20)
        splits = load_dataset(
            {
                "name": "cifar10",
                "source": "direct",
                "archive_path": str(archive),
                "hyperval_size": 1,
                "train_labels_per_class": 5,
            },
            seed=123,
        )

    assert splits.train_images.shape == (50, 32, 32, 3)
    counts = np.bincount(splits.train_labels, minlength=10)
    np.testing.assert_array_equal(counts, np.full((10,), 5))


def _assert_synthetic_mnist_loader() -> None:
    splits = load_synthetic_mnist(
        train_size=20,
        hyperval_size=10,
        test_size=6,
        seed=123,
    )
    assert splits.train_images.shape == (20, 28, 28, 1)
    assert splits.hyperval_images.shape == (10, 28, 28, 1)
    assert splits.test_images.shape == (6, 28, 28, 1)
    assert splits.train_images.dtype == np.float32
    assert 0.0 <= float(splits.train_images.min()) <= 1.0
    assert 0.0 <= float(splits.train_images.max()) <= 1.0


def _assert_mnist_direct_loader() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = Path(tmp) / "mnist_raw"
        resource_md5s = _write_fake_mnist_raw(raw_dir)
        splits = load_mnist_direct(
            data_dir=str(raw_dir),
            resource_md5s=resource_md5s,
            hyperval_size=5,
            seed=123,
        )
        assert splits.train_images.shape == (15, 28, 28, 1)
        assert splits.hyperval_images.shape == (5, 28, 28, 1)
        assert splits.test_images.shape == (6, 28, 28, 1)
        assert splits.train_images.dtype == np.float32
        assert splits.train_labels.dtype == np.int32
        assert 0.0 <= float(splits.train_images.min()) <= 1.0
        assert 0.0 <= float(splits.train_images.max()) <= 1.0

        via_config = load_dataset(
            {
                "name": "mnist",
                "source": "direct",
                "raw_data_dir": str(raw_dir),
                "resource_md5s": resource_md5s,
                "hyperval_size": 5,
            },
            seed=123,
        )
        np.testing.assert_array_equal(via_config.test_labels, splits.test_labels)


def _assert_synthetic_imagenet_loader() -> None:
    splits = load_synthetic_imagenet(
        train_size=4,
        hyperval_size=2,
        test_size=2,
        image_size=64,
        seed=123,
    )
    assert splits.train_images.shape == (4, 64, 64, 3)
    assert splits.hyperval_images.shape == (2, 64, 64, 3)
    assert splits.test_images.shape == (2, 64, 64, 3)
    assert splits.train_labels.dtype == np.int32
    assert splits.train_images.dtype == np.float32
    assert 0.0 <= float(splits.train_images.min()) <= 1.0
    assert 0.0 <= float(splits.train_images.max()) <= 1.0


def _assert_imagenet_stream_info() -> None:
    cfg = load_config("configs/imagenet_stream_debug.yaml")
    validate_config(cfg)
    info = imagenet_stream_info(cfg["data"])
    assert info.input_shape == (1, 64, 64, 3)
    assert info.train_examples == 4
    assert info.hyperval_examples == 2
    assert info.validation_examples == 2
    assert info.train_split == "train[2:6]"
    assert info.hyperval_split == "train[:2]"
    assert info.validation_split == "validation[:2]"
    assert _resize_short_side_for_eval(224) == 256

    image = np.arange(5 * 7 * 3, dtype=np.float32).reshape(5, 7, 3)
    cropped = _center_crop_numpy(image, 3, 3)
    np.testing.assert_array_equal(cropped, image[1:4, 2:5])


def _assert_imagenet_dry_run_no_side_effects() -> None:
    cfg = load_config("configs/imagenet_stream_debug.yaml")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        checkpoint_dir = tmp_path / "imagenet_dry_run"
        cfg["checkpoint_dir"] = str(checkpoint_dir)
        cfg_path = tmp_path / "imagenet_stream_debug.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        result = subprocess.run(
            [
                sys.executable,
                "scripts/train_imagenet_stream.py",
                "--config",
                str(cfg_path),
                "--dry-run",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        assert "imagenet_stream_dry_run" in result.stdout
        assert "classifier_steps_per_epoch=" in result.stdout
        assert "retrain_steps_per_epoch=" in result.stdout
        assert "eval_batches=" in result.stdout
        assert not checkpoint_dir.exists()


def _assert_progressive_schedule() -> None:
    config = {
        "progressive_image_sizes": [32, 64, 128, 224],
        "progressive_boundaries": [10, 20, 30],
    }
    assert progressive_image_size_for_step(config, 0, 224) == 32
    assert progressive_image_size_for_step(config, 10, 224) == 64
    assert progressive_image_size_for_step(config, 20, 224) == 128
    assert progressive_image_size_for_step(config, 30, 224) is None
    assert progressive_image_size_for_step({}, 0, 224) is None


def _assert_cosine_lr_schedule() -> None:
    schedule = build_learning_rate(
        {
            "learning_rate": 0.1,
            "lr_schedule": "cosine",
            "min_learning_rate": 0.0,
            "epochs": 2,
        },
        steps_per_epoch=5,
    )
    np.testing.assert_allclose(float(schedule(0)), 0.1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(float(schedule(10)), 0.0, rtol=1e-6, atol=1e-6)
    mid_value = float(schedule(5))
    assert 0.0 < mid_value < 0.1


def _assert_training_epoch_coverage() -> None:
    assert steps_per_epoch(100, 64) == 2
    assert steps_per_epoch(100, 64, drop_last=True) == 1
    assert steps_per_epoch(45_000, 128) == 352
    assert steps_per_epoch(45_000, 128, drop_last=True) == 351
    assert stop_step_for_stage(10, 100, None) == 100
    assert stop_step_for_stage(10, 100, 25) == 35
    assert stop_step_for_stage(90, 100, 25) == 100


def _assert_weight_decay_mask() -> None:
    params = {
        "conv": {
            "kernel": jnp.ones((3, 3, 3, 8), dtype=jnp.float32),
            "bias": jnp.ones((8,), dtype=jnp.float32),
        },
        "bn": {
            "scale": jnp.ones((8,), dtype=jnp.float32),
            "bias": jnp.ones((8,), dtype=jnp.float32),
        },
        "classifier": {
            "kernel": jnp.ones((8, 10), dtype=jnp.float32),
            "bias": jnp.ones((10,), dtype=jnp.float32),
        },
    }
    mask = _kernel_decay_mask(params)
    assert mask["conv"]["kernel"] is True
    assert mask["conv"]["bias"] is False
    assert mask["bn"]["scale"] is False
    assert mask["bn"]["bias"] is False
    assert mask["classifier"]["kernel"] is True
    assert mask["classifier"]["bias"] is False


def _assert_iterator_resume() -> None:
    images = np.arange(30, dtype=np.float32).reshape(10, 3)
    labels = np.arange(10, dtype=np.int32)

    iterator = NumpyBatchIterator(images, labels, batch_size=4, seed=5)
    first_batch = next(iterator)
    state = copy.deepcopy(iterator.state_dict())
    expected_second = next(iterator)

    restored = NumpyBatchIterator(images, labels, batch_size=4, seed=999)
    restored.load_state_dict(state)
    actual_second = next(restored)

    assert first_batch["image"].shape == (4, 3)
    np.testing.assert_array_equal(actual_second["image"], expected_second["image"])
    np.testing.assert_array_equal(actual_second["label"], expected_second["label"])

    eval_iterator = NumpyBatchIterator(
        images,
        labels,
        batch_size=4,
        seed=5,
        shuffle=False,
        drop_last=False,
    )
    partial_batches = [next(eval_iterator), next(eval_iterator), next(eval_iterator)]
    assert [batch["label"].shape[0] for batch in partial_batches] == [4, 4, 2]


class _AlwaysClassZeroModel:
    def apply(self, variables, images, train=False, return_features=False, **kwargs):
        logits = jnp.stack(
            [
                jnp.ones((images.shape[0],), dtype=jnp.float32),
                jnp.zeros((images.shape[0],), dtype=jnp.float32),
            ],
            axis=-1,
        )
        if return_features:
            return images.reshape((images.shape[0], -1)), logits
        return logits


class _FeatureIdentityModel:
    def apply(self, variables, images, train=False, return_features=False, **kwargs):
        features = images
        params = variables["params"]["classifier"]
        logits = features @ params["kernel"]
        if "bias" in params:
            logits = logits + params["bias"]
        if return_features:
            return features, logits
        return logits


def _assert_evaluate_weights_partial_batches() -> None:
    images = np.zeros((3, 1, 1, 1), dtype=np.float32)
    labels = np.asarray([0, 1, 1], dtype=np.int32)
    iterator = NumpyBatchIterator(
        images,
        labels,
        batch_size=2,
        seed=0,
        shuffle=False,
        drop_last=False,
    )
    state = ClassifierTrainState.create(
        apply_fn=lambda *args, **kwargs: None,
        params={},
        tx=optax.sgd(0.0),
        batch_stats=None,
    )
    metrics = evaluate(
        state,
        _AlwaysClassZeroModel(),
        iterator,
        batches=2,
    )
    assert abs(metrics["accuracy"] - (1.0 / 3.0)) < 1e-6


def _assert_config_validation() -> None:
    cfg = load_config("configs/synthetic_debug.yaml")
    validate_config(cfg)

    bad_cfg = copy.deepcopy(cfg)
    bad_cfg["classifier"]["num_classes"] = 11
    try:
        validate_config(bad_cfg)
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("validate_config accepted a mismatched class count.")


def _assert_all_configs_validate() -> None:
    config_paths = sorted(Path("configs").glob("*.yaml"))
    assert config_paths, "No config files found."
    for path in config_paths:
        config = load_config(str(path))
        if not all(key in config for key in ("data", "classifier", "pretrain", "augnet")):
            continue
        validate_config(config)


def _assert_collect_results() -> None:
    cfg = load_config("configs/synthetic_debug.yaml")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        cfg["checkpoint_dir"] = str(run_dir)
        cfg_path = tmp_path / "synthetic_config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        (run_dir / "metrics.jsonl").write_text(
            "\n".join(
                [
                    '{"metrics":{"accuracy":0.25,"loss":2.0,"top5_accuracy":0.75},"stage":"classifier_eval"}',
                    '{"metrics":{"batches":2,"cg_iters":7,"damping":0.01,"residual_mean":0.001,"residual_max":0.002},"stage":"precompute_s_test"}',
                    '{"metrics":{"pretrain_identity_l2":0.03,"pretrain_tau_abs_mean":0.04},"stage":"pretrain_augnet_last"}',
                    '{"metrics":{"estimated_val_loss_reduction":0.05,"identity_l2":0.06,"tau_abs_mean":0.07},"stage":"augnet_last"}',
                    '{"metrics":{"accuracy":0.375,"loss":1.5,"top5_accuracy":0.875},"stage":"retrained_classifier_eval"}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        row = collect_config_result(str(cfg_path))

    assert row["status"] == "ok"
    assert row["dataset"] == "synthetic_cifar"
    assert row["train_examples"] == 16
    assert row["test_examples"] == 8
    assert row["classifier_batch_size"] == 8
    assert row["eval_batches"] == 1
    assert row["expected_eval_batches"] == 1
    assert row["classifier_steps_per_epoch"] == 2
    assert row["classifier_total_steps"] == 2
    assert row["retrain_steps_per_epoch"] == 2
    assert row["retrain_total_steps"] == 2
    assert row["baseline_error"] == 75.0
    assert row["augnet_error"] == 62.5
    assert row["error_reduction"] == 12.5
    assert row["baseline_top5_accuracy"] == 0.75
    assert row["augnet_top5_accuracy"] == 0.875
    assert row["s_test_batches"] == 2.0
    assert row["s_test_cg_iters"] == 7.0
    assert row["s_test_residual_mean"] == 0.001
    assert row["s_test_residual_max"] == 0.002
    assert row["pretrain_tau_abs_mean"] == 0.04
    assert row["pretrain_identity_l2"] == 0.03
    assert row["augnet_tau_abs_mean"] == 0.07
    assert row["augnet_identity_l2"] == 0.06
    assert row["estimated_val_loss_reduction"] == 0.05

    full_cifar_row = collect_config_result("configs/cifar10_table2_wrn28_10.yaml")
    assert full_cifar_row["train_examples"] == 45_000
    assert full_cifar_row["test_examples"] == 10_000
    assert full_cifar_row["expected_eval_batches"] == 79
    assert full_cifar_row["classifier_steps_per_epoch"] == 352

    assert output_dir_from_arg("runs/table1_results") == Path("runs/table1_results")
    try:
        output_dir_from_arg("runs/table1_results.csv")
    except ValueError as exc:
        assert "--output-dir expects a directory" in str(exc)
    else:
        raise AssertionError("collect_results accepted a file-like --output-dir path.")


def _assert_retrain_probability_sweep_summary() -> None:
    assert _probability_label(0.03) == "p003"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        base_a = tmp_path / "base_seed0.yaml"
        base_b = tmp_path / "base_seed1.yaml"
        base_a.write_text(
            yaml.safe_dump({"retrain": {"learned_aug_probability": 0.10, "learned_aug_input": "baseline"}}),
            encoding="utf-8",
        )
        base_b.write_text(
            yaml.safe_dump({"retrain": {"learned_aug_probability": 0.10, "learned_aug_input": "baseline"}}),
            encoding="utf-8",
        )
        cfg_a = tmp_path / "seed0_p003.yaml"
        cfg_b = tmp_path / "seed1_p003.yaml"
        cfg_c = tmp_path / "seed0_p001.yaml"
        cfg_d = tmp_path / "seed0_p007.yaml"
        cfg_a.write_text(
            yaml.safe_dump(
                {
                    "_sweep": {"base_config": str(base_a)},
                    "retrain": {"learned_aug_probability": 0.03, "learned_aug_input": "raw"},
                }
            ),
            encoding="utf-8",
        )
        cfg_b.write_text(
            yaml.safe_dump(
                {
                    "_sweep": {"base_config": str(base_b)},
                    "retrain": {"learned_aug_probability": 0.03, "learned_aug_input": "raw"},
                }
            ),
            encoding="utf-8",
        )
        cfg_c.write_text(
            yaml.safe_dump({"retrain": {"learned_aug_probability": 0.01}}),
            encoding="utf-8",
        )
        cfg_d.write_text(
            yaml.safe_dump({"retrain": {"learned_aug_probability": 0.07}}),
            encoding="utf-8",
        )
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "results_table.json").write_text(
            json.dumps(
                [
                    {
                        "config": str(cfg_a),
                        "status": "ok",
                        "baseline_error": 5.0,
                        "augnet_error": 4.8,
                        "error_reduction": 0.2,
                    },
                    {
                        "config": str(cfg_b),
                        "status": "ok",
                        "baseline_error": 5.2,
                        "augnet_error": 5.1,
                        "error_reduction": 0.1,
                    },
                    {
                        "config": str(cfg_c),
                        "status": "ok",
                        "baseline_error": 5.0,
                        "augnet_error": 5.3,
                        "error_reduction": -0.3,
                    },
                    {
                        "config": str(cfg_d),
                        "status": "missing_final_eval",
                        "baseline_error": 5.0,
                        "augnet_error": 4.0,
                        "error_reduction": 1.0,
                    },
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        _summarize_results(results_dir / "results_table.json", tmp_path)
        summary = json.loads((tmp_path / "summary_by_probability.json").read_text(encoding="utf-8"))
        by_probability = {row["probability"]: row for row in summary}
        assert by_probability[0.01]["beats_baseline"] is False
        assert by_probability[0.03]["beats_baseline"] is True
        assert by_probability[0.03]["ok_runs"] == 2
        np.testing.assert_allclose(by_probability[0.03]["error_reduction_mean"], 0.15)
        assert (tmp_path / "summary_by_probability.md").exists()
        best = json.loads((tmp_path / "best_probability.json").read_text(encoding="utf-8"))
        assert best["probability"] == 0.03
        assert best["beats_baseline"] is True
        assert best["ok_runs"] == best["runs"]
        assert (tmp_path / "best_probability.md").exists()
        apply_script = tmp_path / "apply_best_probability.py"
        assert apply_script.exists()
        subprocess.run([sys.executable, str(apply_script)], check=True)
        for base_path in (base_a, base_b):
            updated = yaml.safe_load(base_path.read_text(encoding="utf-8"))
            assert updated["retrain"]["learned_aug_probability"] == 0.03
            assert updated["retrain"]["learned_aug_input"] == "raw"


def _assert_compare_to_paper() -> None:
    cfg = load_config("configs/synthetic_debug.yaml")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        cfg["checkpoint_dir"] = str(run_dir)
        cfg_path = tmp_path / "synthetic_config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        (run_dir / "metrics.jsonl").write_text(
            "\n".join(
                [
                    '{"metrics":{"accuracy":0.25,"loss":2.0,"top5_accuracy":0.75},"stage":"classifier_eval"}',
                    '{"metrics":{"accuracy":0.375,"loss":1.5,"top5_accuracy":0.875},"stage":"retrained_classifier_eval"}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        targets = [
            {
                "id": "synthetic_augnet_error",
                "table": "smoke",
                "setting": "Synthetic error",
                "config": str(cfg_path),
                "metric": "augnet_error_percent",
                "direction": "max",
                "target": 63.0,
                "units": "percent",
            },
            {
                "id": "synthetic_augnet_top5",
                "table": "smoke",
                "setting": "Synthetic Top-5",
                "config": str(cfg_path),
                "metric": "augnet_top5_percent",
                "direction": "min",
                "target": 87.0,
                "units": "percent",
            },
        ]
        rows = compare_targets(targets)

    assert [row["status"] for row in rows] == ["pass", "pass"]
    assert rows[0]["actual"] == 62.5
    assert rows[1]["actual"] == 87.5
    assert compare_output_dir_from_arg("runs/paper_compare") == Path("runs/paper_compare")
    try:
        compare_output_dir_from_arg("runs/paper_compare.json")
    except ValueError as exc:
        assert "--output-dir expects a directory" in str(exc)
    else:
        raise AssertionError("compare_to_paper accepted a file-like --output-dir path.")

    paper_targets = load_paper_targets("configs/paper_targets.yaml")
    assert len(paper_targets) == 15
    table1_targets = filter_paper_targets(paper_targets, tables=[1])
    assert len(table1_targets) == 5
    cifar_table2_targets = filter_paper_targets(paper_targets, tables=[2], datasets=["cifar100"])
    assert len(cifar_table2_targets) == 3
    target_id_subset = filter_paper_targets(paper_targets, target_ids=["table3_imagenet_resnet50_top1"])
    assert target_id_subset[0]["config"] == "configs/imagenet_resnet50_paper.yaml"
    for target in paper_targets:
        assert Path(target["config"]).exists()
        assert target["metric"] in {
            "baseline_error_percent",
            "augnet_error_percent",
            "baseline_top1_percent",
            "augnet_top1_percent",
            "baseline_top5_percent",
            "augnet_top5_percent",
        }


def _assert_paper_suite() -> None:
    cfg = load_config("configs/synthetic_debug.yaml")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        base_cfg_path = tmp_path / "synthetic_base.yaml"
        cfg["checkpoint_dir"] = str(tmp_path / "base_run")
        base_cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        target = {
            "id": "synthetic_augnet_error",
            "table": "smoke",
            "setting": "Synthetic error",
            "config": str(base_cfg_path),
            "metric": "augnet_error_percent",
            "direction": "max",
            "target": 60.0,
            "units": "percent",
        }
        top5_target = {
            "id": "synthetic_augnet_top5",
            "table": "smoke",
            "setting": "Synthetic Top-5",
            "config": str(base_cfg_path),
            "metric": "augnet_top5_percent",
            "direction": "min",
            "target": 80.0,
            "units": "percent",
        }
        plan = materialize_suite_configs([target, top5_target], tmp_path / "suite", seeds=[0, 1])
        assert len(plan) == 2
        assert plan[0]["runner"] == "scripts/train.py"
        assert plan[0]["target_ids"] == ["synthetic_augnet_error", "synthetic_augnet_top5"]
        assert [item["metric"] for item in plan[0]["targets"]] == [
            "augnet_error_percent",
            "augnet_top5_percent",
        ]
        assert command_for_run(plan[0], stage="all", resume=True)[-1] == "--resume"
        chunk_command = command_for_run(
            plan[0],
            stage="pretrain_augnet",
            resume=True,
            stop_after_steps=25,
        )
        assert chunk_command[-2:] == ["--stop-after-steps", "25"]
        imagenet_item = {
            "runner": "scripts/train_imagenet_stream.py",
            "config": "configs/imagenet_stream_debug.yaml",
        }
        assert command_for_run(imagenet_item, stage="all", resume=True)[-1] == "--resume"

        accuracies = [0.40, 0.45]
        for item, accuracy_value in zip(plan, accuracies):
            generated = load_config(item["config"])
            run_dir = Path(generated["checkpoint_dir"])
            run_dir.mkdir(parents=True)
            (run_dir / "metrics.jsonl").write_text(
                "\n".join(
                    [
                        '{"metrics":{"accuracy":0.25,"loss":2.0,"top5_accuracy":0.75},"stage":"classifier_eval"}',
                        json.dumps(
                            {
                                "metrics": {
                                    "accuracy": accuracy_value,
                                    "loss": 1.5,
                                    "top5_accuracy": 0.875,
                                },
                                "stage": "retrained_classifier_eval",
                            },
                            sort_keys=True,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

        rows = compare_suite_targets([target], tmp_path / "suite", seeds=[0, 1])
        assert len(rows) == 1
        assert rows[0]["status"] == "pass"
        assert rows[0]["n"] == 2
        assert rows[0]["expected_runs"] == 2
        assert rows[0]["missing_runs"] == 0
        assert abs(rows[0]["actual"] - 57.5) < 1e-6
        assert rows[0]["actual_std"] is not None

        missing_rows = compare_suite_targets([target], tmp_path / "suite", seeds=[0, 1, 2])
        assert missing_rows[0]["status"] == "missing_seed_metrics"
        assert missing_rows[0]["n"] == 2
        assert missing_rows[0]["missing_runs"] == 1


def _assert_suite_status() -> None:
    cfg = load_config("configs/synthetic_debug.yaml")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        base_cfg_path = tmp_path / "synthetic_base.yaml"
        cfg["checkpoint_dir"] = str(tmp_path / "base_run")
        base_cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        target = {
            "id": "synthetic_augnet_error",
            "table": "smoke",
            "setting": "Synthetic error",
            "config": str(base_cfg_path),
            "metric": "augnet_error_percent",
            "direction": "max",
            "target": 60.0,
            "units": "percent",
        }
        suite_dir = tmp_path / "suite"
        plan = materialize_suite_configs([target], suite_dir, seeds=[0, 1])
        (suite_dir / "suite_plan.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        complete_run = Path(plan[0]["checkpoint_dir"])
        complete_run.mkdir(parents=True)
        (complete_run / "metrics.jsonl").write_text(
            "\n".join(
                [
                    '{"metrics":{"accuracy":0.25,"loss":2.0,"top5_accuracy":0.75},"stage":"classifier_eval"}',
                    '{"metrics":{"batches":2,"cg_iters":7,"damping":0.01,"residual_mean":0.001,"residual_max":0.002},"stage":"precompute_s_test"}',
                    '{"metrics":{"pretrain_identity_l2":0.03,"pretrain_tau_abs_mean":0.04},"stage":"pretrain_augnet_last"}',
                    '{"metrics":{"estimated_val_loss_reduction":0.05,"identity_l2":0.06,"tau_abs_mean":0.07},"stage":"augnet_last"}',
                    '{"metrics":{"accuracy":0.40,"loss":1.5,"top5_accuracy":0.875},"stage":"retrained_classifier_eval"}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        for stage in ("classifier", "pretrain_augnet", "augnet", "retrain"):
            with (complete_run / f"{stage}_progress.pkl").open("wb") as f:
                pickle.dump({"completed": True, "next_step": 2, "total_steps": 2}, f)

        partial_run = Path(plan[1]["checkpoint_dir"])
        partial_run.mkdir(parents=True)
        with (partial_run / "classifier_progress.pkl").open("wb") as f:
            pickle.dump({"completed": False, "next_step": 1, "total_steps": 2}, f)

        rows = collect_suite_status(suite_dir)

    assert len(rows) == 2
    assert rows[0]["run_state"] == "complete"
    assert rows[0]["metric_status"] == "ok"
    assert rows[0]["augnet_error"] == 60.0
    assert rows[0]["s_test_residual_mean"] == 0.001
    assert rows[0]["classifier_completed"] is True
    assert rows[1]["run_state"] == "in_progress"
    assert rows[1]["classifier_next_step"] == 1
    assert rows[1]["retrain_progress_status"] == "missing"
    assert summarize_suite_status(rows) == {"complete": 1, "in_progress": 1}


def _assert_preflight() -> None:
    cfg = load_config("configs/cifar10_debug.yaml")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "cifar-10-python.tar.gz"
        _write_fake_cifar10(archive, batch_size=20)
        cfg["data"]["archive_path"] = str(archive)
        cfg["data"]["archive_md5"] = file_md5(archive)
        cfg_path = tmp_path / "cifar10_debug.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        report = preflight_configs([str(cfg_path)])
        assert report[0]["status"] == "ok"
        checks = {check["id"]: check for check in report[0]["checks"]}
        assert checks["config_schema"]["status"] == "ok"
        assert checks["training_scale"]["details"]["train_examples"] == 512
        assert checks["evaluation_coverage"]["status"] == "ok"
        assert checks["cifar_archive"]["status"] == "ok"

        cfg["data"]["archive_md5"] = "0" * 32
        bad_cfg_path = tmp_path / "cifar10_bad_md5.yaml"
        bad_cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        bad_report = preflight_configs([str(bad_cfg_path)])
        assert bad_report[0]["status"] == "error"
        bad_checks = {check["id"]: check for check in bad_report[0]["checks"]}
        assert bad_checks["cifar_archive"]["status"] == "error"

        eval_cfg = load_config("configs/cifar10_table1_labels10.yaml")
        eval_cfg["data"]["archive_path"] = str(archive)
        eval_cfg["data"]["archive_md5"] = file_md5(archive)
        eval_cfg["eval_batches"] = 1
        eval_cfg_path = tmp_path / "cifar10_bad_eval.yaml"
        eval_cfg_path.write_text(yaml.safe_dump(eval_cfg), encoding="utf-8")
        eval_report = preflight_configs([str(eval_cfg_path)])
        assert eval_report[0]["status"] == "error"
        eval_checks = {check["id"]: check for check in eval_report[0]["checks"]}
        assert eval_checks["training_scale"]["details"]["classifier_steps_per_epoch"] == 2
        assert eval_checks["evaluation_coverage"]["status"] == "error"

        mnist_raw = tmp_path / "mnist_raw"
        resource_md5s = _write_fake_mnist_raw(mnist_raw)
        mnist_cfg = load_config("configs/mnist_table1_labels60.yaml")
        mnist_cfg["data"]["raw_data_dir"] = str(mnist_raw)
        mnist_cfg["data"]["resource_md5s"] = resource_md5s
        mnist_cfg_path = tmp_path / "mnist_table1_labels60.yaml"
        mnist_cfg_path.write_text(yaml.safe_dump(mnist_cfg), encoding="utf-8")
        mnist_report = preflight_configs([str(mnist_cfg_path)])
        assert mnist_report[0]["status"] == "ok"
        mnist_checks = {check["id"]: check for check in mnist_report[0]["checks"]}
        assert mnist_checks["mnist_resources"]["status"] == "ok"

        imagenet_cfg = load_config("configs/imagenet_stream_debug.yaml")
        imagenet_cfg_path = tmp_path / "imagenet_stream_debug.yaml"
        imagenet_cfg_path.write_text(yaml.safe_dump(imagenet_cfg), encoding="utf-8")
        imagenet_report = preflight_configs([str(imagenet_cfg_path)])
        imagenet_checks = {check["id"]: check for check in imagenet_report[0]["checks"]}
        imagenet_scale = imagenet_checks["training_scale"]["details"]
        assert imagenet_scale["classifier_steps_per_epoch"] == 2
        assert imagenet_scale["classifier_drop_last"] is True
        assert imagenet_scale["retrain_drop_last"] is True


def _assert_augnet_tau_override() -> None:
    augnet = CIFARAugmentationNetwork(tau_dim=128, tau_dropout=0.5)
    images = jnp.ones((3, 32, 32, 3), dtype=jnp.float32) * 0.5
    variables = augnet.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
        images,
        train=True,
        return_aux=True,
    )
    _, aux = augnet.apply(
        variables,
        images,
        train=False,
        return_aux=True,
    )
    zeros = jnp.zeros_like(aux["tau"])
    augmented, override_aux = augnet.apply(
        variables,
        images,
        train=False,
        return_aux=True,
        tau_override=zeros,
    )
    assert augmented.shape == images.shape
    assert override_aux["tau"].shape == (3, 128)
    assert override_aux["fields"].shape == (3, 32, 32, 18)
    assert bool(jnp.all(jnp.isfinite(override_aux["tau"])))

    bad_tau = zeros.at[0, 0].set(jnp.nan).at[1, 1].set(jnp.inf)
    augmented, override_aux = augnet.apply(
        variables,
        images,
        train=False,
        return_aux=True,
        tau_override=bad_tau,
    )
    assert bool(jnp.all(jnp.isfinite(augmented)))
    assert bool(jnp.all(jnp.isfinite(override_aux["tau"])))
    assert bool(jnp.all(jnp.isfinite(override_aux["fields"])))


def _assert_augnet_optimizer_sanitizes_nonfinite_grads() -> None:
    augnet = CIFARAugmentationNetwork(image_size=8, tau_dim=8, tau_dropout=0.0)
    state = create_augnet_state(
        jax.random.PRNGKey(0),
        augnet,
        input_shape=(1, 8, 8, 3),
        learning_rate=0.01,
        gradient_clip_norm=1.0,
        zero_nonfinite_grads=True,
    )
    grads = jax.tree_util.tree_map(lambda x: jnp.full_like(x, jnp.nan), state.params)
    updates, _ = state.tx.update(grads, state.opt_state, state.params)
    leaves = jax.tree_util.tree_leaves(updates)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    assert all(float(jnp.max(jnp.abs(leaf))) == 0.0 for leaf in leaves)


def _assert_checkpoint_opt_state_migration() -> None:
    augnet = CIFARAugmentationNetwork(image_size=8, tau_dim=8, tau_dropout=0.0)
    old_state = create_augnet_state(
        jax.random.PRNGKey(0),
        augnet,
        input_shape=(1, 8, 8, 3),
        learning_rate=0.01,
        gradient_clip_norm=0.0,
        zero_nonfinite_grads=False,
    )
    new_state = create_augnet_state(
        jax.random.PRNGKey(1),
        augnet,
        input_shape=(1, 8, 8, 3),
        learning_rate=0.01,
        gradient_clip_norm=1.0,
        zero_nonfinite_grads=True,
    )
    old_state = old_state.replace(step=old_state.step + 3)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "old.msgpack"
        save_state(str(path), old_state)
        restored = restore_state(str(path), new_state)
        params_only = restore_state(str(path), new_state, restore_opt_state=False)
    assert int(restored.step) == 3
    assert int(params_only.step) == 3
    old_flat, _ = ravel_pytree(old_state.params)
    restored_flat, _ = ravel_pytree(restored.params)
    params_only_flat, _ = ravel_pytree(params_only.params)
    assert bool(jnp.allclose(old_flat, restored_flat))
    assert bool(jnp.allclose(old_flat, params_only_flat))
    new_opt_flat, _ = ravel_pytree(new_state.opt_state)
    params_only_opt_flat, _ = ravel_pytree(params_only.opt_state)
    assert bool(jnp.allclose(new_opt_flat, params_only_opt_flat))


def _assert_augnet_field_smoothing() -> None:
    checker = jnp.asarray(
        [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]],
        dtype=jnp.float32,
    )[None, :, :, None]
    smoothed = average_pool_same(checker, kernel_size=4)
    assert smoothed.shape == checker.shape
    assert float(jnp.var(smoothed)) < float(jnp.var(checker))

    images = jnp.ones((1, 4, 4, 1), dtype=jnp.float32)
    spatial_params = jnp.tile(checker, (1, 1, 1, 6))
    _, raw_grid = apply_spatial_transform(
        images,
        spatial_params,
        spatial_scale=0.2,
        smoothing_kernel=1,
    )
    _, smooth_grid = apply_spatial_transform(
        images,
        spatial_params,
        spatial_scale=0.2,
        smoothing_kernel=4,
    )
    assert smooth_grid.shape == raw_grid.shape
    assert float(jnp.mean(jnp.abs(smooth_grid - raw_grid))) > 0.0


def _assert_grayscale_augnet_interface() -> None:
    augnet = CIFARAugmentationNetwork(
        image_size=28,
        channels=1,
        tau_dim=128,
        tau_dropout=0.5,
        use_appearance=False,
    )
    images = jnp.ones((3, 28, 28, 1), dtype=jnp.float32) * 0.5
    variables = augnet.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
        images,
        train=True,
        return_aux=True,
    )
    augmented, aux = augnet.apply(
        variables,
        images,
        train=False,
        return_aux=True,
    )
    assert augmented.shape == images.shape
    assert aux["tau"].shape == (3, 128)
    assert aux["fields"].shape == (3, 28, 28, 6)
    assert "appearance_delta" not in aux


def _assert_deep_augnet_interface() -> None:
    augnet = CIFARAugmentationNetwork(
        image_size=64,
        channels=3,
        tau_dim=128,
        tau_dropout=0.5,
        encoder_widths=(8, 16, 32, 32, 64, 64, 64, 64),
        decoder_base_width=64,
        decoder_widths=(64, 64, 64, 32, 32, 16, 16, 8),
    )
    images = jnp.ones((2, 64, 64, 3), dtype=jnp.float32) * 0.5
    variables = augnet.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
        images,
        train=True,
        return_aux=True,
    )
    augmented, aux = augnet.apply(
        variables,
        images,
        train=False,
        return_aux=True,
    )
    assert augmented.shape == images.shape
    assert aux["tau"].shape == (2, 128)
    assert aux["fields"].shape == (2, 64, 64, 18)
    assert "appearance_delta" in aux


def _assert_mnist_convnet_interface() -> None:
    model = MnistConvNet(num_classes=10)
    variables = model.init(
        jax.random.PRNGKey(0),
        jnp.ones((2, 28, 28, 1), dtype=jnp.float32),
        train=True,
    )
    (features, logits), updates = model.apply(
        variables,
        jnp.ones((2, 28, 28, 1), dtype=jnp.float32),
        train=True,
        return_features=True,
        mutable=["batch_stats"],
    )
    eval_features, eval_logits = model.apply(
        variables,
        jnp.ones((2, 28, 28, 1), dtype=jnp.float32),
        train=False,
        return_features=True,
    )
    assert features.shape == (2, 128)
    assert logits.shape == (2, 10)
    assert eval_features.shape == (2, 128)
    assert eval_logits.shape == (2, 10)
    assert "batch_stats" in updates
    assert variables["params"]["classifier"]["kernel"].shape == (128, 10)


def _assert_imagenet_resnet_interface() -> None:
    model = ImageNetResNet(
        stage_sizes=(1, 1, 1, 1),
        widths=(8, 16, 32, 64),
        stem_width=8,
        num_classes=1000,
    )
    variables = model.init(
        jax.random.PRNGKey(0),
        jnp.ones((1, 64, 64, 3), dtype=jnp.float32),
        train=True,
    )
    (features, logits), updates = model.apply(
        variables,
        jnp.ones((1, 64, 64, 3), dtype=jnp.float32),
        train=True,
        return_features=True,
        mutable=["batch_stats"],
    )
    eval_features, eval_logits = model.apply(
        variables,
        jnp.ones((1, 64, 64, 3), dtype=jnp.float32),
        train=False,
        return_features=True,
    )
    assert features.shape == (1, 256)
    assert logits.shape == (1, 1000)
    assert eval_features.shape == (1, 256)
    assert eval_logits.shape == (1, 1000)
    assert "batch_stats" in updates
    assert variables["params"]["classifier"]["kernel"].shape == (256, 1000)


def _assert_top5_eval_metric() -> None:
    model = ImageNetResNet(
        stage_sizes=(1, 1, 1, 1),
        widths=(4, 8, 16, 16),
        stem_width=4,
        num_classes=10,
    )
    state = create_classifier_state(
        jax.random.PRNGKey(0),
        model,
        input_shape=(1, 32, 32, 3),
        learning_rate=0.01,
    )
    metrics = classifier_eval_step(
        state,
        model,
        {
            "image": jnp.ones((2, 32, 32, 3), dtype=jnp.float32),
            "label": jnp.asarray([0, 1], dtype=jnp.int32),
        },
    )
    assert "top5_accuracy" in metrics
    assert 0.0 <= float(metrics["top5_accuracy"]) <= 1.0


def _assert_wide_resnet_interface() -> None:
    model = WideResNet(depth=10, width_multiplier=2, num_classes=10, dropout_rate=0.1)
    variables = model.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
        jnp.ones((1, 32, 32, 3), dtype=jnp.float32),
        train=True,
    )
    (features, logits), updates = model.apply(
        variables,
        jnp.ones((1, 32, 32, 3), dtype=jnp.float32),
        train=True,
        return_features=True,
        mutable=["batch_stats"],
        rngs={"dropout": jax.random.PRNGKey(2)},
    )
    assert features.shape == (1, 128)
    assert logits.shape == (1, 10)
    assert "batch_stats" in updates
    assert variables["params"]["classifier"]["kernel"].shape == (128, 10)


def _assert_preact_resnet18_interface() -> None:
    model = PreActResNet18(num_classes=10, width_multiplier=1)
    variables = model.init(
        jax.random.PRNGKey(0),
        jnp.ones((2, 32, 32, 3), dtype=jnp.float32),
        train=True,
    )
    (features, logits), updates = model.apply(
        variables,
        jnp.ones((2, 32, 32, 3), dtype=jnp.float32),
        train=True,
        return_features=True,
        mutable=["batch_stats"],
    )
    eval_features, eval_logits = model.apply(
        variables,
        jnp.ones((2, 32, 32, 3), dtype=jnp.float32),
        train=False,
        return_features=True,
    )
    assert features.shape == (2, 512)
    assert logits.shape == (2, 10)
    assert eval_features.shape == (2, 512)
    assert eval_logits.shape == (2, 10)
    assert "batch_stats" in updates
    assert variables["params"]["classifier"]["kernel"].shape == (512, 10)


def _assert_shake_shake_interface() -> None:
    model = ShakeShakeResNet(depth=8, base_width=8, num_classes=10)
    variables = model.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
        jnp.ones((2, 32, 32, 3), dtype=jnp.float32),
        train=True,
    )
    (features, logits), updates = model.apply(
        variables,
        jnp.ones((2, 32, 32, 3), dtype=jnp.float32),
        train=True,
        return_features=True,
        mutable=["batch_stats"],
        rngs={"dropout": jax.random.PRNGKey(2)},
    )
    eval_features, eval_logits = model.apply(
        variables,
        jnp.ones((2, 32, 32, 3), dtype=jnp.float32),
        train=False,
        return_features=True,
    )
    assert features.shape == (2, 32)
    assert logits.shape == (2, 10)
    assert eval_features.shape == (2, 32)
    assert eval_logits.shape == (2, 10)
    assert "batch_stats" in updates
    assert variables["params"]["classifier"]["kernel"].shape == (32, 10)


def _assert_pyramidnet_shakedrop_interface() -> None:
    model = PyramidNetShakeDrop(
        depth=20,
        alpha=12,
        num_classes=10,
        bottleneck=False,
        final_keep_prob=0.5,
    )
    variables = model.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
        jnp.ones((2, 32, 32, 3), dtype=jnp.float32),
        train=True,
    )
    (features, logits), updates = model.apply(
        variables,
        jnp.ones((2, 32, 32, 3), dtype=jnp.float32),
        train=True,
        return_features=True,
        mutable=["batch_stats"],
        rngs={"dropout": jax.random.PRNGKey(2)},
    )
    eval_features, eval_logits = model.apply(
        variables,
        jnp.ones((2, 32, 32, 3), dtype=jnp.float32),
        train=False,
        return_features=True,
    )
    assert features.shape == (2, 28)
    assert logits.shape == (2, 10)
    assert eval_features.shape == (2, 28)
    assert eval_logits.shape == (2, 10)
    assert "batch_stats" in updates
    assert variables["params"]["classifier"]["kernel"].shape == (28, 10)


def _assert_influence_shapes() -> None:
    rng = jax.random.PRNGKey(0)
    train_features = jax.random.normal(rng, (6, 4))
    val_features = jax.random.normal(rng, (5, 4))
    train_labels = jnp.asarray([0, 1, 2, 0, 1, 2])
    val_labels = jnp.asarray([2, 1, 0, 2, 1])
    classifier_params = {
        "kernel": jax.random.normal(rng, (4, 3)) * 0.01,
        "bias": jnp.zeros((3,), dtype=jnp.float32),
    }

    per_example_grads = last_layer_grad_per_example(
        train_features,
        train_labels,
        classifier_params,
    )
    assert per_example_grads["kernel"].shape == (6, 4, 3)
    assert per_example_grads["bias"].shape == (6, 3)

    s_test = compute_s_test(
        classifier_params,
        train_features,
        train_labels,
        val_features,
        val_labels,
        damping=1e-2,
        cg_iters=5,
    )
    assert s_test["kernel"].shape == classifier_params["kernel"].shape
    assert s_test["bias"].shape == classifier_params["bias"].shape

    influence = influence_up_loss(
        train_features,
        train_labels,
        classifier_params,
        s_test,
    )
    assert influence.shape == (6,)
    assert bool(jnp.all(jnp.isfinite(influence)))


def _tree_l2_delta(before, after) -> float:
    leaves = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda x, y: jnp.sum(jnp.square(y - x)), before, after)
    )
    return float(jnp.sqrt(sum(leaves)))


def _assert_last_layer_grad_values() -> None:
    features = jnp.asarray(
        [
            [1.0, 2.0],
            [-1.0, 0.5],
        ],
        dtype=jnp.float32,
    )
    labels = jnp.asarray([0, 1], dtype=jnp.int32)
    classifier_params = {
        "kernel": jnp.asarray(
            [
                [0.2, -0.1],
                [0.0, 0.3],
            ],
            dtype=jnp.float32,
        ),
        "bias": jnp.asarray([0.05, -0.02], dtype=jnp.float32),
    }
    logits = classifier_logits(features, classifier_params)
    residual = jax.nn.softmax(logits, axis=-1) - jax.nn.one_hot(labels, 2)
    expected_kernel = jnp.einsum("bd,bc->bdc", features, residual)
    grads = last_layer_grad_per_example(features, labels, classifier_params)
    np.testing.assert_allclose(np.asarray(grads["kernel"]), np.asarray(expected_kernel), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(grads["bias"]), np.asarray(residual), rtol=1e-6, atol=1e-6)


def _assert_classifier_to_augnet_tiny_pipeline() -> None:
    rng = jax.random.PRNGKey(123)
    images = jnp.linspace(0.0, 1.0, 4 * 32 * 32 * 3, dtype=jnp.float32).reshape(4, 32, 32, 3)
    labels = jnp.asarray([0, 1, 2, 3], dtype=jnp.int32)
    val_images = jnp.flip(images, axis=0)
    val_labels = jnp.asarray([3, 2, 1, 0], dtype=jnp.int32)
    train_batch = {"image": images, "label": labels}
    val_batch = {"image": val_images, "label": val_labels}

    classifier = PreActResNet(
        stage_sizes=(1, 1, 1, 1),
        widths=(8, 16, 32, 64),
        num_classes=4,
    )
    classifier_state = create_classifier_state(
        rng,
        classifier,
        input_shape=(1, 32, 32, 3),
        learning_rate=0.01,
        optimizer="sgd",
        momentum=0.9,
        weight_decay=0.0,
    )
    before_classifier_params = classifier_state.params
    classifier_state, train_metrics = classifier_train_step(
        classifier_state,
        classifier,
        train_batch,
        jax.random.PRNGKey(1),
        apply_baseline_augmentation=False,
        cutout_size=0,
    )
    assert _tree_l2_delta(before_classifier_params, classifier_state.params) > 0.0
    for value in train_metrics.values():
        assert bool(jnp.all(jnp.isfinite(value)))

    eval_metrics = classifier_eval_step(classifier_state, classifier, val_batch)
    for value in eval_metrics.values():
        assert bool(jnp.all(jnp.isfinite(value)))
    features, logits = extract_classifier_features(classifier_state, classifier, images)
    assert features.shape == (4, 64)
    assert logits.shape == (4, 4)
    assert classifier_state.params["classifier"]["kernel"].shape == (64, 4)

    s_test = compute_batch_s_test(
        classifier_state,
        classifier,
        train_batch,
        val_batch,
        damping=0.05,
        cg_iters=20,
    )
    residual = compute_batch_s_test_residual(
        classifier_state,
        classifier,
        train_batch,
        val_batch,
        s_test,
        damping=0.05,
    )
    assert float(residual) < 0.25

    augnet = CIFARAugmentationNetwork(
        image_size=32,
        channels=3,
        tau_dim=8,
        tau_dropout=0.0,
        spatial_scale=0.05,
        appearance_scale=0.05,
        encoder_widths=(4, 8),
        decoder_widths=(8,),
        decoder_base_width=8,
    )
    aug_state = create_augnet_state(
        jax.random.PRNGKey(2),
        augnet,
        input_shape=(1, 32, 32, 3),
        learning_rate=1e-3,
        gradient_clip_norm=1.0,
    )
    image_discriminator = ImageDiscriminator(widths=(4, 8))
    feature_discriminator = FeatureDiscriminator()
    image_d_state = create_discriminator_state(
        jax.random.PRNGKey(3),
        image_discriminator,
        input_shape=(1, 32, 32, 3),
        learning_rate=1e-3,
    )
    feature_d_state = create_discriminator_state(
        jax.random.PRNGKey(4),
        feature_discriminator,
        input_shape=(1, 64),
        learning_rate=1e-3,
    )

    before_pretrain_params = aug_state.params
    aug_state, image_d_state, feature_d_state, pretrain_metrics = augnet_pretrain_step(
        aug_state,
        augnet,
        image_d_state,
        image_discriminator,
        feature_d_state,
        feature_discriminator,
        classifier_state,
        classifier,
        train_batch,
        jax.random.PRNGKey(5),
        apply_baseline_augmentation=False,
        cutout_size=0,
        image_loss_weight=1.0,
        feature_loss_weight=1.0,
        identity_l2_weight=1e-3,
    )
    assert _tree_l2_delta(before_pretrain_params, aug_state.params) > 0.0
    for value in pretrain_metrics.values():
        assert bool(jnp.all(jnp.isfinite(value)))

    before_influence_params = aug_state.params
    aug_state, influence_metrics = augnet_influence_train_step(
        aug_state,
        augnet,
        classifier_state,
        classifier,
        train_batch,
        s_test,
        jax.random.PRNGKey(6),
        identity_l2_weight=1e-3,
        influence_clip_value=10.0,
        label_preservation_weight=0.1,
    )
    assert _tree_l2_delta(before_influence_params, aug_state.params) > 0.0
    np.testing.assert_allclose(
        float(influence_metrics["estimated_val_loss_reduction"]),
        -float(influence_metrics["i_aug_loss"]),
        rtol=1e-6,
        atol=1e-6,
    )
    assert 0.0 <= float(influence_metrics["accuracy_on_augmented"]) <= 1.0
    for value in influence_metrics.values():
        assert bool(jnp.all(jnp.isfinite(value)))

    retrain_state = create_classifier_state(
        jax.random.PRNGKey(7),
        classifier,
        input_shape=(1, 32, 32, 3),
        learning_rate=0.01,
        optimizer="sgd",
        momentum=0.9,
        weight_decay=0.0,
    )
    _, retrain_metrics_zero = classifier_train_step_with_augnet(
        retrain_state,
        classifier,
        aug_state,
        augnet,
        train_batch,
        jax.random.PRNGKey(8),
        apply_baseline_augmentation=False,
        cutout_size=0,
        learned_aug_probability=0.0,
        learned_aug_input="baseline",
    )
    _, retrain_metrics_one = classifier_train_step_with_augnet(
        retrain_state,
        classifier,
        aug_state,
        augnet,
        train_batch,
        jax.random.PRNGKey(9),
        apply_baseline_augmentation=False,
        cutout_size=0,
        learned_aug_probability=1.0,
        learned_aug_input="baseline",
    )
    _, retrain_metrics_raw = classifier_train_step_with_augnet(
        retrain_state,
        classifier,
        aug_state,
        augnet,
        train_batch,
        jax.random.PRNGKey(10),
        apply_baseline_augmentation=False,
        cutout_size=0,
        learned_aug_probability=1.0,
        learned_aug_input="raw",
    )
    assert float(retrain_metrics_zero["learned_aug_fraction"]) == 0.0
    assert float(retrain_metrics_one["learned_aug_fraction"]) == 1.0
    assert float(retrain_metrics_raw["learned_aug_fraction"]) == 1.0
    for metrics in (retrain_metrics_zero, retrain_metrics_one, retrain_metrics_raw):
        for value in metrics.values():
            assert bool(jnp.all(jnp.isfinite(value)))


def _is_tpu_backend() -> bool:
    """Return whether the current JAX process is running on a TPU backend."""
    return any(device.platform == "tpu" for device in jax.devices())


def _s_test_check_tolerances() -> tuple[float, float, float]:
    """Return residual, rtol, and atol tolerances for the dense-solve check."""
    if _is_tpu_backend():
        # TPU reductions/HVPs can be slightly noisier than CPU/GPU float32 here.
        return 5e-3, 5e-2, 2e-2
    return 1e-4, 1e-3, 1e-4


def _assert_s_test_matches_dense_solve() -> None:
    """Check CG iHVP against an explicit dense linear solve."""
    rng = jax.random.PRNGKey(7)
    train_features = jax.random.normal(rng, (5, 3))
    val_features = jax.random.normal(jax.random.PRNGKey(8), (4, 3))
    train_labels = jnp.asarray([0, 1, 2, 1, 0])
    val_labels = jnp.asarray([2, 1, 0, 2])
    classifier_params = {
        "kernel": jax.random.normal(jax.random.PRNGKey(9), (3, 3)) * 0.05,
        "bias": jnp.asarray([0.01, -0.02, 0.03], dtype=jnp.float32),
    }
    damping = 0.05

    s_test = compute_s_test(
        classifier_params,
        train_features,
        train_labels,
        val_features,
        val_labels,
        damping=damping,
        cg_iters=80,
    )
    residual = s_test_residual_norm(
        classifier_params,
        train_features,
        train_labels,
        val_features,
        val_labels,
        s_test,
        damping=damping,
    )
    residual_tolerance, dense_rtol, dense_atol = _s_test_check_tolerances()
    assert float(residual) < residual_tolerance, (
        f"s_test residual {float(residual):.6g} exceeds tolerance "
        f"{residual_tolerance:.6g} on backend {jax.default_backend()}"
    )

    engine_state = ClassifierTrainState.create(
        apply_fn=lambda *args, **kwargs: None,
        params={"classifier": classifier_params},
        tx=optax.sgd(0.0),
        batch_stats=None,
    )
    engine_residual = compute_batch_s_test_residual(
        engine_state,
        _FeatureIdentityModel(),
        {"image": train_features, "label": train_labels},
        {"image": val_features, "label": val_labels},
        s_test,
        damping=damping,
    )
    assert float(engine_residual) < residual_tolerance, (
        f"engine s_test residual {float(engine_residual):.6g} exceeds tolerance "
        f"{residual_tolerance:.6g} on backend {jax.default_backend()}"
    )

    flat_params, unravel = ravel_pytree(classifier_params)
    flat_s_test, _ = ravel_pytree(s_test)
    flat_val_grad, _ = ravel_pytree(
        classifier_grad(classifier_params, val_features, val_labels)
    )

    def loss_from_flat(flat_params_value):
        return classifier_loss(
            unravel(flat_params_value),
            train_features,
            train_labels,
        )

    hessian = jax.hessian(loss_from_flat)(flat_params)
    exact = jnp.linalg.solve(
        hessian + damping * jnp.eye(hessian.shape[0], dtype=hessian.dtype),
        flat_val_grad,
    )
    np.testing.assert_allclose(
        np.asarray(flat_s_test),
        np.asarray(exact),
        rtol=dense_rtol,
        atol=dense_atol,
    )


def main() -> None:
    checks = [
        ("cifar_direct_loaders", _assert_cifar_direct_loaders),
        ("balanced_train_subset", _assert_balanced_train_subset),
        ("synthetic_mnist_loader", _assert_synthetic_mnist_loader),
        ("mnist_direct_loader", _assert_mnist_direct_loader),
        ("synthetic_imagenet_loader", _assert_synthetic_imagenet_loader),
        ("imagenet_stream_info", _assert_imagenet_stream_info),
        ("imagenet_dry_run_no_side_effects", _assert_imagenet_dry_run_no_side_effects),
        ("progressive_schedule", _assert_progressive_schedule),
        ("cosine_lr_schedule", _assert_cosine_lr_schedule),
        ("training_epoch_coverage", _assert_training_epoch_coverage),
        ("weight_decay_mask", _assert_weight_decay_mask),
        ("iterator_resume", _assert_iterator_resume),
        ("evaluate_weights_partial_batches", _assert_evaluate_weights_partial_batches),
        ("config_validation", _assert_config_validation),
        ("all_configs_validate", _assert_all_configs_validate),
        ("collect_results", _assert_collect_results),
        ("retrain_probability_sweep_summary", _assert_retrain_probability_sweep_summary),
        ("compare_to_paper", _assert_compare_to_paper),
        ("paper_suite", _assert_paper_suite),
        ("suite_status", _assert_suite_status),
        ("preflight", _assert_preflight),
        ("augnet_tau_override", _assert_augnet_tau_override),
        ("augnet_optimizer_sanitizes_nonfinite_grads", _assert_augnet_optimizer_sanitizes_nonfinite_grads),
        ("checkpoint_opt_state_migration", _assert_checkpoint_opt_state_migration),
        ("augnet_field_smoothing", _assert_augnet_field_smoothing),
        ("grayscale_augnet_interface", _assert_grayscale_augnet_interface),
        ("deep_augnet_interface", _assert_deep_augnet_interface),
        ("mnist_convnet_interface", _assert_mnist_convnet_interface),
        ("imagenet_resnet_interface", _assert_imagenet_resnet_interface),
        ("top5_eval_metric", _assert_top5_eval_metric),
        ("wide_resnet_interface", _assert_wide_resnet_interface),
        ("preact_resnet18_interface", _assert_preact_resnet18_interface),
        ("shake_shake_interface", _assert_shake_shake_interface),
        ("pyramidnet_shakedrop_interface", _assert_pyramidnet_shakedrop_interface),
        ("influence_shapes", _assert_influence_shapes),
        ("last_layer_grad_values", _assert_last_layer_grad_values),
        ("classifier_to_augnet_tiny_pipeline", _assert_classifier_to_augnet_tiny_pipeline),
        ("s_test_matches_dense_solve", _assert_s_test_matches_dense_solve),
    ]
    for name, check in checks:
        check()
        print(f"{name}: ok")


if __name__ == "__main__":
    main()

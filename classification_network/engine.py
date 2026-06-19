from functools import partial
from typing import Any, Dict, Tuple

import flax
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from data.augmentations import cutout, random_crop_flip


class ClassifierTrainState(train_state.TrainState):
    """TrainState for classifier F, including mutable BatchNorm statistics."""

    batch_stats: Any = flax.struct.field(pytree_node=True, default=None)


def _kernel_decay_mask(params: Any) -> Any:
    """Return a PyTree mask that applies weight decay only to kernel tensors."""
    return flax.traverse_util.path_aware_map(
        lambda path, _: path[-1] == "kernel",
        params,
    )


def _make_optimizer(
    optimizer: str,
    learning_rate: Any,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
) -> optax.GradientTransformation:
    """Build the optimizer used by classifier and related train states."""
    if optimizer == "sgd":
        tx = optax.sgd(learning_rate, momentum=momentum, nesterov=True)
    elif optimizer == "adam":
        tx = optax.adam(learning_rate, b1=adam_beta1, b2=adam_beta2)
    elif optimizer == "adamw":
        # AdamW owns weight decay internally, so do not add it again below.
        tx = optax.adamw(
            learning_rate,
            b1=adam_beta1,
            b2=adam_beta2,
            weight_decay=weight_decay,
            mask=_kernel_decay_mask,
        )
        weight_decay = 0.0
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    if weight_decay > 0:
        tx = optax.chain(optax.add_decayed_weights(weight_decay, mask=_kernel_decay_mask), tx)
    return tx


def _sanitize_batch_stats(batch_stats: Any, max_abs: float = 1.0e4) -> Any:
    """Clamp BatchNorm statistics so one bad batch cannot poison evaluation."""
    if batch_stats is None:
        return None

    def clean(path, value):
        """Sanitize a single BatchNorm statistic leaf."""
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            return value
        if path and path[-1] == "var":
            # Variances must stay positive for BatchNorm normalization.
            value = jnp.nan_to_num(value, nan=1.0, posinf=max_abs, neginf=1.0)
            return jnp.clip(value, 1.0e-6, max_abs)
        # Means can be signed, but extreme values make later eval unstable.
        value = jnp.nan_to_num(value, nan=0.0, posinf=max_abs, neginf=-max_abs)
        return jnp.clip(value, -max_abs, max_abs)

    return flax.traverse_util.path_aware_map(clean, batch_stats)


def create_classifier_state(
    rng: jax.Array,
    model: Any,
    input_shape: Tuple[int, int, int, int] = (1, 32, 32, 3),
    learning_rate: Any = 0.1,
    optimizer: str = "sgd",
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    gradient_clip_norm: float = 0.0,
    zero_nonfinite_grads: bool = False,
) -> ClassifierTrainState:
    """Initialize classifier F and attach its optimizer state."""
    params_rng, dropout_rng = jax.random.split(rng)
    # Flax init needs all mutable collections that may be used at train time.
    variables = model.init(
        {"params": params_rng, "dropout": dropout_rng},
        jnp.ones(input_shape, jnp.float32),
        train=True,
    )
    transforms = []
    if zero_nonfinite_grads:
        # Drop non-finite gradient entries before clipping or optimizer updates.
        transforms.append(optax.zero_nans())
    if gradient_clip_norm and gradient_clip_norm > 0:
        transforms.append(optax.clip_by_global_norm(gradient_clip_norm))
    transforms.append(
        _make_optimizer(
            optimizer,
            learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    )
    tx = optax.chain(*transforms)
    return ClassifierTrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        batch_stats=_sanitize_batch_stats(variables.get("batch_stats")),
    )


def accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Compute top-1 accuracy for integer labels."""
    return jnp.mean(jnp.argmax(logits, axis=-1) == labels.astype(jnp.int32))


def top_k_accuracy(logits: jnp.ndarray, labels: jnp.ndarray, k: int = 5) -> jnp.ndarray:
    """Compute top-k accuracy, clipping k to the number of classes."""
    _, top_indices = jax.lax.top_k(logits, min(k, logits.shape[-1]))
    matches = top_indices == labels.astype(jnp.int32)[:, None]
    return jnp.mean(jnp.any(matches, axis=-1))


def normalize_images(
    images: jnp.ndarray,
    image_mean: Any = None,
    image_std: Any = None,
) -> jnp.ndarray:
    """Normalize images when dataset-level mean/std are configured."""
    if image_mean is None or image_std is None:
        return images
    return (images - image_mean) / image_std


def _apply_baseline_augmentation(
    images: jnp.ndarray,
    rng: jax.Array,
    mode: Any,
    cutout_size: int,
) -> jnp.ndarray:
    """Apply the configured hand-written baseline augmentation."""
    if mode is False or mode == "none":
        return images

    rng_crop, rng_cutout = jax.random.split(rng)
    if mode is True or mode == "cifar":
        # CIFAR baseline follows the standard random crop plus horizontal flip.
        images = random_crop_flip(images, rng_crop)
    elif mode == "crop":
        # MNIST uses random crop without horizontal flipping.
        images = random_crop_flip(images, rng_crop, flip_probability=0.0)
    else:
        raise ValueError(f"Unknown baseline augmentation mode: {mode}")

    if cutout_size > 0:
        # Cutout is optional and only applied after geometric augmentation.
        images = cutout(images, rng_cutout, size=cutout_size)
    return images


@partial(
    jax.jit,
    static_argnames=("model", "apply_baseline_augmentation", "cutout_size"),
)
def classifier_train_step(
    state: ClassifierTrainState,
    model: Any,
    batch: Dict[str, jnp.ndarray],
    rng: jax.Array,
    apply_baseline_augmentation: bool = True,
    cutout_size: int = 0,
    image_mean: Any = None,
    image_std: Any = None,
) -> Tuple[ClassifierTrainState, Dict[str, jnp.ndarray]]:
    """Run one supervised classifier update on a minibatch."""
    images = batch["image"]
    labels = batch["label"].astype(jnp.int32)
    rng_aug, rng_dropout = jax.random.split(rng)

    # The classifier baseline path sees only hand-written augmentation.
    images = _apply_baseline_augmentation(
        images,
        rng_aug,
        apply_baseline_augmentation,
        cutout_size,
    )

    def loss_fn(params):
        """Compute classifier loss and collect updated BatchNorm statistics."""
        variables = {"params": params, "batch_stats": state.batch_stats}
        logits, updates = model.apply(
            variables,
            normalize_images(images, image_mean, image_std),
            train=True,
            mutable=["batch_stats"],
            rngs={"dropout": rng_dropout},
        )
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        loss = jnp.mean(losses)
        return loss, (logits, updates["batch_stats"])

    # value_and_grad keeps logits and BatchNorm updates alongside the gradient.
    (loss, (logits, batch_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads, batch_stats=_sanitize_batch_stats(batch_stats))
    metrics = {
        "loss": loss,
        "accuracy": accuracy(logits, labels),
        "top5_accuracy": top_k_accuracy(logits, labels),
    }
    return state, metrics


@partial(
    jax.jit,
    static_argnames=(
        "classifier_model",
        "aug_model",
        "apply_baseline_augmentation",
        "cutout_size",
        "learned_aug_input",
    ),
)
def classifier_train_step_with_augnet(
    state: ClassifierTrainState,
    classifier_model: Any,
    aug_state: Any,
    aug_model: Any,
    batch: Dict[str, jnp.ndarray],
    rng: jax.Array,
    apply_baseline_augmentation: bool = True,
    cutout_size: int = 0,
    learned_aug_probability: float = 1.0,
    learned_aug_input: str = "baseline",
    image_mean: Any = None,
    image_std: Any = None,
) -> Tuple[ClassifierTrainState, Dict[str, jnp.ndarray]]:
    """Run one classifier retraining step with optional learned augmentation."""
    images = batch["image"]
    labels = batch["label"].astype(jnp.int32)
    rng_baseline, rng_aug, rng_mix, rng_dropout = jax.random.split(rng, 4)

    # Baseline images are always available as the fallback training samples.
    baseline_images = _apply_baseline_augmentation(
        images,
        rng_baseline,
        apply_baseline_augmentation,
        cutout_size,
    )

    if learned_aug_input == "raw":
        # Feed original images into G when testing the pure learned-transform path.
        augnet_inputs = images
    elif learned_aug_input == "baseline":
        # Feed baseline-augmented images into G for the stable mixed path.
        augnet_inputs = baseline_images
    else:
        raise ValueError(f"Unknown learned_aug_input: {learned_aug_input}")

    # Keep dropout enabled in G to sample diverse learned augmentations.
    augmented_images = aug_model.apply(
        {"params": aug_state.params},
        augnet_inputs,
        train=True,
        rngs={"dropout": rng_aug},
    )
    aug_probability = jnp.clip(
        jnp.asarray(learned_aug_probability, dtype=baseline_images.dtype),
        0.0,
        1.0,
    )
    mask_shape = (baseline_images.shape[0],) + (1,) * (baseline_images.ndim - 1)
    # Mix examples independently instead of switching the whole batch at once.
    use_augmented = jax.random.bernoulli(rng_mix, p=aug_probability, shape=mask_shape)
    images = jnp.where(use_augmented, augmented_images, baseline_images)

    def loss_fn(params):
        """Compute retraining loss on the mixed baseline/learned batch."""
        variables = {"params": params, "batch_stats": state.batch_stats}
        logits, updates = classifier_model.apply(
            variables,
            normalize_images(images, image_mean, image_std),
            train=True,
            mutable=["batch_stats"],
            rngs={"dropout": rng_dropout},
        )
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        loss = jnp.mean(losses)
        return loss, (logits, updates["batch_stats"])

    (loss, (logits, batch_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads, batch_stats=_sanitize_batch_stats(batch_stats))
    metrics = {
        "loss": loss,
        "accuracy": accuracy(logits, labels),
        "top5_accuracy": top_k_accuracy(logits, labels),
        "learned_aug_fraction": jnp.mean(use_augmented.astype(jnp.float32)),
        "learned_aug_probability": aug_probability,
    }
    return state, metrics


@partial(jax.jit, static_argnames=("model",))
def classifier_eval_step(
    state: ClassifierTrainState,
    model: Any,
    batch: Dict[str, jnp.ndarray],
    image_mean: Any = None,
    image_std: Any = None,
) -> Dict[str, jnp.ndarray]:
    """Evaluate classifier F without mutating BatchNorm statistics."""
    labels = batch["label"].astype(jnp.int32)
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    images = normalize_images(batch["image"], image_mean, image_std)
    logits = model.apply(variables, images, train=False)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
    return {
        "loss": loss,
        "accuracy": accuracy(logits, labels),
        "top5_accuracy": top_k_accuracy(logits, labels),
    }


@partial(jax.jit, static_argnames=("model",))
def extract_classifier_features(
    state: ClassifierTrainState,
    model: Any,
    images: jnp.ndarray,
    image_mean: Any = None,
    image_std: Any = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return final-layer features and logits for influence estimation."""
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    images = normalize_images(images, image_mean, image_std)
    return model.apply(variables, images, train=False, return_features=True)

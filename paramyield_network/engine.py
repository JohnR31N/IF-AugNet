from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import optax

from classification_network.engine import (
    ClassifierTrainState,
    accuracy,
    extract_classifier_features,
    normalize_images,
)
from paramyield_network.influence import compute_s_test, influence_up_loss, s_test_residual_norm


@partial(jax.jit, static_argnames=("classifier_model", "cg_iters"))
def compute_batch_s_test(
    classifier_state: ClassifierTrainState,
    classifier_model: Any,
    train_batch: Dict[str, jnp.ndarray],
    val_batch: Dict[str, jnp.ndarray],
    damping: float = 1e-2,
    cg_iters: int = 50,
    image_mean: Any = None,
    image_std: Any = None,
) -> Dict[str, jnp.ndarray]:
    """Compute a minibatch iHVP vector for the final classifier layer."""
    # Freeze classifier F and extract the feature vectors used by the top layer.
    train_features, _ = extract_classifier_features(
        classifier_state,
        classifier_model,
        train_batch["image"],
        image_mean=image_mean,
        image_std=image_std,
    )
    val_features, _ = extract_classifier_features(
        classifier_state,
        classifier_model,
        val_batch["image"],
        image_mean=image_mean,
        image_std=image_std,
    )
    return compute_s_test(
        classifier_state.params["classifier"],
        train_features,
        train_batch["label"],
        val_features,
        val_batch["label"],
        damping=damping,
        cg_iters=cg_iters,
    )


@partial(jax.jit, static_argnames=("classifier_model",))
def compute_batch_s_test_residual(
    classifier_state: ClassifierTrainState,
    classifier_model: Any,
    train_batch: Dict[str, jnp.ndarray],
    val_batch: Dict[str, jnp.ndarray],
    s_test: Dict[str, jnp.ndarray],
    damping: float = 1e-2,
    image_mean: Any = None,
    image_std: Any = None,
) -> jnp.ndarray:
    """Compute the relative residual for a minibatch s_test estimate."""
    train_features, _ = extract_classifier_features(
        classifier_state,
        classifier_model,
        train_batch["image"],
        image_mean=image_mean,
        image_std=image_std,
    )
    val_features, _ = extract_classifier_features(
        classifier_state,
        classifier_model,
        val_batch["image"],
        image_mean=image_mean,
        image_std=image_std,
    )
    return s_test_residual_norm(
        classifier_state.params["classifier"],
        train_features,
        train_batch["label"],
        val_features,
        val_batch["label"],
        s_test,
        damping=damping,
    )


@partial(
    jax.jit,
    static_argnames=("aug_model", "classifier_model"),
)
def augnet_influence_train_step(
    aug_state: Any,
    aug_model: Any,
    classifier_state: ClassifierTrainState,
    classifier_model: Any,
    batch: Dict[str, jnp.ndarray],
    s_test: Dict[str, jnp.ndarray],
    rng: jax.Array,
    identity_l2_weight: float = 0.0,
    influence_clip_value: float = 0.0,
    label_preservation_weight: float = 0.0,
    image_mean: Any = None,
    image_std: Any = None,
) -> Tuple[Any, Dict[str, jnp.ndarray]]:
    """Update E/G by optimizing the influence of learned augmentations."""
    images = batch["image"]
    labels = batch["label"].astype(jnp.int32)

    def loss_fn(params):
        """Compute influence loss and regularizers for the augmentation model."""
        # G(E(x)) produces differentiable augmented images and aux diagnostics.
        augmented, aux = aug_model.apply(
            {"params": params},
            images,
            train=True,
            return_aux=True,
            rngs={"dropout": rng},
        )
        # Classifier F stays frozen; gradients flow through F into augmented.
        features, logits = classifier_model.apply(
            {"params": classifier_state.params, "batch_stats": classifier_state.batch_stats},
            normalize_images(augmented, image_mean, image_std),
            train=False,
            return_features=True,
        )
        # Original influence is subtracted to optimize replacement benefit.
        original_features, _ = classifier_model.apply(
            {"params": classifier_state.params, "batch_stats": classifier_state.batch_stats},
            normalize_images(images, image_mean, image_std),
            train=False,
            return_features=True,
        )
        augmented_influence = influence_up_loss(
            features,
            labels,
            classifier_state.params["classifier"],
            s_test,
        )
        original_influence = influence_up_loss(
            original_features,
            labels,
            classifier_state.params["classifier"],
            s_test,
        )
        influence = augmented_influence - original_influence
        # Optional clipping prevents a few extreme samples from dominating G.
        clip_value = jnp.asarray(influence_clip_value, dtype=influence.dtype)
        clipped_influence = jnp.clip(influence, -clip_value, clip_value)
        clipped_influence = jnp.where(clip_value > 0.0, clipped_influence, influence)
        raw_influence_loss = jnp.mean(influence)
        influence_loss = jnp.mean(clipped_influence)
        # Label preservation discourages class-destroying augmentations.
        label_preservation_loss = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        )
        # A small identity penalty keeps spatial fields close to the input.
        identity_l2 = jnp.mean(jnp.square(augmented - images))
        loss = (
            influence_loss
            + identity_l2_weight * identity_l2
            + label_preservation_weight * label_preservation_loss
        )
        metrics = {
            "loss": loss,
            "i_aug_loss": influence_loss,
            "raw_i_aug_loss": raw_influence_loss,
            "label_preservation_loss": label_preservation_loss,
            "augmented_influence": jnp.mean(augmented_influence),
            "original_influence": jnp.mean(original_influence),
            "estimated_val_loss_reduction": -influence_loss,
            "identity_l2": identity_l2,
            "accuracy_on_augmented": accuracy(logits, labels),
            "tau_abs_mean": jnp.mean(jnp.abs(aux["tau"])),
        }
        return loss, metrics

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(aug_state.params)
    aug_state = aug_state.apply_gradients(grads=grads)
    return aug_state, metrics

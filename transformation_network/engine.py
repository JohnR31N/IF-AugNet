from functools import partial
from typing import Any, Dict, Tuple

import flax
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from classification_network.engine import (
    ClassifierTrainState,
    _apply_baseline_augmentation,
    _make_optimizer,
    normalize_images,
)


class AugmentTrainState(train_state.TrainState):
    """TrainState for the coupled E/G augmentation model."""

    pass


class DiscriminatorTrainState(train_state.TrainState):
    """TrainState for image-space or feature-space discriminator."""

    pass


def create_augnet_state(
    rng: jax.Array,
    model: Any,
    input_shape: Tuple[int, int, int, int] = (1, 32, 32, 3),
    learning_rate: Any = 1e-2,
    optimizer: str = "adam",
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    gradient_clip_norm: float = 1.0,
    zero_nonfinite_grads: bool = True,
) -> AugmentTrainState:
    """Initialize E/G parameters and optimizer state."""
    init_rng, dropout_rng = jax.random.split(rng)
    # Use train=True so dropout-related collections are initialized consistently.
    variables = model.init(
        {"params": init_rng, "dropout": dropout_rng},
        jnp.ones(input_shape, jnp.float32),
        train=True,
    )
    transforms = []
    if zero_nonfinite_grads:
        # Influence gradients can spike; sanitize them before clipping.
        transforms.append(optax.zero_nans())
    if gradient_clip_norm and gradient_clip_norm > 0:
        transforms.append(optax.clip_by_global_norm(gradient_clip_norm))
    transforms.append(
        _make_optimizer(
            optimizer,
            learning_rate,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
        )
    )
    tx = optax.chain(*transforms)
    return AugmentTrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
    )


def create_discriminator_state(
    rng: jax.Array,
    model: Any,
    input_shape: Tuple[int, ...],
    learning_rate: float = 2e-4,
    beta1: float = 0.5,
    beta2: float = 0.999,
) -> DiscriminatorTrainState:
    """Initialize a discriminator and its Adam optimizer."""
    variables = model.init(rng, jnp.ones(input_shape, jnp.float32))
    tx = optax.adam(learning_rate, b1=beta1, b2=beta2)
    return DiscriminatorTrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
    )


def _ragan_discriminator_loss(real_logits: jnp.ndarray, fake_logits: jnp.ndarray) -> jnp.ndarray:
    """Relativistic average GAN loss for discriminator updates."""
    real_rel = real_logits - jnp.mean(fake_logits)
    fake_rel = fake_logits - jnp.mean(real_logits)
    return jnp.mean(jax.nn.softplus(-real_rel)) + jnp.mean(jax.nn.softplus(fake_rel))


def _ragan_generator_loss(real_logits: jnp.ndarray, fake_logits: jnp.ndarray) -> jnp.ndarray:
    """Relativistic average GAN loss for generator updates."""
    real_rel = real_logits - jnp.mean(fake_logits)
    fake_rel = fake_logits - jnp.mean(real_logits)
    return jnp.mean(jax.nn.softplus(real_rel)) + jnp.mean(jax.nn.softplus(-fake_rel))


def _apply_progressive_resolution(images: jnp.ndarray, image_size: Any = None) -> jnp.ndarray:
    """Downsample then upsample images for progressive GAN pretraining."""
    if image_size is None or image_size <= 0 or image_size >= images.shape[1]:
        return images
    low_res = jax.image.resize(
        images,
        (images.shape[0], image_size, image_size, images.shape[-1]),
        method="linear",
    )
    return jax.image.resize(
        low_res,
        images.shape,
        method="linear",
    )


@partial(
    jax.jit,
    static_argnames=(
        "aug_model",
        "image_discriminator",
        "feature_discriminator",
        "classifier_model",
        "apply_baseline_augmentation",
        "cutout_size",
        "progressive_image_size",
    ),
)
def augnet_pretrain_step(
    aug_state: AugmentTrainState,
    aug_model: Any,
    image_discriminator_state: DiscriminatorTrainState,
    image_discriminator: Any,
    feature_discriminator_state: DiscriminatorTrainState,
    feature_discriminator: Any,
    classifier_state: ClassifierTrainState,
    classifier_model: Any,
    batch: Dict[str, jnp.ndarray],
    rng: jax.Array,
    apply_baseline_augmentation: bool = True,
    cutout_size: int = 16,
    progressive_image_size: int | None = None,
    image_loss_weight: float = 1.0,
    feature_loss_weight: float = 1.0,
    identity_l2_weight: float = 0.0,
    image_mean: Any = None,
    image_std: Any = None,
) -> Tuple[
    AugmentTrainState,
    DiscriminatorTrainState,
    DiscriminatorTrainState,
    Dict[str, jnp.ndarray],
]:
    """Run one RaGAN-style pretraining step for G and its discriminators."""
    images = batch["image"]
    rng_baseline, rng_aug_d, rng_aug_g = jax.random.split(rng, 3)
    # Real samples are hand-augmented images from the baseline policy.
    real_images = _apply_baseline_augmentation(
        images,
        rng_baseline,
        apply_baseline_augmentation,
        cutout_size,
    )
    real_images_view = _apply_progressive_resolution(real_images, progressive_image_size)
    images_view = _apply_progressive_resolution(images, progressive_image_size)

    classifier_variables = {
        "params": classifier_state.params,
        "batch_stats": classifier_state.batch_stats,
    }

    # Fake images for discriminator training are detached from G.
    fake_images = aug_model.apply(
        {"params": aug_state.params},
        images,
        train=True,
        rngs={"dropout": rng_aug_d},
    )
    fake_images_for_d = jax.lax.stop_gradient(
        _apply_progressive_resolution(fake_images, progressive_image_size)
    )
    # Feature discriminator sees frozen classifier features.
    real_features, _ = classifier_model.apply(
        classifier_variables,
        normalize_images(real_images_view, image_mean, image_std),
        train=False,
        return_features=True,
    )
    fake_features_for_d, _ = classifier_model.apply(
        classifier_variables,
        normalize_images(fake_images_for_d, image_mean, image_std),
        train=False,
        return_features=True,
    )
    real_features_for_d = jax.lax.stop_gradient(real_features)
    fake_features_for_d = jax.lax.stop_gradient(fake_features_for_d)

    def discriminator_loss_fn(image_params, feature_params):
        """Train discriminators to separate real baseline samples from G samples."""
        image_real_logits = image_discriminator.apply({"params": image_params}, real_images_view)
        image_fake_logits = image_discriminator.apply({"params": image_params}, fake_images_for_d)
        feature_real_logits = feature_discriminator.apply({"params": feature_params}, real_features_for_d)
        feature_fake_logits = feature_discriminator.apply({"params": feature_params}, fake_features_for_d)

        image_loss = _ragan_discriminator_loss(image_real_logits, image_fake_logits)
        feature_loss = _ragan_discriminator_loss(feature_real_logits, feature_fake_logits)
        loss = image_loss_weight * image_loss + feature_loss_weight * feature_loss
        metrics = {
            "d_loss": loss,
            "d_image_loss": image_loss,
            "d_feature_loss": feature_loss,
            "d_real_logit": jnp.mean(image_real_logits),
            "d_fake_logit": jnp.mean(image_fake_logits),
        }
        return loss, metrics

    # Update image and feature discriminators together.
    (_, d_metrics), (image_grads, feature_grads) = jax.value_and_grad(
        discriminator_loss_fn,
        argnums=(0, 1),
        has_aux=True,
    )(image_discriminator_state.params, feature_discriminator_state.params)
    image_discriminator_state = image_discriminator_state.apply_gradients(grads=image_grads)
    feature_discriminator_state = feature_discriminator_state.apply_gradients(grads=feature_grads)

    def generator_loss_fn(aug_params):
        """Train G to fool both discriminators while staying near identity."""
        # Recompute generated samples with live G params so gradients reach G.
        generated, aux = aug_model.apply(
            {"params": aug_params},
            images,
            train=True,
            return_aux=True,
            rngs={"dropout": rng_aug_g},
        )
        generated_view = _apply_progressive_resolution(generated, progressive_image_size)
        generated_features, _ = classifier_model.apply(
            classifier_variables,
            normalize_images(generated_view, image_mean, image_std),
            train=False,
            return_features=True,
        )
        image_real_logits = image_discriminator.apply(
            {"params": image_discriminator_state.params},
            jax.lax.stop_gradient(real_images_view),
        )
        # Fake logits are not detached here; they provide gradients for G.
        image_fake_logits = image_discriminator.apply(
            {"params": image_discriminator_state.params},
            generated_view,
        )
        feature_real_logits = feature_discriminator.apply(
            {"params": feature_discriminator_state.params},
            jax.lax.stop_gradient(real_features),
        )
        feature_fake_logits = feature_discriminator.apply(
            {"params": feature_discriminator_state.params},
            generated_features,
        )

        image_loss = _ragan_generator_loss(image_real_logits, image_fake_logits)
        feature_loss = _ragan_generator_loss(feature_real_logits, feature_fake_logits)
        # Identity penalty keeps pretraining from drifting too far from inputs.
        identity_l2 = jnp.mean(jnp.square(generated_view - images_view))
        loss = (
            image_loss_weight * image_loss
            + feature_loss_weight * feature_loss
            + identity_l2_weight * identity_l2
        )
        metrics = {
            "g_loss": loss,
            "g_image_loss": image_loss,
            "g_feature_loss": feature_loss,
            "pretrain_identity_l2": identity_l2,
            "pretrain_tau_abs_mean": jnp.mean(jnp.abs(aux["tau"])),
            "pretrain_progressive_image_size": jnp.asarray(
                progressive_image_size or images.shape[1],
                dtype=jnp.float32,
            ),
        }
        return loss, metrics

    # Update only E/G in the generator pass.
    (_, g_metrics), aug_grads = jax.value_and_grad(generator_loss_fn, has_aux=True)(aug_state.params)
    aug_state = aug_state.apply_gradients(grads=aug_grads)
    metrics = {**d_metrics, **g_metrics}
    return aug_state, image_discriminator_state, feature_discriminator_state, metrics

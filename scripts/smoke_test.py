from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp

from classification_network import (
    ResNet18,
    classifier_train_step,
    classifier_train_step_with_augnet,
    create_classifier_state,
)
from paramyield_network import (
    augnet_influence_train_step,
    compute_batch_s_test,
)
from transformation_network import (
    CIFARAugmentationNetwork,
    FeatureDiscriminator,
    ImageDiscriminator,
    augnet_pretrain_step,
    create_augnet_state,
    create_discriminator_state,
)


def main() -> None:
    rng = jax.random.PRNGKey(0)
    (
        rng_model,
        rng_aug,
        rng_image_d,
        rng_feature_d,
        rng_data,
        rng_train,
        rng_pretrain,
        rng_aug_step,
        rng_retrain,
    ) = jax.random.split(rng, 9)

    classifier = ResNet18(num_classes=10, width_multiplier=1)
    augnet = CIFARAugmentationNetwork(tau_dropout=0.5)
    image_discriminator = ImageDiscriminator()
    feature_discriminator = FeatureDiscriminator()

    classifier_state = create_classifier_state(
        rng_model,
        classifier,
        input_shape=(1, 32, 32, 3),
        learning_rate=0.01,
        optimizer="sgd",
    )
    aug_state = create_augnet_state(
        rng_aug,
        augnet,
        input_shape=(1, 32, 32, 3),
        learning_rate=1e-3,
    )
    image_discriminator_state = create_discriminator_state(
        rng_image_d,
        image_discriminator,
        input_shape=(1, 32, 32, 3),
        learning_rate=1e-4,
    )
    feature_discriminator_state = create_discriminator_state(
        rng_feature_d,
        feature_discriminator,
        input_shape=(1, 512),
        learning_rate=1e-4,
    )

    images = jax.random.uniform(rng_data, (8, 32, 32, 3), dtype=jnp.float32)
    labels = jax.random.randint(rng_data, (8,), 0, 10)
    batch = {"image": images, "label": labels}
    val_batch = {"image": images[::-1], "label": labels[::-1]}

    classifier_state, classifier_metrics = classifier_train_step(
        classifier_state,
        classifier,
        batch,
        rng_train,
        apply_baseline_augmentation=False,
    )
    (
        aug_state,
        image_discriminator_state,
        feature_discriminator_state,
        pretrain_metrics,
    ) = augnet_pretrain_step(
        aug_state,
        augnet,
        image_discriminator_state,
        image_discriminator,
        feature_discriminator_state,
        feature_discriminator,
        classifier_state,
        classifier,
        batch,
        rng_pretrain,
        cutout_size=4,
        identity_l2_weight=0.01,
    )
    s_test = compute_batch_s_test(
        classifier_state,
        classifier,
        batch,
        val_batch,
        damping=1e-2,
        cg_iters=5,
    )
    aug_state, aug_metrics = augnet_influence_train_step(
        aug_state,
        augnet,
        classifier_state,
        classifier,
        batch,
        s_test,
        rng_aug_step,
        identity_l2_weight=0.01,
    )
    classifier_state, retrain_metrics = classifier_train_step_with_augnet(
        classifier_state,
        classifier,
        aug_state,
        augnet,
        batch,
        rng_retrain,
        apply_baseline_augmentation=False,
    )

    print("classifier_loss", float(classifier_metrics["loss"]))
    print("classifier_accuracy", float(classifier_metrics["accuracy"]))
    print("pretrain_g_loss", float(pretrain_metrics["g_loss"]))
    print("augnet_loss", float(aug_metrics["loss"]))
    print("estimated_val_loss_reduction", float(aug_metrics["estimated_val_loss_reduction"]))
    print("retrain_loss", float(retrain_metrics["loss"]))


if __name__ == "__main__":
    main()

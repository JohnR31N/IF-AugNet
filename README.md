# IF-AugNet

A JAX/Flax implementation scaffold for
[Learning Augmentation Network via Influence Functions (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Lee_Learning_Augmentation_Network_via_Influence_Functions_CVPR_2020_paper.html).

The current codebase maps the paper roles to small, direct packages:

- `classification_network/`: classifier `F`, Flax backbones, classifier
  TrainState, standard training/eval, and final retraining with learned
  augmentation.
- `paramyield_network/`: parameter-yield encoder `E`, last-layer influence
  estimation, conjugate-gradient iHVPs, and the influence objective used to
  train `E/G`.
- `transformation_network/`: transformation model `G`, spatial/appearance
  transforms, discriminators, and RaGAN-style pretraining.
- `data/`: dataset loading and baseline augmentations.
- `scripts/`: smoke tests, training entrypoints, paper-suite helpers, and
  result collection.

## Setup

Use Python 3.10/3.11. The repo includes a Windows-friendly dependency pin for
`orbax-checkpoint`; newer Orbax wheels can exceed Windows path limits when the
virtual environment lives inside a deep repository path.

```powershell
.\.ifaugnet\Scripts\python.exe -m pip install -r requirements.txt
```

## Smoke Test

Run the lightweight regression checks first. They do not require pytest or a
network connection, and cover CIFAR archive parsing, iterator resume state,
config validation, and final-layer influence tensor shapes:

```powershell
.\.ifaugnet\Scripts\python.exe scripts\run_checks.py
```

Then verify the full differentiable chain without loading a dataset:

1. one ResNet classifier update,
2. one RaGAN-style AugNet pretraining update,
3. last-layer `s_test = H^-1 grad L_val` estimation,
4. one AugNet influence update,
5. one classifier retraining update using learned AugNet.

```powershell
.\.ifaugnet\Scripts\python.exe scripts\smoke_test.py
```

Expected output includes `classifier_loss`, `augnet_loss`, and
`estimated_val_loss_reduction`.

To audit the implementation against the paper-level CIFAR reproduction path,
including explicit uncovered benchmark families, run:

```powershell
.\.ifaugnet\Scripts\python.exe scripts\audit_paper_alignment.py --config configs\cifar10_paper.yaml
```

Before launching long runs, preflight the target configs. This checks config
schemas, estimated training scale, CIFAR archive presence/checksums, and TFDS
dependencies for MNIST/ImageNet:

```powershell
.\.ifaugnet\Scripts\python.exe scripts\preflight.py --output runs\preflight.json
```

## Tiny End-to-End Debug

This runs the real training script through every stage on a synthetic
CIFAR-shaped dataset. Use it before launching a TFDS-backed run.

```powershell
.\.ifaugnet\Scripts\python.exe scripts\train.py --config configs\synthetic_debug.yaml --stage all
```

There is also a small Wide-ResNet debug config for validating the WRN training
path without launching WRN-28-10:

```powershell
.\.ifaugnet\Scripts\python.exe scripts\train.py --config configs\synthetic_wrn_debug.yaml --stage all
```

And a small Shake-Shake debug config for validating stochastic branch mixing:

```powershell
.\.ifaugnet\Scripts\python.exe scripts\train.py --config configs\synthetic_shakeshake_debug.yaml --stage all
```

PyramidNet+ShakeDrop also has a small debug config:

```powershell
.\.ifaugnet\Scripts\python.exe scripts\train.py --config configs\synthetic_pyramidnet_debug.yaml --stage all
```

MNIST has a separate 28x28x1 debug config using the paper-style 4-layer CNN
path. It does not require downloading MNIST:

```powershell
.\.ifaugnet\Scripts\python.exe scripts\train.py --config configs\synthetic_mnist_debug.yaml --stage all
```

A real-data MNIST debug config exercises the direct IDX/GZip loader without
TensorFlow:

```powershell
.\.ifaugnet\Scripts\python.exe scripts\train.py --config configs\mnist_direct_debug.yaml --stage all
```

ImageNet has a small synthetic debug config that validates the 1000-class
ResNet-style backbone, Top-5 metric, and configurable 8-layer AugNet path
without requiring ImageNet:

```powershell
.\.ifaugnet\Scripts\python.exe scripts\train.py --config configs\synthetic_imagenet_debug.yaml --stage all
```

The streaming ImageNet runner can be checked without opening TFDS by using its
dry-run mode. This reports split names and step counts without initializing the
ResNet/AugNet states or creating a checkpoint directory:

```powershell
.\.ifaugnet\Scripts\python.exe scripts\train_imagenet_stream.py --config configs\imagenet_stream_debug.yaml --dry-run
```

## CIFAR Starter Runs

Quick real CIFAR-10 smoke run with a small data subset:

```powershell
.\.ifaugnet\Scripts\python.exe scripts\train.py --config configs\cifar10_debug.yaml --stage all
.\.ifaugnet\Scripts\python.exe scripts\train.py --config configs\cifar100_debug.yaml --stage all
```

By default, CIFAR configs use `data.source: direct`, which downloads and parses
the official CIFAR Python archives without TensorFlow. If automatic download is
slow on your network, download the archive manually from the
[CIFAR page](https://www.cs.toronto.edu/~kriz/cifar.html), place it under
`.data/cifar_raw/`, or set:

```yaml
data:
  archive_path: C:/path/to/cifar-10-python.tar.gz
```

Small default CIFAR configs:

```powershell
.\.ifaugnet\Scripts\python.exe scripts\train.py --config configs\cifar10.yaml
.\.ifaugnet\Scripts\python.exe scripts\train.py --config configs\cifar100.yaml
```

Table 1 low-label CIFAR configs use balanced `data.train_labels_per_class`
sampling and ResNet-56:

```powershell
.\.ifaugnet\Scripts\python.exe scripts\train.py --config configs\cifar10_table1_labels10.yaml
.\.ifaugnet\Scripts\python.exe scripts\train.py --config configs\cifar10_table1_labels100.yaml
.\.ifaugnet\Scripts\python.exe scripts\train.py --config configs\cifar100_table1_labels100.yaml
```

Table 1 MNIST configs use direct IDX/GZip loading, a 4-layer CNN, and balanced
1%/10% labeled subsets (approximately 60 and 600 labels per class), so they do
not require TensorFlow:

```powershell
.\.ifaugnet\Scripts\python.exe scripts\train.py --config configs\mnist_table1_labels60.yaml
.\.ifaugnet\Scripts\python.exe scripts\train.py --config configs\mnist_table1_labels600.yaml
```

Table 2 configs cover CIFAR-10 and CIFAR-100 with Wide-ResNet-28-10,
Shake-Shake (26 2x96d), and PyramidNet+ShakeDrop using standard
crop/flip/Cutout plus learned AugNet. A smaller `2x32d` Shake-Shake config is
kept only for lighter local experimentation.

```powershell
.\.ifaugnet\Scripts\python.exe scripts\train.py --config configs\cifar10_table2_wrn28_10.yaml
.\.ifaugnet\Scripts\python.exe scripts\train.py --config configs\cifar10_table2_shakeshake26_2x96d.yaml
.\.ifaugnet\Scripts\python.exe scripts\train.py --config configs\cifar10_table2_pyramidnet_shakedrop.yaml
.\.ifaugnet\Scripts\python.exe scripts\train.py --config configs\cifar100_table2_wrn28_10.yaml
.\.ifaugnet\Scripts\python.exe scripts\train.py --config configs\cifar100_table2_shakeshake26_2x96d.yaml
.\.ifaugnet\Scripts\python.exe scripts\train.py --config configs\cifar100_table2_pyramidnet_shakedrop.yaml
```

ImageNet paper-scale configs capture the paper's ResNet-50/ResNet-200,
50,000-sample hyper-validation split, batch size 4096, learning rate 1.6,
90/180/240 decay schedule, 8-layer AugNet `E/G` shape, and progressive
AugNet pretraining at 32/64/128/224 resolution:

```powershell
.\.ifaugnet\Scripts\python.exe scripts\train_imagenet_stream.py --config configs\imagenet_resnet50_paper.yaml
.\.ifaugnet\Scripts\python.exe scripts\train_imagenet_stream.py --config configs\imagenet_resnet200_paper.yaml
```

The ImageNet runner uses streaming TFDS iterators with Inception-style random
resized crop/aspect/color preprocessing for train-time batches and deterministic
resize/center-crop preprocessing for validation. The current Windows venv does
not install TensorFlow by default; use a TensorFlow-capable environment with
`imagenet2012` prepared under `data.data_dir`. For long ImageNet jobs, add
`--resume`; the runner restores latest stage checkpoints, skips completed
stages, and advances streaming iterators to the saved batch step. Full
paper-scale ImageNet still needs actual ResNet-50/200 metric runs.

The default `all` stage runs the paper-style sequence:

1. train the classifier `F`,
2. pretrain AugNet `E/G` against baseline augmented samples,
3. train AugNet with the last-layer influence objective,
4. retrain a fresh classifier with baseline augmentation plus learned AugNet.

You can also run stages separately:

```powershell
.\.ifaugnet\Scripts\python.exe scripts\train.py --stage classifier
.\.ifaugnet\Scripts\python.exe scripts\train.py --stage pretrain_augnet
.\.ifaugnet\Scripts\python.exe scripts\train.py --stage augnet
.\.ifaugnet\Scripts\python.exe scripts\train.py --stage retrain
```

Checkpoints are written under `runs/<dataset>` by default and ignored by git.
Each run writes `runs/<dataset>/run_manifest.json` with the resolved config,
command, package versions, and JAX device information. Scalar metrics are
appended to `runs/<dataset>/metrics.jsonl`. The checked-in `cifar10.yaml` and
`cifar100.yaml` configs are still small enough to validate the pipeline; use
`configs/cifar10_paper.yaml` or `configs/cifar100_paper.yaml` as a starting point
for longer paper-scale runs.

Long runs write stage-level latest checkpoints according to
`checkpoint_every_steps`. Resume an interrupted run from the latest stage
checkpoint, or skip already-completed stages, with:

```powershell
.\.ifaugnet\Scripts\python.exe scripts\train.py --config configs\cifar10_paper.yaml --stage all --resume
.\.ifaugnet\Scripts\python.exe scripts\train_imagenet_stream.py --config configs\imagenet_resnet50_paper.yaml --stage all --resume
```

To advance one long stage in smaller chunks, select that stage and add
`--stop-after-steps`. The command saves progress and exits before final stage
evaluation/checkpointing until the stage reaches its configured total:

```powershell
.\.ifaugnet\Scripts\python.exe scripts\train.py --config configs\mnist_table1_labels60.yaml --stage pretrain_augnet --resume --stop-after-steps 500
```

Summarize a run into flat CSV, JSON, and SVG curves:

```powershell
.\.ifaugnet\Scripts\python.exe scripts\summarize_metrics.py --metrics runs\cifar10_debug\metrics.jsonl
```

Collect multiple runs into a paper-style CSV/Markdown results table. By
default this reads the Table 1 low-label configs and reports baseline error,
AugNet retraining error, Top-5 when available, train/test scale, eval batch
coverage, error reduction, and key influence/AugNet diagnostics such as
`s_test` residuals and estimated validation loss reduction:

```powershell
.\.ifaugnet\Scripts\python.exe scripts\collect_results.py --output-dir runs\table1_results
```

Compare completed runs against the paper's Table 1/2/3 Proposed targets from
`configs/paper_targets.yaml`. Without `--strict`, missing metrics are reported
as pending rows; with `--strict`, any missing or failing target exits non-zero.

```powershell
.\.ifaugnet\Scripts\python.exe scripts\compare_to_paper.py --output-dir runs\paper_compare
.\.ifaugnet\Scripts\python.exe scripts\compare_to_paper.py --strict
```

For paper-style repeated runs, first materialize a suite of per-seed configs.
The dry run writes `runs/paper_suite/suite_plan.json` and generated configs
without launching training. Each plan item records the generated config,
checkpoint directory, command, and paper target IDs/metrics covered by that run:

```powershell
.\.ifaugnet\Scripts\python.exe scripts\run_paper_suite.py --dry-run
.\.ifaugnet\Scripts\python.exe scripts\preflight.py --suite-dir runs\paper_suite --output runs\paper_suite\preflight.json
.\.ifaugnet\Scripts\python.exe scripts\run_paper_suite.py --seeds 0 1 2 3 4 --stage all --resume
.\.ifaugnet\Scripts\python.exe scripts\suite_status.py --suite-dir runs\paper_suite --output-dir runs\paper_suite\status
.\.ifaugnet\Scripts\python.exe scripts\compare_to_paper.py --suite-dir runs\paper_suite --strict
```

`suite_status.py` reads `suite_plan.json`, each run's progress pickles, and
metrics JSONL to show which seeds are pending, in progress, complete, or
missing final metrics before strict paper comparison.

Use `--table`, `--dataset`, or `--target-id` to split the paper suite into
smaller, resumable batches. The same filters work for suite generation,
preflight, and paper comparison:

```powershell
.\.ifaugnet\Scripts\python.exe scripts\run_paper_suite.py --table 1 --dry-run --output-dir runs\paper_table1
.\.ifaugnet\Scripts\python.exe scripts\preflight.py --table 1 --strict
.\.ifaugnet\Scripts\python.exe scripts\run_paper_suite.py --table 1 --seeds 0 1 2 3 4 --stage all --resume --output-dir runs\paper_table1
.\.ifaugnet\Scripts\python.exe scripts\compare_to_paper.py --table 1 --suite-dir runs\paper_table1 --strict
```

For long jobs, run a single stage in resumable chunks. The command below
advances each Table 1 seed by 500 AugNet-pretraining steps and saves progress:

```powershell
.\.ifaugnet\Scripts\python.exe scripts\run_paper_suite.py --table 1 --stage pretrain_augnet --resume --stop-after-steps 500 --output-dir runs\paper_table1
```

Export original/augmented image pairs from an AugNet checkpoint:

```powershell
.\.ifaugnet\Scripts\python.exe scripts\visualize_augnet.py --config configs\cifar10_debug.yaml --checkpoint runs\cifar10_debug\augnet.msgpack --output runs\cifar10_debug\augnet_samples.ppm
```

Export a Figure 4-style tau interpolation grid. Rows are original image,
spatial-flow visualization, spatially transformed image, appearance-delta
visualization, and final transformed image:

```powershell
.\.ifaugnet\Scripts\python.exe scripts\visualize_tau_interpolation.py --config configs\cifar10_debug.yaml --checkpoint runs\cifar10_debug\augnet.msgpack --output runs\cifar10_debug\tau_interpolation.ppm
```

## Implementation Notes

The influence objective follows the paper's tractable approximation: compute
influence only through the final fully connected classifier layer, while still
backpropagating the resulting scalar through the frozen classifier feature
extractor into `E` and `G`.

The transformation decoder emits spatial `sw/sb` and RGB appearance `cw/cb`
fields. Both field groups are smoothed with the paper-style 4x4 average
pooling before spatial warping or 1x1 appearance filtering.

The AugNet objective logs `I_aug = Iup(augmented) - Iup(original)`, matching the
paper's replacement influence approximation. The pretraining step uses
relativistic-average GAN losses over image space and the classifier feature
space to match baseline augmented samples.

During the AugNet influence stage, `s_test` is precomputed and reused by
default, matching the paper's fixed iHVP training setup. Set
`augnet.precompute_s_test: false` in a config to recompute it every step for
debugging. Classifier and retraining configs also support step-decay learning
rates through `lr_decay_epochs` and `lr_decay_factor`.
Classifier weight decay is masked to convolution/dense `kernel` parameters only,
leaving BatchNorm scale/bias and classifier bias unregularized.

CIFAR configs support `max_train_size`, `max_hyperval_size`, and `max_test_size`
for quick real-data debugging without changing the full dataset split logic.
They also support `train_labels_per_class` for balanced low-label CIFAR
experiments; do not combine it with `max_train_size`. MNIST uses the same
balanced low-label field and uses random cropping without horizontal flips by
setting `baseline_augmentation: crop` in the classifier, pretraining, and
retraining sections. MNIST paper configs use `data.source: direct` to download
and parse the original IDX/GZip resources without TensorFlow.
CIFAR-10 and CIFAR-100 configs apply standard per-channel mean/std
normalization before classifier feature extraction; AugNet itself still operates
on `[0, 1]` images so its spatial and appearance transforms remain pixel-space
operations. `scripts/preflight.py` verifies CIFAR archives against the official
md5 sums published on the CIFAR download page before long runs.
Evaluation iterators keep the final partial batch and aggregate loss/accuracy
by example count, so paper test-error comparisons cover each test example once
instead of averaging batch means.
The NumPy training runner also keeps the final partial training batch by
default, which matters for low-label Table 1 settings such as 100 CIFAR-10
training examples with a batch size of 64.
The ImageNet streaming runner keeps `drop_last=True` for classifier and retrain
stages by default to keep paper-scale large-batch shapes stable; preflight uses
the same floor step-per-epoch convention for ImageNet.

This implementation targets MNIST, CIFAR, and ImageNet reproduction paths.
Wide-ResNet-28-10, Shake-Shake 26 2x96d, and PyramidNet+ShakeDrop are included
for CIFAR-10 and CIFAR-100. ImageNet now has ResNet-50/ResNet-200 configs,
Top-5 evaluation, an 8-layer AugNet scaffold, a streaming TFDS runner with
Inception-style preprocessing, and progressive-resolution AugNet pretraining.
The remaining gap is full paper-scale metric reproduction across the target
manifest, ideally through the 5-seed suite for CIFAR targets.

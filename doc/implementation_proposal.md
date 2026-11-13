# Dual resolution detector implementation proposal

Implementation status: runnable CNN and dual-resolution ResNet detectors, real-only spectral reconstruction and conditional flow matching references, Hydra configurations, MLflow tracking, and synthetic integration tests are now available. See the [README](../README.md#training-und-experimente) for the implemented interfaces and remaining research extensions. The repository assessment below records the state before this implementation; later sections remain a broader research proposal.

The dual-resolution design is a useful hypothesis, but its first prerequisite is preserving the pixels needed by the local branch. The current converter stretches every image to 128×128, and the repository does not yet provide a runnable supervised training and evaluation workflow. Build that workflow and a reproducible comparison of global, local, and fused models before investing in new self-supervised pretraining.

This proposal turns the [architecture proposal](dual_resolution_ai_image_detector_proposal.md) and [concept figure](dual_resolution.png) into data, model, training, evaluation, and delivery specifications. The initial objective is image-level classification of camera-origin versus fully generated images. Mixed or edited images require separate labels and evaluation; patch scores must not be presented as manipulation localization.

## Current implementation and gaps

Assessment date: 2026-11-13. Code references describe the working tree based on commit `fd76d51`, including existing local changes to training and metrics.

| Area | Present implementation | Consequence and required change |
| --- | --- | --- |
| Conversion | [`src/data/convert.py`](../src/data/convert.py), especially `add_image`, supports loose files, ZIP, split ZIP and ZIP64; applies EXIF orientation, composites transparency on white, and writes RGB PNG plus JSON into tar shards. | Reuse archive handling and error reporting. Replace mandatory 128×128 stretching with an explicitly versioned native-pixel mode. PNG encoding is lossless, but the preceding resize is irreversible. |
| Metadata | Samples contain `source`, `original_size`, `size`, and `mode`. Keys are sequential counters. `original_size` is measured before EXIF orientation. | Add stable identity, labels, source provenance, oriented dimensions, duplicate groups, and explicit splits. Sequential keys change when earlier input files change. |
| Streaming | [`StreamingImageDataset`](../src/data/streaming.py) reads shards sequentially and partitions them between workers/ranks. It returns `(image, metadata)` and pairs a PNG with a following JSON of the same key. | Add a supervised adapter, seeded shuffling, strict record validation, and a dual-view transform. The current reader can silently omit unmatched records. Shard partitioning alone does not give equal training steps across ranks. |
| Architecture | [`src/models/resnet/res_net.py`](../src/models/resnet/res_net.py) implements a custom residual classifier with adaptive average pooling. [`src/model/model.py`](../src/model/model.py) is empty. | Retain the custom model as a scratch baseline. Fix `from res_block import ResBlock` to a package-relative import and expose pooled embeddings. Its architecture is not a drop-in target for torchvision ResNet weights. |
| Trainer | [`src/training.py`](../src/training.py) has a basic gradient loop. `fit` ignores `val_split` and `scheduler_interv`, always steps the scheduler, and saves no checkpoint. | Add validation, explicit scheduler semantics, phase control, checkpoints, and structured batches. The reader's metadata dictionary cannot satisfy the trainer's `y.to(device)` call. |
| Metrics | [`src/online_metrics.py`](../src/online_metrics.py) computes sample-weighted mean loss and argmax accuracy. Its variance is the unweighted sample variance of batch losses. | Preserve the efficient accumulation. Give batch-loss variance an accurate name; do not interpret it as uncertainty in accuracy. Add ranking metrics and false-positive measurements to evaluation. Handle empty loaders and omit undefined numeric metrics from logging. |
| Entry points | [`scripts/train.py`](../scripts/train.py), [`scripts/eval.py`](../scripts/eval.py), [`scripts/inference.py`](../scripts/inference.py), [`src/evaluation.py`](../src/evaluation.py), and [`configs/config.yaml`](../configs/config.yaml) are empty. | README commands currently provide no training or predictions. Implement these interfaces before architecture comparisons. |
| Dependencies | Root [`requirements.txt`](../requirements.txt) lists many unrelated packages but does not explicitly declare `torch`, `torchvision`, Pillow, `python-dotenv`, or `mlflow`. [`src/requirements.txt`](../src/requirements.txt) declares only `torch`. | Consolidate direct dependencies and provide a tested installation path. The existing environment is not evidence that installation from the root file works. |
| Tests | [`tests/test_image_conversion.py`](../tests/test_image_conversion.py) contains four conversion/streaming tests, including multi-worker loading and split archives. | Retain these regression tests and add contract tests for labels, splits, native crops, training, and checkpoint inference. |

Observed validation: all four existing tests pass using `.venv/bin/python -m unittest discover -s tests -v` outside the restricted sandbox. The sandbox blocks local sockets used by the worker test. Package import checks fail for the custom ResNet (`No module named 'res_block'`) and trainer (`No module named 'mlflow'`). The local `data/processed-v2/manifest.json` reports zero images, no shards, and four incomplete archives; despite the directory name, it uses the current unversioned 128×128 format. It cannot support a model benchmark.

## Architectural decisions

Keep separate global and local encoders, simple feature concatenation, and mean pooling across local crops. Separate encoder parameters allow specialization, but do not themselves establish complementary features or prevent dataset shortcuts. Neither the proposal nor the current code establishes that reconstruction pretraining improves detection of unseen generators.

Use two independent torchvision ResNet-18 instances with `IMAGENET1K_V1` initialization for the first controlled experiment. This gives an inexpensive, reproducible reference and isolates the effect of native crops before changing backbone and pretraining objective together. It is a supervised-pretrained reference, not the independent SSL system illustrated in the concept figure. Introduce an existing DINOv2 global encoder and local reconstruction pretraining as subsequent experiments.

Frozen representation baselines deserve a serious comparison: Ojha et al. demonstrated generalization using pretrained vision-language features with simple classifiers. That supports testing frozen features, not a guarantee about this two-branch design. [Towards Universal Fake Image Detectors](https://arxiv.org/abs/2302.10174). DINOv2 provides general-purpose self-supervised visual features, which motivates the later global-backbone experiment without establishing forensic performance. [DINOv2](https://arxiv.org/abs/2304.07193).

The following decisions resolve ambiguities in the initial proposal:

| Question | Initial specification | Reason |
| --- | --- | --- |
| What is native resolution? | Pixels after deterministic decoding, orientation, and RGB conversion, before any geometric resize by this pipeline. | An uploaded image may already have been resized; original camera or generator resolution cannot be inferred from its dimensions. |
| How is the full image represented? | Fit the whole image inside 224×224 with preserved aspect ratio and symmetric padding. | Retains the entire scene without stretching it. Padding remains a possible shortcut and must be audited. |
| How many patches? | One for the first comparison; four as the next experiment, using the same budget for training and evaluation. | Isolates the benefit and cost of additional evidence. |
| What if either dimension is below 256? | No local crop; route to a global head trained for this fallback. | Avoids manufacturing detail by upscaling or introducing artificial local padding. |
| Are small images discarded? | No; report coverage and separate results for fallback images. | Excluding them could create an easy, unrepresentative benchmark. |
| Which classifier convention? | Two logits, integer labels `0=real`, `1=generated`, cross-entropy loss. | Matches the current argmax metric convention and avoids mixed sigmoid/softmax semantics. |
| Is frequency input required? | No explicit FFT branch in the first implementation. | A native RGB crop does not automatically implement frequency-domain modeling. A spectral objective or input needs its own controlled experiment. |
| What is deferred? | Cross-attention, learned patch weighting, RL selection, and training a global foundation encoder from scratch. | First establish whether the second view adds measurable value. |

## Data specifications

### Source inventory and labels

Introduce `scripts.prepare_dataset` and a source inventory with one record per original asset, including archive members. Dataset adapters must assign labels from documented dataset provenance. A reviewed mapping from source directories may be an adapter, but training must consume explicit labels and never infer them from arbitrary path substrings.

Required inventory fields:

| Field | Contract |
| --- | --- |
| `sample_id` | SHA-256 of original encoded asset bytes. Retain multiple provenance records for identical assets, but include the content only once in a split. Conflicting labels require resolution or quarantine. |
| `source` | Structured relative locator: file path plus optional archive member and member index. Do not rely on splitting an ambiguous string containing `!`. |
| `label`, `label_origin` | Integer `0` or `1`, plus dataset/version or annotation source establishing the label. Unknown, mixed, or ambiguous labels are excluded from supervised training and counted. |
| `dataset_id`, `source_domain` | Versioned dataset identifier and collection or acquisition domain. These are reporting fields, never detector inputs. |
| `generator_family`, `generator_version` | Required for generated examples in held-out-family experiments; null for real images. Unknown-family generated samples may be used only in separately named experiments. |
| `group_id` | Identity shared by duplicates, derived images, bursts, or known related generations. Includes exact decoded-pixel matches and reviewed near-duplicate candidates. |
| `encoded_size`, `oriented_size` | `[width, height]` before and after EXIF orientation. Store the orientation operation explicitly. |
| `source_format`, `source_mode`, `compression` | Observed original encoding, color mode, and available compression evidence, such as JPEG quantization-table digest and subsampling. Unknown values remain null; do not invent a JPEG quality number. |
| `pixel_hash`, `preprocess_version` | Digest of canonical RGB pixels including dimensions, and the exact canonicalization recipe identifier. |
| `split` | One immutable assignment from the split protocol below; derived views inherit it. |

Canonicalization v1 uses the first frame, EXIF transpose, conversion to RGBA, white alpha compositing, and conversion to RGB in that order, matching current behavior except for removing the resize. It must record ICC-profile presence and source bit depth; implicit RGB conversion is not a claim of color-managed conversion. Keep original assets available so alternate color handling can be evaluated later. Version any change to this recipe.

### Native storage and migration

Extend conversion with explicit profiles `legacy128` and `native_v2`; preserve the current default until documentation and consumers migrate. The native profile writes `<sample_id>.png` at the oriented dimensions and `<sample_id>.json` containing the validated inventory record. Do not precompute just one local crop: doing so would permanently restrict later sampling experiments.

Each split gets its own manifest and shards. A native manifest must declare `schema_version: 2`, `profile: native_v2`, `resize: none`, `preprocess_version`, inventory and split digests, class/domain/family counts, and each shard's image count, byte size, and checksum. Write to `.partial` files, rename completed shards, and publish the manifest atomically only after all referenced shards are complete. Readers reject unknown versions and missing or mismatched members instead of silently dropping examples.

Retain the existing image-count cap and add a 1 GiB target shard size, closing before the next record would exceed the target. A single large record may exceed that target and must be reported. Default ingestion limits are 50 million decoded pixels and 256 MiB per encoded member; quarantine larger assets with a reason instead of resizing them silently. These are initial configurable resource limits, with exclusion counts broken down by label and domain.

Migration must reconvert from the original assets into a new directory. Never upscale legacy128 outputs and call them native. A dual-resolution run must reject a legacy manifest even if original dimensions remain in its metadata. Legacy data remains usable for a separately identified 128×128 baseline. Estimate native storage and decoding cost from a representative sample before a full conversion.

### Splits and leakage prevention

Create and freeze split assignments before crops, augmentation, detector training, or any project-specific self-supervised pretraining. Exact byte hashes, canonical pixel hashes, and known derivation relationships must not cross splits. Use a 64-bit perceptual hash with Hamming distance at most 4 as an initial near-duplicate candidate search; review candidates before merging groups, and version the resulting grouping decisions. A perceptual hash alone is not a guarantee against leakage.

Maintain these split roles:

| Split | Purpose |
| --- | --- |
| `train` | Supervised fitting and project-specific pretraining. |
| `val_id` | Model selection within known generator families and collection domains. |
| `val_ood` | Development families/domains withheld from fitting; used for architecture and hyperparameter selection. |
| `calibration` | Disjoint groups from known domains for fitting a score calibration and operational threshold after model selection. |
| `test_id` | Locked in-distribution test. |
| `test_ood` | Locked generator families and real-image source domains never used for model selection or calibration. |

For sources not reserved for OOD evaluation, start with group assignments of 70% train, 10% validation, 10% calibration, and 10% test, stratified by label and generator/source where feasible. Percentages are targets at group level; publish actual counts. OOD families and source domains are assigned explicitly, not by random image splitting. Include both real and generated images in each scored evaluation suite.

Require at least two known generator families for training, one development family, and two final held-out families for the initial generalization claim. If the inventory cannot support that, use grouped cross-validation over the available families and report it as exploratory; it does not replace a locked external test. Hold out real collection domains as well, otherwise recognition of the real dataset can masquerade as generator generalization. Freeze related real/generated pairs and common source-image derivatives together.

Publish a cross-tabulation of class, domain, format, dimensions, aspect ratio, and local-crop availability. Build matched diagnostic subsets using shorter-side bins `<256`, `256–511`, `512–1023`, and `>=1024`, format, and known JPEG metadata. Matching and reweighting must be reported separately from results on the natural test distribution. Where sources and labels are inseparable, collect additional data; balancing image counts cannot resolve that confound.

### Sampling and resource behavior

Keep the tar streaming approach. Shuffle the training shard list using `(run_seed, epoch)` and shuffle encoded records inside each worker with a buffer bounded by both 512 records and 128 MiB. Decode only when consuming a record; buffering hundreds of full-resolution float tensors is unacceptable. Keep validation and test order deterministic.

Derive crop randomness from a stable hash of `(run_seed, epoch, sample_id, view_index)`, independent of worker scheduling. Add `set_epoch`; initially recreate nonpersistent workers each epoch so epoch changes are visible. Evaluation crop selection must not depend on RNG state.

Implement the first release on one training process, with DataLoader workers allowed. Explicitly reject distributed training until a tested epoch planner assigns equal full-batch counts to each rank. A later DDP implementation must construct balanced shard assignments, deterministically truncate each rank to the smallest available full-batch count, record omitted samples, and rotate assignments across epochs. Evaluation must cover each sample exactly once across ranks and aggregate metrics globally. Simply adding `drop_last=True` to the current iterable loader does not establish equal rank lengths.

## Input and model contracts

### View construction

Decode one canonical image and derive both views from it. For global input, use scale `s = 224 / max(width, height)`, round resized dimensions to the nearest integer with a minimum of one, and bilinear resize with antialiasing. Center the result in a 224×224 tensor; put any odd extra padding on the right or bottom. Fill padding with the normalization mean so it becomes zero after normalization. Upscaling a small global image is permitted and is not counted as new native evidence.

For a valid local view, select a 256×256 crop directly from canonical RGB pixels with integer coordinates and no interpolation. Training samples top-left coordinates uniformly without replacement, taking at most `min(K, (width-255)*(height-255))` crops. Evaluation with `K=1` uses the center, with offsets rounded down; `K=4` uses top-left, top-right, bottom-left, and bottom-right. Deduplicate identical coordinates, retain the first occurrence, and mask unused slots. Overlap is allowed. An image exactly 256×256 has one valid patch, but it offers limited scale separation; report that case separately.

For both branches convert to float32 `[0,1]`, then normalize with mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`. These values follow the chosen ResNet weights. Their bundled transform uses a center crop, so reuse the normalization but implement the whole-frame transform explicitly. [Torchvision ResNet-18 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).

Start with no color jitter, blur, random resized crops, JPEG augmentation, or flips. Add corruption training as a named ablation after the clean baseline. For that ablation, apply a transformation to the canonical image before generating both views, with the same class-independent probability; record the post-transformation dimensions and availability. This prevents inconsistent branch preprocessing from becoming an unintended cue.

### Batch and forward interface

The supervised dataset and custom collator must produce:

```python
batch = {
    "global": FloatTensor[B, 3, 224, 224],
    "local": FloatTensor[B, K, 3, 256, 256],
    "local_mask": BoolTensor[B, K],
    "target": LongTensor[B],
    "loss_weight": FloatTensor[B],     # training weights; ones for evaluation
    "sample_id": list[str],
    "crop_boxes": LongTensor[B, K, 4],  # x, y, width, height; -1 if absent
    "metadata": list[dict],            # CPU only, for reports
}
```

Missing patch slots contain zeros after normalization and must never be passed through the local encoder. `forward(global_view, local_views, local_mask)` returns `logits [B,2]`, `global_logits [B,2]`, `local_logits [B,2]`, and `local_valid [B]`; local logits are defined as zero and excluded from scoring/loss where invalid. Paths, generator identity, image dimensions, and crop coordinates must not enter the neural classifier.

### Encoder and fusion layers

Each ResNet-18 exposes the 512-dimensional vector before its final classification layer. Each branch has its own `LayerNorm(512) -> Linear(512,256) -> GELU` projector. The local encoder processes only valid crops, and their projected embeddings are pooled with a masked arithmetic mean. An empty set produces a zero vector without division by zero.

The global and local classification heads each use `Linear(256,128) -> GELU -> Dropout(0.1) -> Linear(128,2)`. The fusion head uses the same structure with 512 input dimensions after concatenation. If any patch is valid, the model selects the fusion logits; otherwise it selects the global logits. Local-only evaluation must return “unavailable” for missing local views and report coverage rather than assigning fabricated predictions.

For a sample with a valid local view, the initial loss is:

```text
CE(fusion_logits, target) + 0.2 * CE(global_logits, target)
```

For a sample without one, use only `CE(global_logits, target)`. Apply the training weights specified below and average per-image losses over the batch, so adding patches does not increase an image's weight. The auxiliary global term trains a useful fallback even when most training images have valid crops. The local auxiliary coefficient is initially zero; test `0.2` separately. Train standalone global/local baselines with the same weighting and ordinary cross-entropy, without auxiliary terms.

Freeze encoder parameters and keep their BatchNorm layers in evaluation mode during head training, including after the parent model receives `.train()`. Fine-tuning the last residual stage must still leave BatchNorm running statistics frozen initially. The current unconditional `model.train()` would otherwise invalidate the meaning of “frozen encoder.”

Zero masking still exposes local availability indirectly. Audit performance by availability, train a metadata baseline including it, and test dropping all local patches on 10% of otherwise eligible training examples as an ablation. Do not add an explicit availability feature to the classifier. Compare full-frame letterboxing with stretching as a diagnostic for aspect-ratio and padding shortcuts, while keeping the main whole-frame policy fixed.

## Training and runtime specifications

### Training phases

Use AdamW and a fixed learning rate in the initial reference run; scheduler `none` must be a supported value. Begin with these proposed defaults, then tune only on development splits:

| Phase | Trainable parameters | Initial schedule |
| --- | --- | --- |
| Frozen reference | Projectors and active classification heads | Up to 10 epochs; learning rate `1e-3`; weight decay `1e-4`. |
| Selective fine-tuning | Above plus each encoder's last residual stage | Start from the selected frozen checkpoint; up to 10 epochs; head rate `1e-4`, encoder rate `1e-5`; weight decay `1e-4`. |
| Full fine-tuning comparison | All encoder stages plus heads | Separate run from the same frozen checkpoint; same rates and maximum epochs as selective fine-tuning. |

Use batch size 16 and seeds `17, 29, 43` for comparisons. Allow gradient accumulation if memory requires it, keeping effective batch size recorded. Start in float32; enable mixed precision only after comparing finite losses and inference scores on a fixture. Record device, peak accelerator memory, throughput, and total training time. The one-patch fused model requires one global and one local encoder pass per image; four patches require five total passes before any batching benefits.

At every epoch, run deterministic validation with inference mode and compute all selection metrics. Select the checkpoint maximizing macro generator AUROC on `val_ood`, with lower validation cross-entropy as the tie-breaker; stop after three epochs without improvement. If no valid OOD development split exists, use `val_id` and mark selection accordingly. Never change the selector after observing the final test. A class-missing selection split is an error, not an AUROC of zero.

Visit each training image once per epoch using the bounded shuffle. Equalize the contribution of classes, generator families or real source domains, and related-image groups through per-image loss weights computed from the training inventory. For an image in class `c`, stratum `s` (family for generated, domain for real), and group `g`, set `w = N_train / (2 * S_c * G_cs * N_csg)`, where `S_c` is the number of strata in the class, `G_cs` is its stratum's group count, and `N_csg` is that group's image count within the stratum. This gives mean weight one over the full training set and equal total contribution per class, without requiring random seeks through tar shards. Recompute weights for the eligible training subset in local-only runs.

Use `mean(w_i * loss_i)` per batch; do not renormalize weights within individual batches or add class-weighted cross-entropy on top. Keep the last partial batch. Report the weight distribution and an unweighted-training ablation, since sparse groups may otherwise exert disproportionate influence. Validation, calibration, and test metrics use their original prevalence and unweighted losses. Log actual images and batches processed; do not assume the length of an iterable DataLoader is exact with multiple workers.

### Configuration and commands

Use `src/model/` as the canonical model package, as already documented in the README. Move the scratch residual implementation there during implementation and retain forwarding imports from `src/models/resnet/` if consumers need them. Do not grow two parallel model registries.

The proposed `configs/dual_resolution.yaml` must contain these values plus the dataset/checkpoint locations and split digests. Resolve and save all defaults; reject unknown keys and invalid combinations such as native crops from legacy shards.

```yaml
schema_version: 1
seed: 17
data:
  root: data/native-v2
  profile: native_v2
  workers: 2
  shuffle_records: 512
  shuffle_bytes: 134217728
  train_weighting: balanced_class_domain_group
views:
  global_size: 224
  global_resize: letterbox_bilinear
  local_size: 256
  patches: 1
  eval_sampler: center
  small_image_policy: global_only
model:
  global_encoder: torchvision_resnet18
  local_encoder: torchvision_resnet18
  weights: IMAGENET1K_V1
  share_weights: false
  embedding_dim: 256
  pooling: masked_mean
  mode: fused
training:
  phase: frozen
  batch_size: 16
  epochs: 10
  patience: 3
  optimizer: adamw
  head_lr: 0.001
  weight_decay: 0.0001
  scheduler: none
  global_aux_weight: 0.2
  local_aux_weight: 0.0
  local_dropout: 0.0
evaluation:
  selection_split: val_ood
  selection_metric: macro_generator_auroc
  target_fpr: 0.01
logging:
  backend: jsonl
  output_dir: results/dual-resolution-seed17
```

The following commands are interfaces to implement, not currently available functionality:

```bash
python -m scripts.prepare_dataset --sources configs/sources.yaml --output data/inventory
python -m scripts.convert_images --profile native_v2 --inventory data/inventory/samples.jsonl --output-dir data/native-v2
python -m scripts.train --config configs/dual_resolution.yaml
python -m scripts.eval --checkpoint results/dual-resolution-seed17/best.pt --split test_ood
python -m scripts.inference --checkpoint results/dual-resolution-seed17/best.pt --input example.jpg
```

Extend `scripts.convert_images` while retaining its existing command and tests. The new inventory includes split assignments; conversion writes split directories under its output root. For four-patch experiments, set `patches: 4` and `eval_sampler: corners`; reject incompatible sampler/budget settings.

Consolidate dependencies into one project definition with direct dependencies on PyTorch, torchvision, Pillow, NumPy, PyYAML, python-dotenv, and scikit-learn. Make MLflow an optional logging extra with lazy import. JSONL metrics and local artifacts must work without an MLflow service or package. Validate a compatible torch/torchvision pair on the chosen Python version and record a lock for the supported platform; do not replace the existing list with another indiscriminate environment freeze.

### Checkpoints and inference

Write `last.pt` after each completed epoch and `best.pt` when the selection metric improves, using atomic replacement. Include model, optimizer, optional scheduler, and mixed-precision scaler states; epoch and update counters; Python/NumPy/PyTorch RNG states; resolved config; model and preprocessing schema versions; label map; backbone weight identifiers and checksums; split/inventory digests; code revision and dirty-tree status; and environment versions.

Guarantee deterministic epoch-boundary resume on the same supported software/device configuration. Mid-epoch cursor restoration is deferred. A changed inventory, normalization recipe, crop policy, or model shape must fail resume validation unless explicitly starting a new run. Weight acquisition belongs to setup; inference must load local checkpoint contents without downloading a backbone.

Inference shares the evaluation transform and returns one structured record containing `sample_id`, generated score, thresholded label, calibration status, threshold, number of valid patches, selected route (`fused` or `global_only`), crop coordinates, and checkpoint/preprocess identifiers. Define the generated score as `softmax(selected_logits / temperature)[1]`, with temperature 1 before calibration. Save the fitted temperature and threshold in a versioned calibration artifact bound to the checkpoint digest; use it in both evaluation and inference. Without that artifact, use threshold 0.5 and mark the result uncalibrated. Treat the score as model evidence, not proof of image origin. Corrupt inputs return a specific error record and no prediction; batch commands return nonzero if any input fails.

## Evaluation and decision criteria

### Required comparisons

Run the following in order with the same data assignments and seeds. Use the full supported image population for global/fused evaluation and the same eligible subset for direct global/local/fused comparisons.

| Experiment | What it isolates |
| --- | --- |
| Metadata-only logistic regression | Resolution, aspect ratio, source format, JPEG evidence, and local availability as shortcuts; no paths or source IDs as features. Fit preprocessing on training data only. |
| Global-only ResNet reference | The useful baseline before extra views. |
| Native local-only reference with one crop | Whether local evidence is independently predictive, with coverage disclosed. |
| Frozen two-branch fusion with one crop | Whether a second view improves upon the single branches. |
| Two independent global encoders with the same fusion head | Whether extra model capacity alone explains a fusion gain. Match the two-encoder training recipe and report the remaining input-size/compute difference. |
| Selective and full fine-tuning | Whether supervised adaptation improves or harms OOD performance. |
| Four native crops with masked mean pooling | Accuracy versus the extra patch budget. Compare fixed corners and seeded random evaluation crops at the same budget. |
| Native versus downsampled local control | Whether fine detail matters: downsample each selected native patch to 128×128 and restore to 256×256, keeping field of view and model constant. |
| DINOv2 global features and local pretraining variants | Whether specialized representations improve the established two-view reference. |

Compare auxiliary losses, local dropout, and corruption augmentation one at a time against the selected reference. Do not execute a full Cartesian product of all options. Before the more expensive experiments, inspect class/domain counts, metadata-baseline performance, and performance by image size.

### Metrics and robustness

Generated is the positive class. Store per-image score, label, group, source domain, generator family, selected route, and corruption variant. Compute AUROC, average precision, balanced accuracy, confusion counts, and cross-entropy. Report average precision with class prevalence, since it depends on prevalence. For macro generator AUROC, compare each generated family against a predefined real-image reference cohort, then average family AUROCs equally. Keep that real cohort fixed across model comparisons and disclose its composition.

Select the operational threshold on the calibration split's real images only: choose the lowest threshold with empirical false-positive rate at most 1%, using `score >= threshold` as the positive rule and handling tied scores conservatively. Report test TPR and actual test FPR at this fixed threshold; a nominal 1% calibration FPR is not a guarantee of 1% test FPR. Require at least 1,000 independent real groups for the initial 1% operating-point report; otherwise mark it exploratory and show confidence intervals and the available group count. Do not make a 0.1% FPR claim from this minimum sample size.

If calibrated probabilities are needed, split the calibration groups into disjoint temperature-fit and threshold-selection subsets, maintaining the minimum real-group count in the latter. Fit one positive temperature using cross-entropy and report Brier score and a 15-bin equal-width reliability diagram. Keep the calibration method and threshold fixed across the final test corruptions.

Evaluate clean images and these separate, deterministic transformations of the canonical input: JPEG quality 95, 75, and 50 with fixed encoder settings and 4:2:0 subsampling; downscaling by 0.75 and 0.5 with bilinear antialiasing; and downscale 0.5 followed by JPEG quality 75. Apply transformations before extracting either view. Report changes in local availability alongside changes in metrics; post-resize local pixels are native to the delivered test image, not the original asset. Store the exact transform recipe and codec versions.

Report per-generator, per-real-domain, resolution-bin, format, and fallback results, as well as aggregate scores. Use 1,000 paired bootstrap resamples of `group_id` for 95% confidence intervals on differences between models, retaining the same groups and real reference cohort within a resample. Recompute each seed's metric in every resample and take the mean paired difference across seeds; do not average scores into an undeclared ensemble. Report seed mean and standard deviation separately; do not treat repeated patches, corruptions, or seed predictions as independent images. Undefined subset metrics are null with an explanatory count.

Profile end-to-end latency including decode and transforms, batch-one p50/p95 latency, batched throughput, peak accelerator memory, and the average/maximum actual patch count on named hardware. A model with four crops is not comparable in cost to a one-crop model without these measurements.

### Acceptance gates

Engineering completion requires a clean-environment CPU smoke run that converts fixtures, trains for two epochs, validates, saves and resumes a checkpoint, and produces matching evaluation/inference scores. All acceptance tests below must pass. This is independent of whether the research hypothesis succeeds.

For the first research milestone, predeclare a practical target of at least **0.02 absolute macro generator AUROC improvement** over the strongest single-branch development baseline, with a paired 95% confidence interval above zero on the locked OOD test. Select that baseline and compute this primary comparison on the common native-crop-eligible cohort, so local-only coverage cannot change the comparison population. Also publish full-population global/fused results. Require no more than **0.02 absolute balanced-accuracy loss** against the standalone global baseline on either the full clean ID cohort or the small-image fallback cohort. Use the same fixed calibration procedure for all models. Report all seeds and individual families even when the aggregate passes. These are proposed project thresholds, not expected results.

Select the strongest single-branch baseline and freeze all settings before opening the final test. The double-global control determines whether gains can reasonably be attributed to distinct views; if it matches fusion, report that the experiment has not isolated a native-detail benefit. If the predefined quality gate fails, retain the simpler model and report the result. Do not tune on the locked test to obtain a passing result. No deployment latency limit is specified yet; choose among qualifying models using measured cost until target hardware and a service budget are supplied.

## Later pretraining and patch selection experiments

Use a pinned, locally cached DINOv2 ViT-S/14 checkpoint as the first self-supervised global candidate, adapting its feature dimension through the same projector interface. Keep the 224×224 full-frame policy and verify the checkpoint's normalization and feature contract. Compare frozen and selectively fine-tuned variants with the local encoder unchanged. Record external pretraining provenance; unseen by project training does not establish absence from a foundation model's pretraining corpus.

Preserving input pixels alone does not guarantee detail-preserving features: the ResNet reference immediately reduces spatial resolution through its stride-2 convolution and max pooling. As a separate local-encoder experiment, keep the convolution weights and kernel shape but change its stride to 1 and replace the initial max pool with identity. Fine-tune under the same recipe and measure the added memory and compute. Keep this change separate from reconstruction pretraining so their effects can be distinguished.

For the local reconstruction experiment, keep the same local encoder architecture and training-only inventory. Mask 50% of 16×16 blocks in a 256×256 RGB crop, replace masked pixels by the normalization mean, and attach a temporary convolutional upsampling decoder producing a 256×256 RGB reconstruction. The first objective is mean L1 error on masked pixels only, measured in `[0,1]` RGB space. Compare it with the same objective plus `0.1 * L_spectral`, where `L_spectral` is mean L1 distance between `log(1 + abs(rfft2(image, norm="ortho")))` of the reconstruction and target over all channels/frequencies. Fix the mask seed and recipe per run.

Pretrain these variants for the same 20-epoch budget and starting encoder weights, discard the decoder, then use the identical detector training recipe. Include zero-pretraining and continued supervised-local-training controls; record total compute. Select using downstream `val_ood` performance, not reconstruction loss. This is an exploratory objective: successful reconstruction may emphasize common image content rather than evidence useful for classification.

Only after fixed multi-patch sampling improves quality, compare attention pooling against the masked mean at equal `K`. RL selection requires an additional gate: demonstrable room for improvement over random and fixed-grid crops at the same maximum patch count, including selection overhead. A candidate training reward is the incremental increase in true-label log probability minus a fixed cost per patch. At inference the policy must observe only the global features and already sampled patches, with stop action and maximum budget four; it cannot use labels or classification correctness. Defer RL if ordinary multi-crop fusion does not justify its cost.

## Implementation sequence and acceptance tests

Each milestone should be a reviewable change with its own runnable command and artifacts.

| Milestone | Files and responsibilities | Acceptance evidence |
| --- | --- | --- |
| 1. Runnable foundation | Consolidated dependency definition; `src/model/` imports and embedding interface; `src/training.py`, `src/online_metrics.py`, basic config and train/eval/inference scripts. | Imports work without MLflow. CPU fixture training, validation, checkpoint loading, and inference run. Empty data raises a clear error. Scheduler `none` works; update/epoch/metric scheduler modes are explicit if added. |
| 2. Versioned data | Extend `src/data/convert.py` and its CLI; add `src/data/inventory.py`, `src/data/splits.py`, `scripts/prepare_dataset.py`; extend `src/data/streaming.py`. | Original-pixel preservation, explicit labels, deterministic IDs and splits, duplicate-group isolation, strict record pairing, checksum validation, and native/legacy rejection tests pass. Existing archive tests remain green. |
| 3. Dual views and model | Add `src/data/views.py`, `src/data/collate.py`, `src/model/encoders.py`, and `src/model/dual_resolution.py`. | Coordinate-pattern fixtures prove crops contain exact source pixels. Portrait, landscape, EXIF-rotated, exactly 256×256, narrow, and small images follow the specified policies. `K=1/4` and all-missing masks yield finite outputs. Frozen weights and BatchNorm buffers do not change after training steps. |
| 4. Reproducible experiments | Complete trainer phases, balancing, deterministic workers, config validation, checkpoints, and optional logging. | Same seed gives the same per-image crops across worker counts; different epochs change training crops. Epoch-boundary resume matches uninterrupted CPU training under the same configuration. Every input has the required class/provenance fields. |
| 5. Evaluation and decision | Implement `src/evaluation.py`, calibration, corruption suites, report generation, and baseline configurations. | Known-score fixtures verify AUROC/AP and thresholds including ties and single-class subsets. Eval and inference scores match for the same checkpoint/input. Final report includes all mandatory baselines, coverage, grouped intervals, and cost measurements. |
| 6. Research extensions | Separate configs and modules for DINOv2, local pretraining, attention, and optionally selection policy. | Each extension changes one declared factor, uses the established splits, and reports its incremental quality and compute cost. |

Update the README during implementation with real installation commands, dataset provenance instructions, and runnable examples. Archive each experiment's resolved configuration, checksums, metrics, and selection decision under `results/<run_id>/`. Keep large weights ignored as they are now. The first dataset preparation milestone must resolve generator provenance, complete source archives, and enough independent real groups; architecture work cannot substitute for those inputs.

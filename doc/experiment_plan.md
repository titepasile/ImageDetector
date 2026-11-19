# Experiment plan

Status: running since 2026-11-19 on the `test` branch (CPU only, 16 cores).
Runner: [`scripts/run_experiment_plan.sh`](../scripts/run_experiment_plan.sh).
Results are collected in [`experiment_results.md`](experiment_results.md) once runs finish.

## Fixed setup

| Item | Value |
| --- | --- |
| Training data | GenImage ADM, debiased by `scripts.prepare_genimage`: 256×256 PNG, both classes JPEG-compressed once with matched quality/subsampling. 4,000 images per class for training, 1,000 per class for validation (both from GenImage `train/`). |
| Views | Default 128×128 global view (whole image, resized) and one native 128×128 crop. |
| Test sets (fixed) | `adm256`: 1,000 + 1,000 from GenImage ADM `val/`. `adm128`: the same images, centre 128 crop (no resampling). `biggan128`: 1,000 + 1,000 from GenImage BigGAN `val/`, native 128×128. |
| Metrics | AUROC and TPR at 5% FPR with 95% bootstrap intervals (1,000 class-stratified resamples); FPR/TPR at thresholds chosen on validation (5% FPR and max balanced accuracy). |
| Seeds | 17, 29, 43 for the main comparison; 17 otherwise. |
| Optimiser | AdamW, lr 1e-3, cosine decay per epoch; early stopping after 4 epochs without improvement. |

All evaluation outputs: `outputs/plan/analysis/<run>/<test set>/` with `predictions.jsonl`, `metrics.json` and `summary.json`.

## Experiments

| # | Question | Method | Cost (approx.) |
| --- | --- | --- | --- |
| 1 | Did image borders cause the AUROC of 1.0 at 256 px? | Re-score the old 256 checkpoints after replacing a 16/32 px border with grey or a reflection, identically for both classes. Occlusion maps (13) give a second view. | 15 min, no training |
| 2 | Is the 0.5 threshold usable? | Thresholds chosen on validation (5% FPR, balanced accuracy); report test FPR/TPR at them and threshold-free TPR at 5% FPR. | minutes |
| 3 | Baseline vs. dual-resolution, robustly | 3 seeds × 15 epochs each, mean ± spread. | see 4/5 |
| 4 | Does each branch help? | Global only (`baseline`), local only (`local_only`), fused (`dual_resolution`), same data and seeds. | ≈ 13 h with 3 and 5 |
| 5 | Is training long enough? | 15 epochs with cosine decay; validation curves per epoch. | included above |
| 6 | Does it generalise the other way? | Train on BigGAN (4,000 per class, 128 native), test on ADM. | ≈ 1.7 h |
| 7 | Robustness | Test images re-compressed (JPEG 90/75/50), resampled (×0.75 and back) and blurred (σ = 1), identically for both classes. | evaluation only |
| 11 | Which frequencies differ between real and AI? | Mean power spectrum of the noise residual (image − 3×3 median) per class and generator. | 15 min |
| 12 | Which frequency band does the model use? | Test images low-passed or high-passed at 0.125 cycles/pixel. | evaluation only |
| 13 | Where in the image does the model look? | Occlusion maps: score change when each cell of an 8×8 grid is replaced with the mean colour; border/interior ratio. | 10 min per model |
| 14 | Content or low-level traces? | Shuffle 32×32 or 64×64 tiles: content is destroyed, local statistics stay. | evaluation only |
| 15 | Does augmentation help? | Training-time blur (σ ∈ [0, 3]) and JPEG (q ∈ [30, 100]), each with p = 0.5 (Wang et al., 2020); baseline and dual. | ≈ 2.5 h |
| 16 | Does more data help? | Baseline with 2k / 4k / 10k / 20k images per class (nested subsets of one 20k build). | ≈ 3 h |
| 17 | Which crop size? | Dual-resolution with 32 / 64 px local crops (128 is the main run). | ≈ 40 min |
| 18 | Is the model too small? | Baseline with channels 32/64/128; dual with base_channels 32 (6 epochs, cost ×4). | ≈ 3.5 h |
| 19 | False positives on real photos from other sources? | **Blocked:** needs real photos that are not from ImageNet (e.g. own photos or COCO). | – |
| 20 | More generators | **Partly blocked:** only ADM and BigGAN are usable. GLIDE contains only the last split volume (0 AI images); Midjourney, VQDM, SD 1.4/1.5 and Wukong folders are empty. 6 and 21 cover two generators. | – |
| 21 | Train on two generators | ADM + BigGAN at 128 native, 2,000 per class each; real images deduplicated across sources. | ≈ 1.7 h |
| 22 | How certain are the numbers? | Bootstrap intervals for every result (see Metrics). | minutes |

## Known confounds

* The main runs (3–5) train on 256 px ADM images: the global view is downscaled ×2, the local crop is native. `adm128` and `biggan128` are native 128, so the global view is not rescaled there. The local branch sees native pixels in every test set; the global branch does not.
* BigGAN real images (ImageNet `val`) and ADM test real images (also ImageNet `val`) can overlap. This only affects the two test sets, not training.
* All real images use standard IJG quantisation tables, like Pillow; remaining encoder differences were not found but cannot be excluded.

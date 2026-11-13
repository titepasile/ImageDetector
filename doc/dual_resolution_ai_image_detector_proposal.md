# Dual-Resolution AI-Generated Image Detection

## Motivation

AI-generated images may exhibit evidence at different spatial scales. Global structure and semantic inconsistencies are easier to assess from the entire image, while local textures, noise statistics, and generation artifacts may be lost when an image is downsampled.

A second challenge is **generalization to unseen generators**. Training a detector end-to-end can encourage shortcuts tied to the training dataset—for example, resolution, compression, or correlations between the two input views—rather than transferable forensic features. Meanwhile, generic representation learning may suppress the subtle details needed for detection.

**Core hypothesis:** Separately pretrained global and local encoders, designed to preserve complementary information, can improve out-of-distribution detection compared with either encoder alone or purely end-to-end supervised training.

## Proposed Architecture

1. **Global branch:** Resize the *entire* input image to a fixed resolution (e.g., 256 × 256) and encode it using a pretrained vision model. This branch captures scene-level composition, structure, and semantic context.
2. **Local branch:** Extract a patch at the image's **native resolution** (e.g., 256 × 256 pixels when available) and pass it to a separate encoder. This branch targets fine-grained spatial and frequency-domain evidence without first downsampling the patch.
3. **Feature fusion:** Concatenate the two embeddings and train a lightweight classifier for real vs. AI-generated images. For multiple patches, first aggregate their embeddings (e.g., mean or attention pooling), then concatenate with the global embedding. Cross-attention is an optional later experiment, not a prerequisite.

## Training Strategy

- **Separate pretraining:** Use objectives suited to each branch. The global encoder can use semantic self-supervised representations (e.g., DINO-style learning); the local encoder can explore detail-preserving objectives such as masked spatial or spectral reconstruction. These are hypotheses to compare, not guaranteed improvements.
- **Controlled fine-tuning:** Start with frozen encoders and a trained fusion head, then compare selective or full fine-tuning. Optional auxiliary classification heads and branch dropout can discourage dependence on a single branch or a brittle relationship between embeddings.
- **Resolution-aware inputs:** If an image is smaller than the target patch (e.g., 128 × 128), do not treat an upscaled copy as new native-resolution evidence. Use smaller native-resolution crops with compatible encoder input handling, or explicitly handle the case where no useful distinct local view exists.
- **Dataset controls:** Balance or stratify real and generated samples by resolution, compression, and preprocessing wherever possible. Otherwise, the detector may learn source-specific shortcuts instead of generation evidence.

## Evaluation

Compare **global-only**, **local-only**, and **fused** models; frozen vs. fine-tuned encoders; and single vs. multiple local patches. Evaluate not only in-distribution accuracy but also on **held-out generator families** and under controlled resizing or recompression. A metadata-only baseline (resolution/compression) can help reveal dataset bias.

## Optional Extension: RL-Based Patch Selection

Replace random patch sampling with a policy that selects successive native-resolution regions based on the global embedding and previously observed evidence. Reward improvements in classification of the ground-truth label while penalizing additional crops. Compare against random and fixed-grid selection at the **same patch budget**. This extension tests whether adaptive sampling adds value without making RL a requirement of the core model.

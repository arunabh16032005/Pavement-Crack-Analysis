# Crack Segmentation, Severity Assessment, and PCI Framework
## Implementation Progress: Classical 50% Baseline and Roadmap to 100%

**Status:** Classical baseline completed and executed successfully on Kaggle.  
**Current scope:** Core data-to-segmentation-to-severity pipeline.  
**Full scope:** Quantum-enhanced, multi-domain segmentation; supervised severity assessment; and PCI/DI estimation.

---

## 1. Purpose of this document

This document records the implementation completed so far, the evidence produced by the completed Kaggle run, and the work required to reach the full architecture described in `architecture.md`.

The project is deliberately being built in two stages:

1. **Classical baseline (completed):** establish a reproducible, working reference pipeline using a ResNet34 U-Net and a conventional MLP severity model.
2. **Full framework (next):** add multi-domain features, VMamba/VM-UNet and quantum modules, real severity labels, PCI/DI estimation, and comprehensive cross-dataset validation.

This sequencing is important. A quantum or hybrid component should only be retained when it improves against the verified classical baseline under the same data splits and metrics.

---

## 2. Current end-to-end workflow

```text
CrackVision12K image + ground-truth mask
        |
        v
Dataset discovery, pairing, split organisation, and validation
        |
        v
Resize + augmentation + ImageNet normalisation
        |
        v
ResNet34 U-Net -> predicted binary crack mask
        |
        +------------------------------+
        |                              |
        v                              v
Segmentation evaluation       Structural descriptor extraction
                              (geometry, topology, network features)
                                             |
                                             v
                              Feature normalisation + 4-class MLP
                                             |
                                             v
                              Provisional severity prediction
```

The production-quality path will retain this overall flow. The remaining implementation replaces or extends selected blocks rather than discarding the working baseline.

---

## 3. Completed implementation

### 3.1 Project structure, configuration, and reproducibility

The repository has a modular structure with separate data, model, feature, training, evaluation, and visualisation components. `config.py` centralises the principal paths and hyperparameters, including the image size, augmentation parameters, ResNet encoder choice, loss weights, optimisation settings, class mapping, random seed, checkpoints, and result locations.

The Kaggle runner copies the source into `/kaggle/working/PROJ1`, extracts uploaded ZIP datasets into a writable location, checks package availability, and exports checkpoints, reports, and features at the end of a successful run. This makes Kaggle execution independent of local Windows file paths.

### 3.2 Data acquisition and organisation (Phase 1)

Implemented components include:

- Expected raw-data directory structure for CrackVision12K, CrackForest, DeepCrack, and Mendeley Pavement.
- Dataset verification and flexible directory detection in `src/data/organizer.py`.
- Image/mask pairing by filename stem, including masks with a different image extension.
- Support for CrackVision12K's pre-split `train`, `val`, and `test` structure.
- Support for CrackForest image files and MATLAB ground-truth files, including conversion of `.mat` masks to PNG.
- Support for DeepCrack test image/mask directories.
- Perceptual-hash utilities for train/test near-duplicate checks and a quarantine mechanism.
- A configurable severity-folder mapper for future Mendeley data ingestion.

For the Kaggle execution, CrackVision12K was attached as its own input dataset. The runner found the ZIP file, extracted it, and copied the supplied train/validation/test image-mask pairs into the processed structure. This is why the final report includes `val` and `test_cv12k` results.

### 3.3 Preprocessing and augmentation (Phase 2)

The preprocessing implementation provides the following building blocks:

- RGB conversion and spatial standardisation.
- Non-local-means denoising and CLAHE contrast enhancement functions for use in preprocessing/caching workflows.
- Binary-mask preparation with nearest-neighbour resizing.
- ImageNet normalisation for the pretrained ResNet encoder.
- Joint image/mask spatial augmentation during training: horizontal flip, vertical flip, 90-degree rotations, affine translation/scale/rotation, and elastic deformation.
- Image-only intensity augmentation: brightness/contrast adjustment and Gaussian noise.

An important Kaggle reliability correction was made: the active train, validation, and test transforms resize all samples to **512 x 512** before batching. This prevents PyTorch DataLoader failures when original images have different native dimensions. The augmentation implementation was also adjusted to work with current Albumentations releases.

### 3.4 Classical segmentation network (Phase 4 baseline)

The current segmentation model is `ResNetUNet` in `src/models/unet.py`.

- **Encoder:** ResNet34, optionally initialised from ImageNet weights.
- **Decoder:** four transpose-convolution decoder blocks with skip connections from ResNet stages.
- **Output:** one-channel per-pixel crack logit map; sigmoid is applied only when probabilities or masks are needed.
- **Robustness:** if pretrained weights cannot be downloaded because Kaggle Internet is disabled and no cached copy exists, the model falls back to random encoder initialisation rather than crashing.

This is intentionally the classical reference model. It produces a segmentation mask that can be evaluated directly and also supplies the visual structure required by the downstream descriptor and severity modules.

### 3.5 Segmentation objective, training, and checkpointing (Phase 8 baseline)

The segmentation loss combines three terms:

\[
L = 0.5L_{BCE} + 0.5L_{Dice} + 0.3L_{Boundary}.
\]

- **BCE with logits** gives stable pixel-level optimisation.
- **Soft Dice loss** addresses the strong crack/background class imbalance.
- **Boundary-aware BCE** gives additional weight to the dilated boundary ring around crack labels, encouraging sharper crack outlines.

The trainer uses AdamW, cosine learning-rate annealing, optional mixed precision, gradient accumulation, validation mIoU monitoring, early stopping, and best-checkpoint saving. The successful committed Kaggle run used a T4 x2 GPU and the safer DataLoader settings `num_workers=0` and `pin_memory=False` after the initial interactive session stalled with worker processes.

### 3.6 Structural descriptor extraction (Phase 5)

`StructuralDescriptorExtractor` converts a binary mask into an ordered numerical feature vector. It handles empty and very small masks safely. Extracted feature families include:

| Family | Examples | Method |
|---|---|---|
| Geometry | skeleton length, mean/max/std crack width, area, area ratio, aspect ratio | Skeletonisation and Euclidean distance transform |
| Topology | density, connected components, junctions, endpoints, fragmentation index | Connected-component and neighbourhood analysis |
| Orientation | 18-bin orientation histogram, dominant orientation, orientation entropy | Skeleton-gradient analysis |
| Network structure | graph nodes/edges, degree, path length, clustering coefficient | Skeleton-derived NetworkX graph |

The features are saved as NumPy `.npz` files together with feature names, making the severity model reproducible and inspectable.

### 3.7 Classical severity model (Phase 6 baseline)

The baseline severity model is a four-class MLP:

```text
Structural features
  -> Linear(128) -> BatchNorm -> ReLU -> Dropout
  -> Linear(64)  -> BatchNorm -> ReLU -> Dropout
  -> Linear(32)  -> ReLU
  -> Linear(4)   -> severity logits
```

Features are z-score normalised using statistics fitted only on the training set. The classifier uses weighted cross-entropy to reduce the effect of class imbalance.

**Important limitation:** the completed Kaggle run did **not** include the Mendeley Pavement severity-labelled dataset. It therefore used the implemented fallback that derives **pseudo-severity labels** from the same structural descriptors using density, width, component count, and junction-count rules. Consequently, the current severity score demonstrates that this fallback is internally learnable; it is **not** a valid estimate of real-world, human-labelled severity accuracy. Replacing pseudo-labels with Mendeley's real labels is a required next step.

### 3.8 Evaluation and reporting

The evaluation module computes pixel accuracy, crack IoU, mean IoU, precision, recall, and F1. It can evaluate every available dataset loader, save predicted masks, generate training curves, and create a Markdown report.

The Kaggle runner also safely handles absent optional datasets. CrackForest and DeepCrack were not attached to this execution, so they were not included in the reported metrics.

---

## 4. Results from the completed Kaggle run

The generated `evaluation_report.md` reported:

| Dataset | mIoU | F1 | Pixel accuracy | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| CrackVision12K validation split (`val`) | 0.7271 | 0.5844 | 0.9734 | 0.6186 | 0.5878 |
| CrackVision12K supplied test split (`test_cv12k`) | 0.7274 | 0.5847 | 0.9730 | 0.6205 | 0.5918 |

The segmentation objective of at least **0.70 mIoU** was met on both available CrackVision12K splits.

The reported severity value was:

| Measure | Value | Interpretation |
|---|---:|---|
| Best severity F1 | 0.9309 | Provisional only: evaluated against rule-derived pseudo-labels, not Mendeley human labels |

### Interpretation

- The close validation and supplied-test mIoU values (0.7271 and 0.7274) suggest consistent performance within CrackVision12K's provided splits.
- Pixel accuracy is high because background pixels dominate crack images. For a thin-crack segmentation task, mIoU, crack IoU, F1, precision, and recall are more informative than pixel accuracy alone.
- Precision is moderately higher than recall. This indicates the model misses some crack pixels; future work should examine threshold calibration, false-negative overlays, boundary quality, and loss weighting.
- These results are a valid **classical baseline**, not evidence that the proposed quantum modules improve performance. A controlled ablation is needed for that claim.

---

## 5. What is implemented versus what remains

| Architecture phase | Current state | Evidence / next action |
|---|---|---|
| 1. Data acquisition and organisation | Implemented | CrackVision12K exercised; attach and validate CrackForest, DeepCrack, and Mendeley next |
| 2. Preprocessing and augmentation | Implemented | Active 512 x 512 batching and augmentation used in Kaggle run |
| 3. Multi-domain feature extraction and quantum encoding | Deferred | Add handcrafted feature branches and quantum encoding/fusion |
| 4. Segmentation network | Classical baseline implemented | ResNet34 U-Net completed; replace/augment encoder with VMamba/VM-UNet in an ablation |
| 5. Structural descriptors | Implemented | Geometry, topology, orientation, and graph features available |
| 6. Severity prediction | Baseline implemented | Replace pseudo-label training with Mendeley labels; then introduce hybrid/quantum classifier |
| 7. PCI/DI assessment | Deferred | Define calibrated standard-compliant mapping and validate on labelled survey data |
| 8. Training and evaluation | Baseline implemented | Extend to multi-task/hybrid objectives, calibration, and cross-dataset experiments |

---

## 6. Roadmap from the 50% baseline to the full 100% framework

### Stage A — Complete data and scientific validation

1. **Attach and organise CrackForest and DeepCrack.** Run the existing organiser and evaluate the current ResNet34 U-Net without retraining on those datasets first. Report domain gaps against CrackVision12K validation performance.
2. **Attach Mendeley Pavement.** Inspect its folder names, record the mapping to Low/Moderate/Severe/Critical, and create class-balanced train/validation splits using `SeverityMapper`.
3. **Eliminate pseudo-label evaluation.** Segment Mendeley images using the best segmentation checkpoint, extract descriptors from predicted masks, and train/test the severity MLP against real Mendeley labels. Retain the pseudo-label model only as a baseline, never as the primary severity claim.
4. **Audit data leakage and duplicates.** Run perceptual-hash checks between the segmentation training data and every external test set. Quarantine duplicates before reporting cross-dataset results.
5. **Add qualitative evaluation.** Save overlays for true positives, false positives, false negatives, thin cracks, shadows, texture, and complex crack networks.

### Stage B — Add the multi-domain feature layer (Phase 3)

Implement each feature family as an independently testable module:

- **Frangi vesselness:** enhance elongated crack-like structures across scales.
- **Gabor filter bank:** represent directional texture and orientation.
- **Wavelet features:** capture multi-resolution crack edges and fine detail.
- **Learned state-space features:** introduce a VMamba/VM-UNet encoder branch for long-range spatial context.

Start with a classical fusion mechanism such as channel concatenation followed by 1 x 1 convolution or attention. Train and compare one addition at a time against the ResNet34 U-Net baseline. Track mIoU, crack F1, recall, compute cost, and external-domain generalisation. Do not add a quantum layer until the classical feature fusion behaviour is understood.

### Stage C — Upgrade the segmentation model (Phase 4 full)

1. Implement the VMamba/VM-UNet backbone behind the same segmentation-model interface: input tensor -> raw one-channel logits.
2. Reuse the current data loaders, loss, trainer, checkpoint format where possible, so comparisons remain fair.
3. Train the following controlled experiments with fixed splits and seeds:

   - ResNet34 U-Net baseline (already available)
   - VM-UNet without quantum modules
   - VM-UNet plus classical multi-domain fusion
   - VM-UNet plus the proposed quantum enhancement

4. Compare accuracy, inference latency, VRAM use, parameter count, and external-test mIoU. Adopt a more complex model only when it offers a measurable benefit.

### Stage D — Add quantum encoding and hybrid modules

The quantum components should be introduced as replaceable, tested modules rather than embedded throughout the pipeline.

1. **Feature reduction:** reduce learned/fused feature vectors to a small fixed dimension suitable for a limited number of qubits.
2. **Encoding:** implement the planned angle/amplitude encoding with explicit normalisation and numerical checks.
3. **Variational circuit:** add a parameterised circuit with a documented qubit count, layers, entanglement pattern, measurements, and classical-to-quantum gradient interface.
4. **Integration:** inject quantum outputs through residual gating or feature modulation, preserving a classical bypass mode.
5. **Training policy:** begin with differentiable simulation on small feature vectors. Use hardware runs only for final demonstration/inference experiments, with shot count, backend, noise assumptions, and reproducibility recorded.
6. **Ablation:** report a classical feature-reduction control model with an equivalent parameter budget. The quantum branch must be compared to this control, not merely to a weaker baseline.

### Stage E — Complete supervised severity assessment (Phase 6 full)

1. Use the segmentation model's **predicted** masks—not ground-truth masks—for the operational severity pipeline.
2. Train the current MLP on Mendeley labels as the supervised classical reference.
3. Add calibration: class-wise precision/recall, macro F1, balanced accuracy, confusion matrix, reliability curves, and confidence thresholds.
4. Implement the planned quantum neural-network classifier as an optional replacement for the MLP head, using the same normalised descriptor vector and exactly the same data splits.
5. Investigate joint learning only after a strong two-stage baseline exists. A multi-task setup can share an encoder while optimising segmentation loss and severity loss, but must avoid severity labels being assigned to incorrectly segmented regions.

### Stage F — Implement PCI/DI assessment (Phase 7)

PCI/DI cannot be justified by a simple direct mapping from a four-class severity label. The implementation should:

1. Define the target pavement-assessment standard and the survey-unit assumptions used by the project.
2. Convert the segmentation and descriptor outputs into calibrated distress quantities: crack density/area, length, width, crack type, and severity.
3. Implement a transparent deduction-value calculation or a learned calibration model tied to a labelled PCI/DI reference dataset.
4. Aggregate deduct values at the survey-unit level and apply the selected standard's correction procedure.
5. Validate predicted PCI/DI against expert or ground-truth survey values, reporting MAE, RMSE, correlation, and error by condition range.

The document, code, and final report must state clearly whether the final value is a standard-compliant PCI, a proxy index, or a model-estimated condition score.

### Stage G — Final evaluation, reproducibility, and reporting

- Evaluate all model variants on CrackVision12K, CrackForest, and DeepCrack with no test-set tuning.
- Report mean and variation over multiple seeds where compute permits.
- Publish exact dataset versions, splits, duplicate-handling outcomes, hyperparameters, package versions, hardware, training time, and checkpoint selection rule.
- Produce qualitative predictions and failure cases alongside aggregate metrics.
- Compare each full-system component through an ablation table.
- Package the final inference workflow: image -> segmentation overlay -> descriptors -> severity probabilities -> calibrated PCI/DI result with confidence/limitations.

---

## 7. Immediate next implementation milestone

The highest-value next milestone is **supervised, cross-dataset validation**:

1. Add CrackForest, DeepCrack, and Mendeley Pavement to Kaggle.
2. Run the existing pipeline without changing the ResNet34 U-Net baseline.
3. Produce external-test segmentation metrics for CrackForest and DeepCrack.
4. Replace pseudo-severity labels with Mendeley labels and report macro F1 plus a confusion matrix.
5. Use these results as the locked baseline for the VMamba and quantum ablation experiments.

This establishes whether the current system generalises, makes the severity result scientifically meaningful, and gives the full architecture a fair benchmark to beat.

---

## 8. Current deliverables

- Source implementation under `src/`.
- Central configuration in `config.py`.
- Kaggle execution runner in `kaggle_notebook.py`.
- Best segmentation and severity checkpoints generated by the completed run.
- Structural feature archives and normalisation parameters.
- Training curves and evaluation report generated in Kaggle output.
- This implementation-status and roadmap document.

The project has therefore completed its intended classical functional baseline. The remaining work is not a rewrite; it is a measured expansion and validation of this baseline into the full multi-domain, hybrid quantum, severity-supervised, PCI/DI-capable system.

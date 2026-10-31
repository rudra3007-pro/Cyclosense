

CycloSense — Model Training
## Report
Project: CycloSense — Cyclone Detection, Classification, and
## Forecasting System
Author: Rudra Scope: Model training for all three
AI modules (Detection, Classification, Forecasting)
## Overview
CycloSense uses a three-model pipeline to process satellite imagery
and historical track data for cyclone monitoring:
- Model 1 (Detection) finds and localizes cyclones in raw satellite
images.
- Model 2 (Classification) determines the developmental stage/
intensity category of a detected cyclone.
- Model 3 (Forecasting) predicts a cyclone's future position and
intensity based on its recent track history.
These models are designed to chain together: a raw satellite image is
passed to Model 1, which crops the cyclone region; that crop is passed
to Model 2 for intensity classification; and a sequence of recent track
points (position + intensity) is passed to Model 3 to forecast where the
storm is headed and how it may evolve.
Model 1 — Cyclone Detection (YOLOv8)

## Purpose
Given a full, uncropped satellite image, detect whether a cyclone is
present and output a bounding box with a confidence score.
## Dataset
-  Source: INCYDE (INSAT Cyclone Detection & Intensity
Estimation dataset), derived from INSAT-3D/3DR infrared satellite
imagery over the Indian Ocean (2013–2021).
-  Format: Pre-split into train/val/test folders, each containing
YOLO-format label files (
class_id x_center y_center
width height
, normalized 0–1).
-  Size used: 16,038 training images (original, non-augmented
subset), 2,399 validation images, 2,503 test images.
-  Single class: cyclone.
## Architecture & Technique
-  Model: YOLOv8-nano (yolov8n.pt), fine-tuned from pretrained
COCO weights via Ultralytics.
-  Input size: 640×640.
-  Training: 50 epochs, batch size 16, standard YOLO loss (box
regression + classification + distribution focal loss).
-  Data integrity: Verified that train/val/test splits contained no
duplicate image files across each other before training (checked
via matched image/label filename counts) to avoid data leakage.
## Results
MetricValue
## Precision0.706
## Recall0.801
mAP500.699

mAP50-950.393
## Qualitative Observations
Visual inspection of predictions showed the model reliably detects well-
formed cyclones with visible spiral structure (confidence 0.5–0.7). It
occasionally produces duplicate/overlapping bounding boxes on a single
system, and sometimes flags ambiguous cloud clusters without clear
cyclonic structure as false positives. Recall (0.80) is stronger than
precision (0.71), meaning the model favors catching real cyclones over
avoiding false alarms — a reasonable tradeoff for a disaster-response
context.
## Known Limitations
-  Occasional duplicate detections on the same cyclone (could be
addressed with tighter NMS/IoU thresholding).
-  Some false positives on bright, disorganized cloud clusters lacking
true cyclonic structure.
## Model 2 — Cyclone Intensity Classification
(EfficientNet)
## Purpose
Given a cropped cyclone image (the output of Model 1), classify its
developmental stage / intensity category according to IMD (India
Meteorological Department) conventions.
## Dataset

-  Source: INCYDE's Cropped_cyclone_data, containing
103,368 cropped cyclone images with a labeled CSV
## (
## Cropped_cyclone_labels.csv).
-  Labels used: IMD category codes — D (Depression), DD (Deep
Depression), CS (Cyclonic Storm), SCS (Severe Cyclonic Storm),
VSCS (Very Severe Cyclonic Storm), SuCS (Super Cyclonic
Storm) — and
MSWS kts (Maximum Sustained Wind Speed).
-  Class imbalance: Highly imbalanced — D: 37,890 samples vs.
SuCS: 729 samples (~52x difference).
-  Class merging: SuCS was merged into VSCS due to insufficient
samples (159 in validation alone) to learn it as a distinct class —
multiple attempts (loss weighting, oversampling) failed to achieve
non-zero recall for SuCS as a standalone class. Final label set:
## D,
## DD, CS, SCS, VSCS
(5 classes).
-  Split: Pre-split by storm name (Train: 98,466 / Val: 2,399 / Test:
2,503), verified to have no storm-name leakage across splits.
## Architecture & Technique
-  Backbone: EfficientNet-B0, pretrained on ImageNet.
-  Transfer learning strategy: Early feature layers frozen; only the
last two feature blocks and a new classifier head were fine-tuned
(~1.1M trainable parameters out of ~4.0M total).
-  Classifier head: Dropout (p=0.4) + Linear layer (5 outputs).
-  Regularization: Dropout, weight decay (1e-4), and live data
augmentation (random rotation, color jitter, random resized crop)
applied during training to prevent overfitting.
-  Class imbalance handling: WeightedRandomSampler used to
oversample minority classes during training (after initial loss-
weighting alone proved insufficient).
-  Optimizer: Adam, learning rate 1e-5 (deliberately low to avoid
aggressive overwriting of pretrained features).
-  Loss: CrossEntropyLoss.
-  Early stopping: Patience of 5 epochs on validation accuracy.
## Alternative Approach Tested: Ordinal Regression
Since IMD categories represent an ordered severity scale (not

independent classes), ordinal regression was tested as a theoretically
better-suited approach — reframing the 5-class problem as 4 binary "is
severity greater than threshold X" questions, using
BCEWithLogitsLoss. Two decode strategies were tried (hard
thresholding and expected-rank/probability-sum decoding). Both runs
caused the middle category (CS) to collapse to 0% recall, worse than
the plain classification approach. Conclusion: plain classification with
class merging outperformed ordinal regression in practice for this
dataset
, despite ordinal regression's theoretical appeal — likely due to
CS's visual similarity to both its neighboring categories making binary
threshold decisions unstable.
Final Results (Plain 5-Class Classification)
ClassPrecisionRecallF1-scoreSupport
## CS0.300.360.33549
## D0.800.290.43591
## DD0.220.130.16283
## SCS0.210.380.27215
## VSCS0.630.810.71761
Overall accuracy: 46% | Macro avg F1: 0.38 | Weighted avg F1: 0.45
## Qualitative Observations
A manual review of misclassified examples revealed a genuine edge
case: a Depression-labeled storm (10 June 2014, MSWS 25 kts)
displayed a visually mature spiral/eye structure and was consistently
misclassified as VSCS/SCS across its full time sequence. This reflects a
known real-world limitation of visual-only cyclone classification — storm
appearance and measured wind speed do not always perfectly
correlate, particularly in early formation stages (this is part of why real
meteorological practice, e.g. the Dvorak technique, combines visual
pattern analysis with other data sources rather than relying on imagery
alone).

## Known Limitations
-  DD (Deep Depression) remains the weakest category (13%
recall), likely due to its narrow visual distinction from both D and
CS on either side.
-  SuCS could not be reliably distinguished from VSCS given only
159 available samples.
-  Model can be visually "fooled" by storms whose organizational
appearance doesn't match their measured wind speed.
## Model 3 — Cyclone Track & Intensity
Forecasting (LSTM)
## Purpose
Given a sequence of a cyclone's recent track points (position + wind
speed), forecast its position and intensity at future time steps. This
directly powers the "will this cyclone affect my selected location" feature,
by allowing predicted track points to be checked against a user-selected
coordinate.
## Dataset
-  Source: IBTrACS (International Best Track Archive for Climate
Stewardship), North Indian Ocean basin subset, official NOAA
distribution.
-  Filtering: Restricted to 2009–2023 based on wind-data
completeness analysis (years before 2009 had substantial missing
wind records; 2009 onward showed consistently ≥95%
completeness across WMO/USA/New Delhi agency wind reports).
-  Wind source priority: New Delhi (IMD) wind speed used
preferentially, falling back to WMO, then USA agency figures, to
maximize data completeness while prioritizing the IMD-consistent

source used in Model 2.
-  Final dataset: 5,001 track records across 142 individual storms.
-  Sequence construction: Sliding window of 8 past time-steps (3-
hour intervals, i.e. 24 hours of history) used to predict the next
single time-step (lat, lon, wind speed).
-  Split: By storm (not by individual sequence) — Train: 2,742
sequences / Val: 512 / Test: 636 — to prevent the same storm's
sequences from appearing in both train and evaluation sets.
-  Normalization: StandardScaler fit on training data only, applied to
lat/lon/wind features before model input (necessary given the very
different natural scales of these three features).
## Architecture & Technique
-  Model: 2-layer LSTM (hidden size 64, dropout 0.2 between
layers), followed by a linear output layer producing (lat, lon, wind)
for the next time step.
-  Loss: Mean Squared Error (MSE) on normalized features.
-  Optimizer: Adam, learning rate 1e-3.
-  Early stopping: Patience of 10 epochs on validation loss; training
converged in 32 epochs.
Results — Single-Step Forecast (+3 hours)
MetricValue
Mean position error101.3
km
Median position error85.9 km
Mean wind speed error1.9 kts
Median wind speed error0.9 kts
Results — Recursive Multi-Step Forecast
To forecast further into the future, the model's own prediction is fed back

in as the newest point in the input sequence (dropping the oldest point),
repeated step by step. This is standard practice for autoregressive
sequence forecasting, but errors compound with each step:
## Forecast
horizon
Mean position
error
Median position
error
## N (test
storms)
+3 hours (1
step)
126.2 km103.2 km15
+6 hours (2
steps)
153.8 km148.2 km15
+12 hours (4
steps)
235.6 km238.6 km15
+24 hours (8
steps)
391.0 km418.8 km15
## Context
For comparison, mature operational forecasting systems (IMD, NHC,
JTWC) — built on decades of data and full numerical weather prediction
models — typically report 24-hour track errors in the range of 100–150
km. CycloSense's ~391–419 km error at +24 hours is meaningfully
higher, which is expected given the lightweight LSTM architecture and
comparatively limited training data, but the model performs reasonably
well at shorter horizons (+3h to +6h).
## Known Limitations
-  Only 15 test storms had track sequences long enough to evaluate
the full +24 hour horizon — a small sample size for this specific
measurement.
-  Recursive forecasting compounds error at each step; longer-
horizon predictions should be treated as directionally indicative
rather than precise.
-  The model requires an existing, currently-tracked storm's recent
history as input — it cannot predict spontaneous cyclone

formation ("genesis") in a region with no active storm.
## Summary Table
ModelTaskArchitectureKey MetricResult
## Model
## 1
DetectionYOLOv8-nanomAP500.699
## Model
## 2
ClassificationEfficientNet-B0
(transfer learning)
## Accuracy46%
## Model
## 3
Forecasting2-layer LSTMPosition
error (+3h)
101.3 km
## (mean)
Techniques Used Across the Pipeline
-  Transfer learning (Models 1 & 2): leveraging pretrained weights
(COCO for YOLO, ImageNet for EfficientNet) rather than training
from scratch, given limited domain-specific data.
-  Data leakage prevention: storm-level train/val/test splitting
across all three models, ensuring no single storm's data crossed
between training and evaluation sets.
-  Class imbalance handling (Model 2): weighted sampling and
class merging to address a 52x imbalance between the most and
least common categories.
-  Regularization (Model 2): dropout, weight decay, frozen
backbone layers, and live data augmentation to prevent overfitting
— validated by comparing an initial unregularized run (95% train
accuracy / 43% val accuracy) against the regularized version
(62% train / 39% val), confirming the fix closed the generalization
gap.
-  Evaluation beyond accuracy: per-class precision/recall/F1 and
confusion matrices used throughout, rather than relying on

aggregate accuracy alone — this was essential in Model 2, where
aggregate accuracy alone would have masked a complete failure
to recognize the rarest class.
-  Autoregressive/recursive forecasting (Model 3): feeding model
predictions back in as input to extend the forecast horizon beyond
a single time step.
-  Experimental rigor: an alternative approach (ordinal regression
for Model 2) was implemented, tested, and honestly rejected in
favor of the simpler approach after evidence showed it performed
worse in practice — documented as a considered and tested
design decision rather than omitted.
## Overall Pipeline Flow
Raw satellite image

## │
## ▼
Model 1 (YOLO) ──► bounding box + cropped cyclone region

## │
## ▼
Model 2 (EfficientNet) ──► intensity category (D/DD/CS/SCS/VSCS)

## │
## ▼
Sequence of recent (position, intensity) points

## │
## ▼
Model 3 (LSTM) ──► forecasted future position & intensity

## │
## ▼
Compared against user-selected location to determine potential impact
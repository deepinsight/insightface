# Commercial Evaluation Mode

Enterprise Evaluation Mode helps teams evaluate InsightFace locally with their
own data before choosing a commercial model license, private model evaluation,
SDK/API access, SLA, or custom training path.

All processing is local by default. No images, embeddings, videos, or reports
are uploaded automatically.

## 1:1 Verification

Select **1:1 Verification**, choose an identity-folder root, and decide whether
to enable **Auto Split**.

Identity folders use one subfolder per person:

```text
dataset_1v1/
  0001__Alice/
    gallery.jpg
    img002.jpg
    img003.jpg
  0002__Bob/
    img001.jpg
    img002.jpg
```

With **Auto Split**, a file containing `gallery` is used as that identity's
gallery image. If no such file exists, the first sorted image is used. All
other images become probes, and every comparison is probe vs gallery across
identities. Without Auto Split, every image is treated as a probe and the page
runs full pairwise probe-vs-probe comparisons.

The evaluation reports the best cosine threshold accuracy, the selected
threshold, positive and negative pair counts, and TAR@FAR for `1e-6`, `1e-5`,
`1e-4`, and `1e-3` with corresponding thresholds.

## Dataset Validation And Multi-face Policy

Run **Validate Dataset** before every evaluation. The GUI checks folder layout,
Auto Split rules, gallery/probe availability, generated 1:1 positive and
negative pairs, 1:N gallery coverage, image readability, and detected face
counts.

The **Multi-face handling** option controls images with more than one detected
face:

- **Require exactly one face** is the default and blocks the run when any image
  contains multiple faces.
- **Use largest face** keeps the image and uses the largest detected face.
- **Use largest centered face** keeps the image and selects the face with the
  best area-minus-center-distance score, favoring large faces near the image
  center.
- **Mark as skip** skips multi-face images. If a required gallery image is
  skipped, validation fails because the gallery sample is unavailable.

## 1:N Identification

Without **Auto Split**, prepare:

```text
dataset_1n/
  gallery/
    0001__Alice/
      enroll_001.jpg
    0002__Bob/
      enroll_001.jpg
  probe/
    0001__Alice/
      test_001.jpg
    0002__Bob/
      test_001.jpg
  unknown/
    unknown_001.jpg
```

With **Auto Split**, prepare:

```text
dataset/
  identities/
    0001__Alice/
      img001.jpg
      img002.jpg
    0002__Bob/
      img001.jpg
      img002.jpg
```

1:N evaluation always requires gallery images. The report includes Top1 and
TAR@FAR for `1e-5`, `1e-4`, `1e-3`, and `1e-2`, with corresponding thresholds.

## Interpreting Metrics

- FAR: false accept rate. Lower means fewer different-person pairs accepted.
- FRR: false reject rate. Lower means fewer same-person pairs rejected.
- TAR: true accept rate at a target FAR.
- Top1: correct person is the first search result.

Thresholds are business decisions. Higher thresholds usually reduce false
accepts but increase false rejects.

## Exporting Reports

Reports are exported automatically after a run. Click **Export PDF** to choose
a PDF destination. Reports include:

1. Executive Summary
2. Evaluation Scenario
3. Dataset Summary
4. Model and Runtime
5. License Status
6. Metrics
7. Threshold Recommendation
8. Error Analysis
9. Latency and Hardware
10. Deployment Considerations
11. Responsible Use and Compliance Notice
12. Commercial Licensing Next Steps
13. Appendix: Raw Results

Markdown, HTML, and PDF are supported when the GUI extra is installed.

## Commercial License Notice

This evaluation may use research or non-commercial model files. Production or
commercial deployment requires an appropriate commercial model license. Please
contact InsightFace for commercial model licensing, private model evaluation,
SDK/API access, SLA, or custom training.

This report does not provide legal advice. Users are responsible for consent,
privacy, retention, and compliance with applicable biometric regulations.

# Dataset Layout

This repository does not store datasets.

Minimum expected layout:

```text
data/
  ssl/
    images/
      image_001.jpg
      image_002.jpg
  detection/
    images/
      train/
      val/
    labels/
      train/
      val/
  detection.yaml
```

Example `data/detection.yaml`:

```yaml
path: data/detection
train: images/train
val: images/val
names:
  0: object
```

Self-supervised pretraining uses only `ssl.images` and ignores labels. Detection fine-tuning uses standard YOLO labels.


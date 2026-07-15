# YOLOv8 SSL Experiment

Eksperimen ini membandingkan tiga inisialisasi YOLOv8n untuk object detection ketika anotasi terbatas:

1. YOLOv8n pretrained standar Ultralytics.
2. YOLOv8n random initialization.
3. YOLOv8n dengan backbone yang dipra-latih memakai self-supervised SimCLR, lalu di-fine-tune secara supervised.

Kode ini tidak mengklaim SSL pasti lebih baik. Tujuannya membuat eksperimen yang bisa diuji secara adil.

## Konsep Singkat

Supervised learning memakai label target, misalnya bounding box dan class. Self-supervised learning membuat sinyal latih dari data itu sendiri; di sini dua augmentasi dari gambar yang sama menjadi positive pair, sedangkan gambar lain dalam batch menjadi negative pair. Semi-supervised learning memakai kombinasi label asli dan data tidak berlabel, sering dengan pseudo-labeling. Proyek ini bukan semi-supervised karena tidak memakai pseudo-labeling.

## Pipeline

```text
image
  -> augmentation A and B
  -> YOLOv8 backbone
  -> global average pooling
  -> projection head
  -> NT-Xent loss
  -> save backbone only
  -> transfer compatible tensors into YOLOv8 detector
  -> supervised detection fine-tuning
```

Projection head hanya dipakai saat pretraining SSL dan tidak ditransfer ke detector.

## Struktur Dataset

```text
data/
  ssl/
    images/
  detection/
    images/
      train/
      val/
    labels/
      train/
      val/
  detection.yaml
```

Self-supervised pretraining hanya membaca folder gambar. Fine-tuning memakai format YOLO.

## Instalasi

Local CPU:

```bash
pip install -r requirements.txt
```

Kaggle atau Colab biasanya sudah menyediakan PyTorch. Jangan reinstall PyTorch otomatis; install paket lain saja:

```bash
pip install -q "ultralytics>=8.2,<9.0" "pyyaml>=6.0,<7.0" "tqdm>=4.66,<5.0" "tensorboard>=2.15,<3.0" "pandas>=2.0,<3.0" "matplotlib>=3.8,<4.0" "pytest>=8.0,<9.0" "pillow>=10.0,<12.0"
```

## Konfigurasi Environment

Config utama ada di `configs/experiment.yaml`.

Override tersedia:

```text
configs/envs/local_cpu.yaml
configs/envs/kaggle_gpu.yaml
configs/envs/colab_gpu.yaml
```

Kaggle mendukung `/kaggle/input` dan `/kaggle/working`. Colab mendukung `/content` dan `/content/drive/MyDrive`.

## SSL Pretraining

```bash
python -m src.pretrain_ssl --config configs/experiment.yaml --env-config configs/envs/local_cpu.yaml
```

Output:

```text
outputs/
  ssl/
    checkpoints/
      best.pt
      last.pt
      backbone_best.pt
    logs/
    metrics.json
```

Resume:

```yaml
ssl:
  resume: outputs/ssl/checkpoints/last.pt
```

## Detection Fine-tuning

```bash
python -m src.finetune_detection \
  --config configs/experiment.yaml \
  --env-config configs/envs/local_cpu.yaml \
  --backbone-checkpoint outputs/ssl/checkpoints/backbone_best.pt
```

Transfer bobot mencocokkan nama parameter dan shape tensor. Laporan disimpan ke `weight_transfer_report.json`. Tensor yang tidak cocok tidak dimuat diam-diam; semuanya dicatat.

## Baseline

Pretrained dan random initialization:

```bash
python -m src.run_baselines --config configs/experiment.yaml --env-config configs/envs/local_cpu.yaml
```

Semua fraction plus SSL backbone:

```bash
python -m src.run_baselines \
  --config configs/experiment.yaml \
  --env-config configs/envs/local_cpu.yaml \
  --include-ssl \
  --backbone-checkpoint outputs/ssl/checkpoints/backbone_best.pt
```

## Labeled Fraction

Edit:

```yaml
detection:
  labeled_fraction: 0.1
  fractions: [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
```

Subset train dibuat reproducible dan validation set tidak diubah:

```text
outputs/subsets/train_10_percent.txt
```

## Compare Results

```bash
python -m src.compare_results --output-dir outputs
```

Output:

```text
outputs/comparison.csv
outputs/comparison.json
outputs/comparison.md
outputs/mAP50_vs_fraction.png
outputs/mAP50_95_vs_fraction.png
outputs/precision_vs_fraction.png
outputs/recall_vs_fraction.png
outputs/ssl_training_loss.png
outputs/training_time_by_experiment.png
```

## Test dan Smoke Test

Jalankan test:

```bash
pytest
```

Smoke test lokal CPU memakai config:

```bash
python -m src.pretrain_ssl --config configs/experiment.yaml --env-config configs/envs/local_cpu.yaml
```

Untuk smoke detection, siapkan dataset YOLO kecil lalu:

```bash
python -m src.finetune_detection \
  --config configs/experiment.yaml \
  --env-config configs/envs/local_cpu.yaml \
  --backbone-checkpoint outputs/ssl/checkpoints/backbone_best.pt
```

## Logging

Kode memakai Python logging. Informasi minimum yang dicatat: device, jumlah gambar, batch size, parameter trainable, learning rate, checkpoint path, laporan transfer bobot, dan metrik evaluasi.

## Batasan

- Backbone split mengikuti metadata `model.yaml['backbone']` dari Ultralytics. Jika Ultralytics mengubah struktur internal secara besar, transfer bisa perlu penyesuaian.
- SimCLR tidak memakai bounding box. Jika dataset SSL terlalu berbeda dari dataset detection, hasil bisa tidak membantu.
- Batch kecil di CPU hanya cocok untuk smoke test, bukan kesimpulan riset.
- Hasil eksperimen baru sah setelah semua baseline dan SSL berjalan dengan fraction, validation set, seed, epoch, dan image size yang sama.

## Contoh Output Terminal

```text
INFO | Device: cuda
INFO | SSL images: 12000
INFO | SSL batch size: 64
INFO | Transferred 72 tensors; skipped 0 tensors.
INFO | Saved detection metrics: outputs/detection/ssl_backbone/metrics.json
```

## Contoh Tabel Hasil

| experiment | labeled_fraction | initialization | precision | recall | mAP50 | mAP50-95 | best_epoch | training_time |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_pretrained_10pct | 0.1 | pretrained | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| baseline_random_10pct | 0.1 | random | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| ssl_backbone_10pct | 0.1 | ssl | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |


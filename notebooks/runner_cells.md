# Kaggle / Colab Runner Cells

Use these cells as a lightweight notebook runner. Do not reinstall PyTorch on Kaggle or Colab unless you intentionally need a different runtime.

## 1. Install dependencies

Kaggle or Colab:

```bash
pip install -q "ultralytics>=8.2,<9.0" "pyyaml>=6.0,<7.0" "tqdm>=4.66,<5.0" "tensorboard>=2.15,<3.0" "pandas>=2.0,<3.0" "matplotlib>=3.8,<4.0" "pytest>=8.0,<9.0" "pillow>=10.0,<12.0"
```

Local CPU environment:

```bash
pip install -r requirements.txt
```

## 2. Validate GPU

```python
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
```

## 3. Clone or upload repository

```bash
git clone YOUR_REPOSITORY_URL yolov8_ssl_experiment
cd yolov8_ssl_experiment
```

If you uploaded the folder manually, just `%cd` into that folder.

## 4. Configure dataset

Kaggle uses `/kaggle/input` for read-only datasets and `/kaggle/working` for outputs. Edit `configs/envs/kaggle_gpu.yaml`:

```yaml
ssl:
  image_dir: /kaggle/input/YOUR_SSL_DATASET/images
detection:
  dataset_yaml: /kaggle/input/YOUR_DETECTION_DATASET/detection.yaml
output_dir: /kaggle/working/yolov8_ssl_outputs
```

Colab usually uses `/content` and optionally Google Drive under `/content/drive/MyDrive`. Edit `configs/envs/colab_gpu.yaml` similarly.

## 5. Run SSL pretraining

```bash
python -m src.pretrain_ssl --config configs/experiment.yaml --env-config configs/envs/kaggle_gpu.yaml
```

For Colab:

```bash
python -m src.pretrain_ssl --config configs/experiment.yaml --env-config configs/envs/colab_gpu.yaml
```

## 6. Supervised fine-tuning with SSL backbone

```bash
python -m src.finetune_detection \
  --config configs/experiment.yaml \
  --env-config configs/envs/kaggle_gpu.yaml \
  --backbone-checkpoint /kaggle/working/yolov8_ssl_outputs/ssl/checkpoints/backbone_best.pt
```

## 7. Baselines

```bash
python -m src.run_baselines \
  --config configs/experiment.yaml \
  --env-config configs/envs/kaggle_gpu.yaml
```

Run baselines plus SSL experiment for all fractions:

```bash
python -m src.run_baselines \
  --config configs/experiment.yaml \
  --env-config configs/envs/kaggle_gpu.yaml \
  --include-ssl \
  --backbone-checkpoint /kaggle/working/yolov8_ssl_outputs/ssl/checkpoints/backbone_best.pt
```

## 8. Compare results

```bash
python -m src.compare_results --output-dir /kaggle/working/yolov8_ssl_outputs
```

## 9. Save checkpoints

Kaggle outputs are already in `/kaggle/working/yolov8_ssl_outputs`. In Colab, copy outputs to Drive:

```bash
cp -r /content/yolov8_ssl_outputs /content/drive/MyDrive/
```


 # Sugarcane Leaf Disease Detection

Comparative evaluation of six deep learning architectures for sugarcane leaf disease classification on the Daphal & Koli (2022) field-collected dataset. Evaluates three transformer variants (Swin-T, ViT-B/16, DeiT-S), two convolutional networks (ResNet-50, EfficientNet-B3) and a 3D CNN reference under a unified five-fold stratified cross-validation protocol.

## Dataset

The Sugarcane Leaf Disease Dataset is hosted on [Mendeley Data](https://data.mendeley.com/datasets/9424skmnrk/1) under DOI `10.17632/9424skmnrk.1`. It contains 2,521 RGB images collected with smartphone cameras under field conditions in Maharashtra, India, distributed across five classes (Healthy, Mosaic, RedRot, Rust, Yellow) with resolutions ranging from 240×292 to 1600×1600 pixels.

After downloading, organize the dataset as follows.

```
data/
├── Healthy/
├── Mosaic/
├── RedRot/
├── Rust/
└── Yellow/
```

## Installation

Python 3.10 or higher and a CUDA-capable GPU are required. Tested on NVIDIA Tesla T4 with 15.6 GB VRAM.

```bash
git clone https://github.com/iamRabia-N/sugarcane-leaf-disease-detection.git
cd sugarcane-leaf-disease-detection
pip install -r requirements.txt
```

## Usage

### Training

Train a single architecture across all five folds.

```bash
python train.py --model SwinT --data_root ./data --output_dir ./checkpoints
```

Supported models are `SwinT`, `ViT`, `DeiT`, `ResNet50`, `EfficientNetB3` and `CNN3D`.

Each run produces one best checkpoint per fold (`{model}_fold{0..4}_best.pth`) and a results pickle file containing predictions, labels, probabilities and per-fold metrics (`{model}_results.pkl`). 


### Evaluation

Evaluate saved checkpoints and run statistical analysis.

```bash
python evaluate.py --checkpoint_dir ./checkpoints --data_root ./data --output_dir ./results
```

By default, all six models are evaluated. Use `--models SwinT ViT DeiT` to restrict the evaluation to specific architectures. Outputs include a summary CSV and pickle files with per-fold predictions, metrics and statistical test results.

## Results

Five-fold stratified cross-validation on the full dataset.

| Model | Accuracy (%) | F1 macro (%) | AUC macro |
|---|---|---|---|
| Swin-T | 99.13 ± 0.46 | 99.12 ± 0.47 | 0.9997 |
| ViT-B/16 | 98.89 ± 0.51 | 98.89 ± 0.52 | 0.9990 |
| DeiT-S | 98.29 ± 0.35 | 98.28 ± 0.36 | 0.9987 |
| ResNet-50 | 97.14 ± 0.44 | 97.12 ± 0.44 | 0.9982 |
| EfficientNet-B3 | 97.14 ± 0.63 | 97.13 ± 0.63 | 0.9982 |
| CNN3D | 75.09 ± 1.57 | 74.95 ± 1.60 | 0.9420 |

Pairwise McNemar's tests confirm all transformer versus CNN differences at *p* less than 0.01. The Swin-T and ViT-B/16 comparison is not statistically significant (*p* = 0.361).

## Training Configuration

| Parameter | Value |
|---|---|
| Image size | 224 × 224 |
| Batch size | 32 |
| Optimizer | AdamW (β₁=0.9, β₂=0.999) |
| Base LR | 1e-4 |
| Schedule | Cosine annealing, 5-epoch linear warmup |
| Weight decay | 1e-4 |
| Loss | Focal loss (γ=2.0) with label smoothing (0.1) |
| Augmentation | Mixup (α=0.2), CutMix (α=1.0), color jitter, affine, blur, erasing |
| Max epochs | 50 with early stopping patience of 12 |
| Precision | Mixed (FP16) |
| Seed | 42 |

## Repository Structure

```
.
├── models.py          # Six architecture definitions
├── train.py           # Training script, 5-fold CV, single model per run
├── evaluate.py        # Evaluation, per-class metrics, statistical tests
├── requirements.txt   # Python dependencies
├── README.md          
└── LICENSE            # MIT License
```

## Scope and Limitations

The reported results apply to a single-region dataset collected in Maharashtra, India. Cross-regional and extended-class validation remain outside the scope of the supporting study. Users planning field deployment should validate against locally collected data.

## License

Released under the MIT License. See [LICENSE](LICENSE).

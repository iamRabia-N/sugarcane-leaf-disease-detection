import os
import gc
import copy
import math
import time
import random
import pickle
import argparse
import numpy as np
from PIL import Image
from collections import Counter, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.amp import GradScaler, autocast
import torchvision.transforms as T

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

from models import get_model

CLASS_NAMES = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']
NUM_CLASSES = 5
IMG_SIZE = 224
NUM_FOLDS = 5
BATCH_SIZE = 32
NUM_EPOCHS = 50
BASE_LR = 1e-4
MIN_LR = 1e-7
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
PATIENCE = 12
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
MIXUP_PROB = 0.5
NUM_WORKERS = 2
SEED = 42

DATASET_MEAN = [0.498, 0.528, 0.380]
DATASET_STD = [0.176, 0.175, 0.183]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SugarcaneDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.file_paths[idx]).convert('RGB')
        except Exception:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


class CNN3DDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None, val_transform=None, is_train=True):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.val_transform = val_transform
        self.is_train = is_train

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.file_paths[idx]).convert('RGB')
        except Exception:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        if self.is_train and self.transform:
            views = [self.transform(image) for _ in range(3)]
        else:
            t = self.val_transform if self.val_transform else self.transform
            views = [t(image) for _ in range(3)]
        volume = torch.stack(views, dim=1)
        return volume, self.labels[idx]


train_transform = T.Compose([
    T.Resize((256, 256)),
    T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.8, 1.2),
                        interpolation=T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomApply([T.RandomRotation(degrees=45)], p=0.5),
    T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.15)], p=0.8),
    T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
    T.RandomGrayscale(p=0.05),
    T.RandomApply([T.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15))], p=0.4),
    T.ToTensor(),
    T.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
    T.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
])

val_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
])


def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    if x.dim() == 4:
        _, _, H, W = x.shape
    elif x.dim() == 5:
        _, _, _, H, W = x.shape
    else:
        return x, y, y[index], lam
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    x_clone = x.clone()
    if x.dim() == 4:
        x_clone[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    elif x.dim() == 5:
        x_clone[:, :, :, y1:y2, x1:x2] = x[index, :, :, y1:y2, x1:x2]
    lam = 1 - ((y2 - y1) * (x2 - x1) / (H * W))
    return x_clone, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight,
                                  reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        if self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, epoch, device, is_3d=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        use_mix = random.random() < MIXUP_PROB and epoch >= WARMUP_EPOCHS
        if use_mix:
            if random.random() < 0.5:
                if is_3d:
                    B, C, D, H, W = images.shape
                    images_flat = images.view(B, -1, H, W)
                    images_flat, ya, yb, lam = mixup_data(images_flat, targets, MIXUP_ALPHA)
                    images = images_flat.view(B, C, D, H, W)
                else:
                    images, ya, yb, lam = mixup_data(images, targets, MIXUP_ALPHA)
            else:
                images, ya, yb, lam = cutmix_data(images, targets, CUTMIX_ALPHA)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda'):
            outputs = model(images)
            if use_mix:
                loss = mixup_criterion(criterion, outputs, ya, yb, lam)
            else:
                loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        if not use_mix:
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return running_loss / len(dataloader.dataset), correct / max(total, 1)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs.float(), dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(targets.cpu().numpy())
            all_probs.extend(probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    return running_loss / len(dataloader.dataset), accuracy_score(all_labels, all_preds), all_preds, all_labels, all_probs


def compute_metrics(preds, labels, probs):
    metrics = OrderedDict()
    metrics['accuracy'] = accuracy_score(labels, preds)
    metrics['precision_macro'] = precision_score(labels, preds, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(labels, preds, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(labels, preds, average='macro', zero_division=0)
    metrics['precision_weighted'] = precision_score(labels, preds, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(labels, preds, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(labels, preds, average='weighted', zero_division=0)
    try:
        metrics['auc_macro'] = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
        metrics['auc_weighted'] = roc_auc_score(labels, probs, multi_class='ovr', average='weighted')
    except Exception:
        metrics['auc_macro'] = 0.0
        metrics['auc_weighted'] = 0.0
    metrics['confusion_matrix'] = confusion_matrix(labels, preds).tolist()
    return metrics


def build_file_list(data_root):
    filepaths, labels = [], []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(data_root, cls_name)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(f"Missing class directory: {cls_dir}")
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepaths.append(os.path.join(cls_dir, fname))
                labels.append(cls_idx)
    return np.array(filepaths), np.array(labels)


def train_model(model_name, data_root, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_3d = (model_name == 'CNN3D')

    filepaths, labels = build_file_list(data_root)
    class_counts = Counter(labels)
    total = len(labels)
    class_weights = torch.tensor(
        [total / (NUM_CLASSES * class_counts[i]) for i in range(NUM_CLASSES)],
        dtype=torch.float32
    ).to(device)

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(filepaths, labels))

    results = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n[{model_name}] Fold {fold_idx + 1}/{NUM_FOLDS}")
        torch.cuda.empty_cache()
        gc.collect()
        set_seed(SEED + fold_idx)

        train_fps = filepaths[train_idx]
        train_lbs = labels[train_idx]
        val_fps = filepaths[val_idx]
        val_lbs = labels[val_idx]

        if is_3d:
            train_ds = CNN3DDataset(train_fps, train_lbs, transform=train_transform,
                                    val_transform=val_transform, is_train=True)
            val_ds = CNN3DDataset(val_fps, val_lbs, transform=train_transform,
                                  val_transform=val_transform, is_train=False)
        else:
            train_ds = SugarcaneDataset(train_fps, train_lbs, transform=train_transform)
            val_ds = SugarcaneDataset(val_fps, val_lbs, transform=val_transform)

        class_sample_count = np.bincount(train_lbs, minlength=NUM_CLASSES)
        sample_weights = 1.0 / class_sample_count[train_lbs]
        sample_weights = torch.from_numpy(sample_weights).float()
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                                  num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                                  persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True,
                                persistent_workers=True)

        set_seed(SEED + fold_idx)
        model = get_model(model_name, NUM_CLASSES).to(device)

        criterion = FocalLoss(weight=class_weights, gamma=2.0, label_smoothing=LABEL_SMOOTHING)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=BASE_LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))
        scheduler = CosineWarmupScheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS, BASE_LR, MIN_LR)
        scaler = GradScaler()

        best_val_acc = 0.0
        best_val_loss = float('inf')
        best_state = None
        best_preds = None
        best_labels = None
        best_probs = None
        best_epoch = 0
        patience_counter = 0

        for epoch in range(NUM_EPOCHS):
            current_lr = scheduler.step(epoch)
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer,
                                                    scaler, epoch, device, is_3d=is_3d)
            val_loss, val_acc, val_preds, val_labels_arr, val_probs = validate(model, val_loader, criterion, device)

            improved = val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss)
            if improved:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                best_preds = val_preds.copy()
                best_labels = val_labels_arr.copy()
                best_probs = val_probs.copy()
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1

            print(f"  E{epoch+1:>3}/{NUM_EPOCHS}  tr_loss={train_loss:.4f}  tr_acc={train_acc:.3f}  "
                  f"va_loss={val_loss:.4f}  va_acc={val_acc:.3f}  lr={current_lr:.2e}"
                  f"{'  *' if improved else ''}")

            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1} (best epoch: {best_epoch})")
                break

        fold_metrics = compute_metrics(best_preds, best_labels, best_probs)
        fold_metrics['best_epoch'] = best_epoch

        torch.save(best_state, os.path.join(output_dir, f"{model_name}_fold{fold_idx}_best.pth"))

        results.append({
            'model_name': model_name,
            'fold': fold_idx,
            'metrics': fold_metrics,
            'predictions': best_preds.tolist(),
            'labels': best_labels.tolist(),
            'probabilities': best_probs.tolist(),
        })

        with open(os.path.join(output_dir, f"{model_name}_results.pkl"), 'wb') as f:
            pickle.dump(results, f)

        print(f"  Fold {fold_idx+1} done: acc={fold_metrics['accuracy']:.4f} "
              f"f1={fold_metrics['f1_macro']:.4f} auc={fold_metrics['auc_macro']:.4f}")

        del model, optimizer, scheduler, scaler, criterion, train_ds, val_ds, train_loader, val_loader, best_state
        torch.cuda.empty_cache()
        gc.collect()

    accs = [r['metrics']['accuracy'] for r in results]
    f1s = [r['metrics']['f1_macro'] for r in results]
    print(f"\n[{model_name}] Final: acc={np.mean(accs)*100:.2f} +/- {np.std(accs)*100:.2f}%  "
          f"f1={np.mean(f1s)*100:.2f} +/- {np.std(f1s)*100:.2f}%")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['SwinT', 'ViT', 'DeiT', 'ResNet50', 'EfficientNetB3', 'CNN3D'])
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    args = parser.parse_args()
    set_seed(SEED)
    train_model(args.model, args.data_root, args.output_dir)


if __name__ == '__main__':
    main()
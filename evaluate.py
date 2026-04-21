import os
import argparse
import pickle
import numpy as np
import pandas as pd
from itertools import combinations
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from scipy import stats

from models import get_model
from train import (SugarcaneDataset, CNN3DDataset, val_transform,
                   build_file_list, set_seed, CLASS_NAMES, NUM_CLASSES,
                   NUM_FOLDS, BATCH_SIZE, NUM_WORKERS, SEED)


def load_checkpoint(model_name, fold_idx, checkpoint_dir):
    path = os.path.join(checkpoint_dir, f"{model_name}_fold{fold_idx}_best.pth")
    if not os.path.exists(path):
        return None
    model = get_model(model_name, NUM_CLASSES)
    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    return model


def evaluate_fold(model, file_paths, labels, is_3d, device):
    if is_3d:
        dataset = CNN3DDataset(file_paths, labels, transform=val_transform,
                               val_transform=val_transform, is_train=False)
    else:
        dataset = SugarcaneDataset(file_paths, labels, transform=val_transform)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True)

    model = model.to(device).eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs.float(), dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(targets.numpy())
            all_probs.extend(probs)

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def compute_metrics(preds, labels, probs):
    m = OrderedDict()
    m['accuracy'] = accuracy_score(labels, preds)
    m['precision_macro'] = precision_score(labels, preds, average='macro', zero_division=0)
    m['recall_macro'] = recall_score(labels, preds, average='macro', zero_division=0)
    m['f1_macro'] = f1_score(labels, preds, average='macro', zero_division=0)
    m['precision_weighted'] = precision_score(labels, preds, average='weighted', zero_division=0)
    m['recall_weighted'] = recall_score(labels, preds, average='weighted', zero_division=0)
    m['f1_weighted'] = f1_score(labels, preds, average='weighted', zero_division=0)
    try:
        m['auc_macro'] = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
        m['auc_weighted'] = roc_auc_score(labels, probs, multi_class='ovr', average='weighted')
    except Exception:
        m['auc_macro'] = 0.0
        m['auc_weighted'] = 0.0
    m['confusion_matrix'] = confusion_matrix(labels, preds).tolist()
    return m


def per_class_metrics(preds, labels):
    results = {}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        tp = ((preds == cls_idx) & (labels == cls_idx)).sum()
        fp = ((preds == cls_idx) & (labels != cls_idx)).sum()
        fn = ((preds != cls_idx) & (labels == cls_idx)).sum()
        support = (labels == cls_idx).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        results[cls_name] = {
            'precision': float(p),
            'recall': float(r),
            'f1': float(f),
            'support': int(support),
        }
    return results


def evaluate_all_models(checkpoint_dir, data_root, models_list):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    filepaths, labels = build_file_list(data_root)

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(filepaths, labels))

    all_results = {}
    for model_name in models_list:
        is_3d = (model_name == 'CNN3D')
        fold_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            model = load_checkpoint(model_name, fold_idx, checkpoint_dir)
            if model is None:
                print(f"[{model_name}] fold {fold_idx}: checkpoint not found, skipping")
                continue

            val_fps = filepaths[val_idx]
            val_lbs = labels[val_idx]

            preds, lbs, probs = evaluate_fold(model, val_fps, val_lbs, is_3d, device)
            metrics = compute_metrics(preds, lbs, probs)

            fold_results.append({
                'fold': fold_idx,
                'metrics': metrics,
                'predictions': preds.tolist(),
                'labels': lbs.tolist(),
                'probabilities': probs.tolist(),
            })

            print(f"[{model_name}] fold {fold_idx+1}: "
                  f"acc={metrics['accuracy']:.4f} "
                  f"f1={metrics['f1_macro']:.4f} "
                  f"auc={metrics['auc_macro']:.4f}")

            del model
            torch.cuda.empty_cache()

        all_results[model_name] = fold_results

    return all_results


def mcnemar_test(preds1, preds2, labels):
    correct1 = (preds1 == labels)
    correct2 = (preds2 == labels)
    b = np.sum(correct1 & ~correct2)
    c = np.sum(~correct1 & correct2)
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    return chi2, p_value


def statistical_analysis(all_results):
    models = list(all_results.keys())
    labels_all = {}
    preds_all = {}
    accs_per_fold = {}

    for model_name in models:
        preds_concat = []
        labels_concat = []
        for fold_result in all_results[model_name]:
            preds_concat.extend(fold_result['predictions'])
            labels_concat.extend(fold_result['labels'])
        preds_all[model_name] = np.array(preds_concat)
        labels_all[model_name] = np.array(labels_concat)
        accs_per_fold[model_name] = [r['metrics']['accuracy'] for r in all_results[model_name]]

    print("\n" + "="*70)
    print("McNemar's Test (pairwise)")
    print("="*70)
    print(f"{'Pair':<40} {'chi2':>10} {'p-value':>12}")
    print("-"*70)
    mcnemar_results = {}
    for m1, m2 in combinations(models, 2):
        common_labels = labels_all[m1]
        chi2, p = mcnemar_test(preds_all[m1], preds_all[m2], common_labels)
        mcnemar_results[f"{m1}_vs_{m2}"] = {'chi2': chi2, 'p_value': p}
        print(f"{m1 + ' vs ' + m2:<40} {chi2:>10.4f} {p:>12.6f}")

    print("\n" + "="*70)
    print("Paired t-test on fold-level accuracy")
    print("="*70)
    print(f"{'Pair':<40} {'t':>10} {'p-value':>12}")
    print("-"*70)
    ttest_results = {}
    for m1, m2 in combinations(models, 2):
        if len(accs_per_fold[m1]) == len(accs_per_fold[m2]) and len(accs_per_fold[m1]) > 1:
            t, p = stats.ttest_rel(accs_per_fold[m1], accs_per_fold[m2])
            ttest_results[f"{m1}_vs_{m2}"] = {'t': t, 'p_value': p}
            print(f"{m1 + ' vs ' + m2:<40} {t:>10.4f} {p:>12.6f}")

    return mcnemar_results, ttest_results


def print_summary(all_results):
    print("\n" + "="*70)
    print("Summary (mean +/- std across folds)")
    print("="*70)
    print(f"{'Model':<18} {'Accuracy':>16} {'F1 (macro)':>16} {'AUC':>16}")
    print("-"*70)

    rows = []
    for model_name, fold_results in all_results.items():
        if not fold_results:
            continue
        accs = [r['metrics']['accuracy'] for r in fold_results]
        f1s = [r['metrics']['f1_macro'] for r in fold_results]
        aucs = [r['metrics']['auc_macro'] for r in fold_results]

        print(f"{model_name:<18} "
              f"{np.mean(accs)*100:>6.2f} +/- {np.std(accs)*100:.2f} "
              f"{np.mean(f1s)*100:>6.2f} +/- {np.std(f1s)*100:.2f} "
              f"{np.mean(aucs):>6.4f} +/- {np.std(aucs):.4f}")

        rows.append({
            'Model': model_name,
            'Accuracy_mean': np.mean(accs),
            'Accuracy_std': np.std(accs),
            'F1_macro_mean': np.mean(f1s),
            'F1_macro_std': np.std(f1s),
            'AUC_macro_mean': np.mean(aucs),
            'AUC_macro_std': np.std(aucs),
        })

    return pd.DataFrame(rows)


def print_per_class_summary(all_results):
    print("\n" + "="*70)
    print("Per-class metrics (aggregated across folds)")
    print("="*70)

    for model_name, fold_results in all_results.items():
        if not fold_results:
            continue
        all_preds = []
        all_labels = []
        for r in fold_results:
            all_preds.extend(r['predictions'])
            all_labels.extend(r['labels'])
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        print(f"\n{model_name}")
        print(f"  {'Class':<14} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
        print(f"  {'-'*50}")
        pc = per_class_metrics(all_preds, all_labels)
        for cls_name, m in pc.items():
            print(f"  {cls_name:<14} "
                  f"{m['precision']*100:>7.2f}% "
                  f"{m['recall']*100:>7.2f}% "
                  f"{m['f1']*100:>7.2f}% "
                  f"{m['support']:>8}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['SwinT', 'ViT', 'DeiT', 'ResNet50', 'EfficientNetB3', 'CNN3D'])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(SEED)

    all_results = evaluate_all_models(args.checkpoint_dir, args.data_root, args.models)

    summary_df = print_summary(all_results)
    print_per_class_summary(all_results)
    mcnemar_results, ttest_results = statistical_analysis(all_results)

    summary_df.to_csv(os.path.join(args.output_dir, 'summary.csv'), index=False)
    with open(os.path.join(args.output_dir, 'all_results.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
    with open(os.path.join(args.output_dir, 'statistical_tests.pkl'), 'wb') as f:
        pickle.dump({'mcnemar': mcnemar_results, 'ttest': ttest_results}, f)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()
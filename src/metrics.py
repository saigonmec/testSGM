from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,  # Bổ sung để tính PR-AUC
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score,
)
from typing import Dict


def evaluate_predictions(predictions_map, gt_map, verbose: bool = True) -> Dict:
    """
    Đánh giá kết quả predictions.
    Returns: dict với keys: acc, auc, pr_auc, f1, precision, sensitivity, specificity
    """
    all_preds = []
    all_labels = []
    all_probs = []
    for img_path, (pred_label, conf) in predictions_map.items():
        if img_path in gt_map:
            gt_label = gt_map[img_path]
            all_preds.append(pred_label)
            all_labels.append(gt_label)
            all_probs.append(
                conf
            )  # Sử dụng conf làm probability (giả sử conf là prob cho lớp positive)

    if len(all_preds) == 0:
        if verbose:
            print("[ERROR] Không có dữ liệu để đánh giá")
        return {}

    total = len(all_labels)
    correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    acc = correct / total

    try:
        if len(set(all_labels)) == 2:
            auc = roc_auc_score(all_labels, all_probs)
        else:
            auc = roc_auc_score(
                all_labels, all_probs, multi_class="ovr"
            )  # Hỗ trợ multi-class
    except ValueError:
        auc = None

    try:
        if len(set(all_labels)) == 2:
            pr_auc = average_precision_score(all_labels, all_probs)
        else:
            pr_auc = average_precision_score(
                all_labels, all_probs, average="macro"
            )  # Cho multi-class
    except ValueError:
        pr_auc = None

    try:
        precision = precision_score(
            all_labels,
            all_preds,
            average="binary" if len(set(all_labels)) == 2 else "macro",
            zero_division=0,
        )
    except ValueError:
        precision = 0.0

    try:
        recall = recall_score(
            all_labels,
            all_preds,
            average="binary" if len(set(all_labels)) == 2 else "macro",
            zero_division=0,
        )
    except ValueError:
        recall = 0.0

    try:
        f1 = f1_score(
            all_labels,
            all_preds,
            average="binary" if len(set(all_labels)) == 2 else "macro",
            zero_division=0,
        )
    except ValueError:
        f1 = 0.0

    cm = confusion_matrix(all_labels, all_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        sen = recall
        spec = None

    metrics = {
        "acc": acc,
        "auc": auc,
        "pr_auc": pr_auc,  # Bổ sung PR-AUC
        "f1": f1,
        "precision": precision,
        "sensitivity": sen,
        "specificity": spec,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("KẾT QUẢ ĐÁNH GIÁ")
        print("=" * 60)
        acc_str = f"Accuracy: {acc * 100:.2f}%"
        if auc is not None:
            acc_str += f" | ROC-AUC: {auc * 100:.2f}%"
        if pr_auc is not None:
            acc_str += f" | PR-AUC: {pr_auc * 100:.2f}%"
        print(acc_str)
        metrics_str = (
            f"Precision: {precision * 100:.2f}% | Sensitivity: {sen * 100:.2f}%"
        )
        if spec is not None:
            metrics_str += f" | Specificity: {spec * 100:.2f}%"
        metrics_str += f" | F1: {f1 * 100:.2f}%"
        print(metrics_str)
        print("=" * 60)
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, digits=4, zero_division=0))
        print("Confusion Matrix:")
        print(cm)
        print("=" * 60 + "\n")

    return metrics


def print_summary_table(results: Dict[str, Dict]):
    """In bảng tổng hợp các model"""
    print("\n" + "=" * 120)
    print("BẢNG TỔNG HỢP ĐÁNH GIÁ CÁC MODELS")
    print("=" * 120)
    print(
        f"{'Model':<40} | {'Acc':<8} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'F1':<8} | {'Precision':<10} | {'Sensitivity':<12} | {'Specificity':<12}"
    )
    print("-" * 120)
    for model_name, metrics in results.items():
        acc = (
            f"{metrics.get('acc', 0) * 100:.2f}%"
            if metrics.get("acc") is not None
            else "N/A"
        )
        auc = (
            f"{metrics.get('auc', 0) * 100:.2f}%"
            if metrics.get("auc") is not None
            else "N/A"
        )
        pr_auc = (
            f"{metrics.get('pr_auc', 0) * 100:.2f}%"
            if metrics.get("pr_auc") is not None
            else "N/A"
        )
        f1 = (
            f"{metrics.get('f1', 0) * 100:.2f}%"
            if metrics.get("f1") is not None
            else "N/A"
        )
        prec = (
            f"{metrics.get('precision', 0) * 100:.2f}%"
            if metrics.get("precision") is not None
            else "N/A"
        )
        sens = (
            f"{metrics.get('sensitivity', 0) * 100:.2f}%"
            if metrics.get("sensitivity") is not None
            else "N/A"
        )
        spec = (
            f"{metrics.get('specificity', 0) * 100:.2f}%"
            if metrics.get("specificity") is not None
            else "N/A"
        )
        print(
            f"{model_name:<40} | {acc:<8} | {auc:<8} | {pr_auc:<8} | {f1:<8} | {prec:<10} | {sens:<12} | {spec:<12}"
        )
    print("=" * 120 + "\n")

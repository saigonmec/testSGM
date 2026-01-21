import os
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from src.processing import (
    load_full_model,
    safe_convert_to_rgb,
    build_preprocess,
    predict_binary_label_and_confidence,
    gradcam_heatmap_based,
    overlay_otsu,
    _longest_common_substring,  # Thêm import này ở đầu file nếu chưa có
)
from src.utils import print_stats
from src.plot import parse_bbxs_from_string, draw_bboxes_on_image
from src.metrics import evaluate_predictions, print_summary_table
from src.image_matching import print_image_matching_stats
from src.csv_processor import load_and_process_csv
from src.gradcam_utils_patch import pre_mil_gradcam, mil_gradcam


def run_predictions(
    dataset_folder: str,
    fixed_csv_path: Path,
    pretrained_model_path: str,
    output_folder: str,
    gt_map: Dict = None,
    device: str = None,
    save_gradcam: bool = True,
) -> Tuple[Dict, List[Dict]]:
    """Chạy model qua tất cả ảnh, lưu kết quả"""
    dataset_root = Path(dataset_folder).expanduser().resolve()
    output_root = Path(output_folder).expanduser().resolve()

    dev = (
        torch.device(device)
        if device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[INFO] Using device: {dev}")

    model, input_size, model_name, gradcam_layer, normalize, num_patches, arch_type = (
        load_full_model(pretrained_model_path)
    )
    model = model.to(dev).eval()
    target_layer = gradcam_layer or "layer4"

    image_list = []
    with open(fixed_csv_path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        fieldnames = rdr.fieldnames if rdr.fieldnames else []

        image_path_key = None
        link_key = None
        for k in fieldnames:
            k_lower = k.lower()
            if k_lower == "image_path":
                image_path_key = k
            elif k_lower == "link":
                link_key = k

        for row in rdr:
            img_path_str = (
                row.get(image_path_key, "") if image_path_key else row.get(link_key, "")
            )
            img_path_str = img_path_str.strip().replace("\\", "/")
            if img_path_str:
                image_list.append((img_path_str, row))

    images_by_folder = defaultdict(list)
    for img_path_str, row in image_list:
        rel_path = Path(img_path_str)
        folder_key = rel_path.parent
        images_by_folder[folder_key].append((img_path_str, row))

    predictions_map = {}
    output_rows = []

    for folder_key in tqdm(
        sorted(images_by_folder.keys()), desc="Processing folders", unit="folder"
    ):
        folder_metadata = []

        for img_path_str, row in images_by_folder[folder_key]:
            img_path = Path(img_path_str)
            if not img_path.is_absolute():
                img_path = dataset_root / img_path

            if not img_path.exists():
                print(f"[WARN] Không tìm thấy ảnh: {img_path}")
                continue

            if save_gradcam:
                output_root.mkdir(parents=True, exist_ok=True)

                img_out_dir = output_root / folder_key
                img_out_dir.mkdir(parents=True, exist_ok=True)

                gradcam_out_path = img_out_dir / f"{img_path.stem}_gradcam.png"

            if arch_type is None or str(arch_type).lower() == "based":
                img = safe_convert_to_rgb(Image.open(img_path))
                preprocess = build_preprocess(tuple(input_size), normalize)

                # x = preprocess(img).to(dev)
                x = preprocess(img).unsqueeze(0).to(dev)
                # x.shape sẽ là [1, C, H, W]

                with torch.no_grad():
                    logits = model(x)

                label, conf, _ = predict_binary_label_and_confidence(logits)
                predictions_map[img_path_str] = (label, conf)

                out_row = row.copy()
                out_row["predict"] = str(label)
                out_row["confidence"] = f"{conf:.6f}"
                output_rows.append(out_row)

                gt_label = gt_map.get(img_path_str, "") if gt_map else ""
                folder_metadata.append(
                    {
                        "image_id": img_path.name,
                        "ground_truth": str(gt_label) if gt_label != "" else "",
                        "label": str(label),
                        "confidence_score": f"{conf:.6f}",
                    }
                )

                if save_gradcam:
                    bbxs_str = row.get("bbxs", "[]")
                    bbxs = parse_bbxs_from_string(bbxs_str)
                    if bbxs:
                        img_with_bbox = draw_bboxes_on_image(
                            img, bbxs, color="red", width=3
                        )
                    else:
                        img_with_bbox = img

                    img_out_path = img_out_dir / f"{img_path.stem}.png"
                    try:
                        img_with_bbox.save(img_out_path, format="PNG")
                    except Exception as e:
                        print(f"[ERROR] Không thể lưu ảnh gốc {img_path.stem}: {e}")

                    cam = gradcam_heatmap_based(model, x, target_layer, class_idx=1)
                    overlay = overlay_otsu(img, cam, alpha=0.55)
                    overlay.save(gradcam_out_path, format="PNG")
            else:
                model_tuple = (
                    model,
                    tuple(input_size),
                    model_name,
                    target_layer,
                    normalize,
                    num_patches,
                    arch_type,
                )

                # model, input_tensor, img, target_layer, class_idx, pred_class, pred_prob
                (
                    model_out,
                    input_tensor,
                    img0,
                    layer0,
                    class_idx0,
                    pred_class0,
                    prob0,
                ) = pre_mil_gradcam(model_tuple, str(img_path))

                input_tensor = input_tensor.to(dev)
                model_out = model_out.to(dev)

                label, conf = pred_class0, prob0
                predictions_map[img_path_str] = (label, conf)

                out_row = row.copy()
                out_row["predict"] = str(label)
                out_row["confidence"] = f"{conf:.6f}"
                output_rows.append(out_row)

                gt_label = gt_map.get(img_path_str, "") if gt_map else ""
                folder_metadata.append(
                    {
                        "image_id": img_path.name,
                        "ground_truth": str(gt_label) if gt_label != "" else "",
                        "label": str(label),
                        "confidence_score": f"{conf:.6f}",
                    }
                )

                if save_gradcam:
                    bbxs_str = row.get("bbxs", "[]")
                    bbxs = parse_bbxs_from_string(bbxs_str)
                    if bbxs:
                        img_with_bbox = draw_bboxes_on_image(
                            img0, bbxs, color="red", width=3
                        )
                    else:
                        img_with_bbox = img0

                    img_out_path = img_out_dir / f"{img_path.stem}.png"
                    try:
                        img_with_bbox.save(img_out_path, format="PNG")
                    except Exception as e:
                        print(f"[ERROR] Không thể lưu ảnh gốc {img_path.stem}: {e}")

                    cam = mil_gradcam(model_out, input_tensor, layer0, class_idx=1)
                    if cam.ndim == 3:
                        cam = cam.max(axis=0).astype(np.uint8)
                    overlay = overlay_otsu(img0, cam, alpha=0.55)
                    overlay.save(gradcam_out_path, format="PNG")

        # Chỉ tạo folder và lưu metadata.csv nếu save_gradcam=True
        if save_gradcam:
            folder_csv_path = output_root / folder_key / "metadata.csv"
            folder_csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(folder_csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "image_id",
                        "ground_truth",
                        "label",
                        "confidence_score",
                    ],
                )
                w.writeheader()
                w.writerows(folder_metadata)

    out_fieldnames = list(fieldnames) if fieldnames else []
    if "predict" not in out_fieldnames:
        out_fieldnames.append("predict")
    if "confidence" not in out_fieldnames:
        out_fieldnames.append("confidence")

    # Chỉ lưu output_csv_path nếu save_gradcam=True
    if save_gradcam:
        output_csv_path = output_root / "metadata.csv"
        with open(output_csv_path, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=out_fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
        print(f"[INFO] Đã lưu kết quả dự đoán tổng: {output_csv_path}")

    return predictions_map, output_rows


def main(
    dataset_folder: str,
    ground_truth_path: str,
    pretrained_model_paths: List[str],
    output_folder: str,
    device: str = None,
    save_gradcam: bool = True,
    ready_csv: bool = False,
):
    """Main workflow"""
    print("\n" + "=" * 60)
    print("BẮT ĐẦU WORKFLOW")
    print("=" * 60 + "\n")

    print("[Kiểm tra khớp ảnh với image_id trong CSV]")
    print_image_matching_stats(dataset_folder, ground_truth_path)

    print("[Bước 1-2] Đọc và xử lý ground-truth CSV...")
    fixed_csv_path, processed_rows, stats = load_and_process_csv(
        ground_truth_path, dataset_folder, ready_csv=ready_csv
    )

    if stats:
        print_stats(stats)

    gt_map = {}
    with open(fixed_csv_path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        fieldnames = rdr.fieldnames if rdr.fieldnames else []

        image_path_key = None
        link_key = None
        for k in fieldnames:
            k_lower = k.lower()
            if k_lower == "image_path":
                image_path_key = k
            elif k_lower == "link":
                link_key = k

        for row in rdr:
            img_path_str = (
                row.get(image_path_key, "") if image_path_key else row.get(link_key, "")
            )
            img_path_str = img_path_str.strip().replace("\\", "/")
            cancer_val = row.get("cancer", "").strip()
            try:
                gt_label = int(float(cancer_val))
                gt_map[img_path_str] = gt_label
            except Exception:
                pass

    # Chạy predictions cho từng model
    all_results = {}

    for idx, model_path in enumerate(pretrained_model_paths, 1):
        print(f"\n[Bước 4.{idx}] Chạy model predictions: {Path(model_path).name}...")

        # Load model info để lấy tên
        _, input_size, model_name, _, _, _, _ = load_full_model(model_path)
        print("[INFO] Model input size:", input_size, "Model name:", model_name)
        model_display_name = (
            f"{model_name}_{input_size[0]}x{input_size[1]}_{Path(model_path).stem}"
            if model_name
            else f"Model{idx}"
        )

        # Tạo output folder riêng cho mỗi model
        model_output_folder = Path(output_folder) / model_display_name

        predictions_map, output_rows = run_predictions(
            dataset_folder=dataset_folder,
            fixed_csv_path=fixed_csv_path,
            pretrained_model_path=model_path,
            output_folder=str(model_output_folder),
            gt_map=gt_map,
            device=device,
            save_gradcam=save_gradcam,
        )

        print(f"[Bước 5.{idx}] Đánh giá kết quả model: {model_display_name}...")
        metrics = evaluate_predictions(predictions_map, gt_map, verbose=True)
        all_results[model_display_name] = metrics

    # In bảng tổng hợp nếu có nhiều hơn 1 model
    if len(pretrained_model_paths) > 1:
        model_names = list(all_results.keys())
        # Loại bỏ prefix/suffix trước khi tìm phần chung
        stripped_names = [_strip_model_name(m) for m in model_names]
        common = _longest_common_substring(stripped_names)
        short_names = [m.replace(common, "") if common else m for m in stripped_names]
        short_results = {
            short: all_results[orig] for short, orig in zip(short_names, model_names)
        }
        print_summary_table(short_results)
        if common:
            print(f"\nCommon part in model names: '{common}'")

    print("\n" + "=" * 60)
    print("HOÀN THÀNH WORKFLOW")
    print("=" * 60 + "\n")


def _strip_model_name(name: str) -> str:
    parts = name.split("_")
    # Loại bỏ prefix: bỏ tất cả các phần trước phần đầu tiên chứa số (thường là tên model version)
    prefix_idx = 0
    for i, p in enumerate(parts):
        if any(c.isdigit() for c in p):
            prefix_idx = i
            break
    parts = parts[prefix_idx:]
    # Loại bỏ suffix 'full' nếu có
    if parts and parts[-1].lower() == "full":
        parts = parts[:-1]
    return "_".join(parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGM Prediction Workflow")
    parser.add_argument(
        "--dataset_folder", type=str, required=True, help="Root folder chứa các ảnh"
    )
    parser.add_argument(
        "--ground_truth_path",
        type=str,
        required=True,
        help="Đường dẫn file CSV ground-truth",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        required=True,
        help="Đường dẫn model pretrained (model chính)",
    )
    parser.add_argument(
        "--pretrained_model_path2",
        type=str,
        default=None,
        help="Đường dẫn model pretrained thứ 2 (optional)",
    )
    parser.add_argument(
        "--pretrained_model_path3",
        type=str,
        default=None,
        help="Đường dẫn model pretrained thứ 3 (optional)",
    )
    parser.add_argument(
        "--pretrained_model_path4",
        type=str,
        default=None,
        help="Đường dẫn model pretrained thứ 4 (optional)",
    )
    parser.add_argument(
        "--pretrained_model_path5",
        type=str,
        default=None,
        help="Đường dẫn model pretrained thứ 5 (optional)",
    )
    parser.add_argument(
        "--pretrained_model_path6",
        type=str,
        default=None,
        help="Đường dẫn model pretrained thứ 6 (optional)",
    )
    parser.add_argument(
        "--output_folder", type=str, required=True, help="Thư mục lưu kết quả"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device để chạy model (cuda/cpu)"
    )
    parser.add_argument(
        "--save_gradcam", action="store_true", help="Lưu ảnh gradcam và ảnh gốc"
    )
    parser.add_argument(
        "--ready_csv",
        action="store_true",
        help="CSV đã có cột cancer, không cần tạo lại",
    )

    args = parser.parse_args()

    # Collect all model paths
    model_paths = [args.pretrained_model_path]
    if args.pretrained_model_path2:
        model_paths.append(args.pretrained_model_path2)
    if args.pretrained_model_path3:
        model_paths.append(args.pretrained_model_path3)
    if args.pretrained_model_path4:
        model_paths.append(args.pretrained_model_path4)
    if args.pretrained_model_path5:
        model_paths.append(args.pretrained_model_path5)
    if args.pretrained_model_path6:
        model_paths.append(args.pretrained_model_path6)

    main(
        dataset_folder=args.dataset_folder,
        ground_truth_path=args.ground_truth_path,
        pretrained_model_paths=model_paths,
        output_folder=args.output_folder,
        device=args.device,
        save_gradcam=args.save_gradcam,
        ready_csv=args.ready_csv,
    )

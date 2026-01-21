import os
import csv
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from skimage.filters import threshold_otsu
from src.engines import evaluate_model

import re
import matplotlib.pyplot as plt

VIEW_ORDER = [("CC", "R"), ("CC", "L"), ("MLO", "R"), ("MLO", "L")]
SAVE_GRADCAM = False

# Not use in pipeline
GRADCAM_RE = re.compile(
    r"^(?P<base>.+?)_(?P<view>CC|MLO)_(?P<side>L|R)(?:_(?P<idx>\d+))?_gradcam\.png$",
    re.IGNORECASE,
)


# ===== Your original loader (kept) =====
def load_full_model(
    model_path: str,
) -> tuple[
    nn.Module,
    tuple[int, int],
    str | None,
    str | None,
    dict[str, list[float]] | None,
    int | None,
    str | None,
]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = checkpoint["model"]
    model.eval()

    input_size = checkpoint.get("input_size", (448, 448))
    model_name = checkpoint.get("model_name", None)
    gradcam_layer = checkpoint.get("gradcam_layer", None)
    normalize = checkpoint.get("normalize", None)
    num_patches = checkpoint.get("num_patches", None)
    arch_type = checkpoint.get("arch_type", None)

    return (
        model,
        input_size,
        model_name,
        gradcam_layer,
        normalize,
        num_patches,
        arch_type,
    )


def load_full_model2(
    model_path: str,
):
    from torch import nn  # local import for typing compatibility

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = checkpoint["model"]
    model.eval()

    input_size = checkpoint.get("input_size", (448, 448))
    model_name = checkpoint.get("model_name", None)
    gradcam_layer = checkpoint.get("gradcam_layer", None)
    normalize = checkpoint.get("normalize", None)
    num_patches = checkpoint.get("num_patches", None)
    arch_type = checkpoint.get("arch_type", None)
    # print(f"[load_full_model] Loaded model from {model_path}")
    # print(f"[load_full_model] Model input size: {input_size}, Model name: {model_name}")
    # print(f"[load_full_model] GradCAM layer: {gradcam_layer}, Normalize: {normalize}")
    # print(f"[load_full_model] Num patches: {num_patches}, Arch type: {arch_type}")

    return (
        model,
        input_size,
        model_name,
        gradcam_layer,
        normalize,
        num_patches,
        arch_type,
    )


def safe_convert_to_rgb(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGB" else img.convert("RGB")


def list_images(root: Path, exts=(".png", ".jpg", ".jpeg")) -> List[Path]:
    out = []
    for dp, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(exts):
                out.append(Path(dp) / f)
    return out


def build_preprocess(input_size: Tuple[int, int], normalize: Optional[Dict[str, Any]]):
    ops = [transforms.Resize(input_size), transforms.ToTensor()]
    if normalize and "mean" in normalize and "std" in normalize:
        ops.append(transforms.Normalize(mean=normalize["mean"], std=normalize["std"]))
    return transforms.Compose(ops)


def _get_module_by_name(model: torch.nn.Module, layer_name: str) -> torch.nn.Module:
    mods = dict(model.named_modules())
    if layer_name not in mods:
        raise ValueError(f"Target layer '{layer_name}' not found in model.")
    return mods[layer_name]


def gradcam_heatmap_based(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_layer: str,
    class_idx: int,
) -> np.ndarray:
    """Minimal GradCAM for CNN-like 'based' models. Returns uint8 heatmap."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    acts, grads = [], []
    layer = _get_module_by_name(model, target_layer)

    def fwd_hook(_, __, output):
        acts.append(output)

    def bwd_hook(_, __, grad_out):
        grads.append(grad_out[0])

    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)

    out = model(x)
    if out.ndim == 1:
        out = out.unsqueeze(0)

    model.zero_grad(set_to_none=True)
    out[0, class_idx].backward()

    h1.remove()
    h2.remove()
    if not acts or not grads:
        raise RuntimeError("GradCAM hooks failed.")

    A = acts[0].detach()  # [1,C,H,W]
    G = grads[0].detach()  # [1,C,H,W]
    w = G.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * A).sum(dim=1))[0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = (cam * 255).to(torch.uint8).cpu().numpy()
    return cam


def overlay_otsu(
    img_rgb: Image.Image, cam_uint8: np.ndarray, alpha: float = 0.55
) -> Image.Image:
    cam_img = Image.fromarray(cam_uint8).resize(img_rgb.size, Image.Resampling.BILINEAR)
    cam_np = np.array(cam_img)

    thr = threshold_otsu(cam_np)
    mask = cam_np >= thr

    x = cam_np.astype(np.float32) / 255.0
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    heat = np.stack([r, g, b], axis=-1)

    base = np.array(img_rgb).astype(np.float32) / 255.0
    out = base.copy()
    out[mask] = (1 - alpha) * base[mask] + alpha * heat[mask]
    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)

    img_out = Image.fromarray(out)  # no mode= (Pillow 13 warning fix)
    if img_out.mode != "RGB":
        img_out = img_out.convert("RGB")
    return img_out


def predict_binary_label_and_confidence(
    logits: torch.Tensor,
) -> Tuple[int, float, float]:
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)

    if logits.shape[1] == 1:
        p1 = torch.sigmoid(logits)[0, 0].item()
    else:
        probs = torch.softmax(logits, dim=1)[0]
        p1 = float(probs[1].item()) if probs.numel() > 1 else 0.0

    label = 1 if p1 >= 0.5 else 0
    conf = p1 if label == 1 else (1 - p1)
    return label, float(conf), float(p1)


def SGMpredict_folder(
    input_root: str,
    model_path: str,
    ground_truth_path: Optional[str] = None,
    output_root: Optional[str] = None,
    device: Optional[str] = None,
) -> Path:
    """
    - Dự đoán dựa trên danh sách ảnh từ file CSV ground-truth (cột image_path hoặc link)
    - Per-subfolder result.csv
    - Always save original as PNG
    - Only label==1 => overlay GradCAM; else *_gradcam.png is the original image
    - Uses load_full_model() to get arch_type/num_patches/gradcam_layer/normalize/input_size
    """
    in_root = Path(input_root).expanduser().resolve()
    out_root = (
        Path(output_root).expanduser().resolve()
        if output_root
        else in_root.parent / f"{in_root.name}_predict"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"ground_truth_path: " + str(ground_truth_path))
    if ground_truth_path is not None and os.path.exists(ground_truth_path):
        print(f"ground_truth_path ok.")

    dev = (
        torch.device(device)
        if device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[SGMpredict_folder] Using device: {dev}")

    # Load full model metadata
    model, input_size, model_name, gradcam_layer, normalize, num_patches, arch_type = (
        load_full_model(model_path)
    )
    model = model.to(dev).eval()
    target_layer = gradcam_layer or "layer4"
    preprocess = build_preprocess(tuple(input_size), normalize)

    # Load ground-truth CSV, xử lý group/chuẩn hóa
    image_list = []
    if ground_truth_path is not None and os.path.exists(ground_truth_path):
        with open(ground_truth_path, "r", newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            fieldnames = rdr.fieldnames if rdr.fieldnames else []
            image_path_key = None
            link_key = None
            for k in fieldnames:
                if k.lower() == "image_path":
                    image_path_key = k
                if k.lower() == "link":
                    link_key = k
            for row in rdr:
                img_path_str = (
                    row.get(image_path_key, "")
                    if image_path_key
                    else row.get(link_key, "")
                )
                img_path_str = img_path_str.strip()
                if img_path_str:
                    image_list.append(img_path_str)
    else:
        # Nếu không có ground-truth CSV thì lấy toàn bộ ảnh trong thư mục như cũ
        image_list = [str(p) for p in list_images(in_root)]

    rows_by_folder: Dict[Path, List[Dict[str, str]]] = {}

    for img_path_str in tqdm(image_list, desc="SGMpredict", unit="img"):
        img_path = Path(img_path_str)
        if not img_path.is_absolute():
            img_path = in_root / img_path
        if not img_path.exists():
            print(f"[WARN] Không tìm thấy ảnh: {img_path}")
            continue
        rel = (
            img_path.relative_to(in_root)
            if img_path.is_relative_to(in_root)
            else Path(img_path.name)
        )
        rel_dir = rel.parent
        img_our_dir = out_root / rel_dir / "img"
        # img_our_dir.mkdir(parents=True, exist_ok=True)

        # Load and save original as PNG
        img = safe_convert_to_rgb(Image.open(img_path))

        x = preprocess(img).unsqueeze(0).to(dev)

        with torch.no_grad():
            logits = model(x)

        label, conf, _ = predict_binary_label_and_confidence(logits)

        out_grad_path = img_our_dir / f"{img_path.stem}.png"

        rows_by_folder.setdefault(rel_dir, []).append(
            {
                "image_id": str(img_path.name),
                "label": str(label),
                "confidence_score": f"{conf:.6f}",
            }
        )

    #  Compute and print prediction stats
    if ground_truth_path is not None and os.path.exists(ground_truth_path):
        evaluate_model(
            rows_by_folder=rows_by_folder,
            ground_truth_csv=ground_truth_path,
        )

    # Write CSV per subfolder
    if SAVE_GRADCAM:
        for rel_dir, rows in rows_by_folder.items():
            csv_path = (
                (out_root / rel_dir / "metadata.csv")
                if str(rel_dir) != "."
                else (out_root / "metadata.csv")
            )
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f, fieldnames=["image_id", "label", "confidence_score"]
                )
                w.writeheader()
                w.writerows(rows)

        print(f"[DONE] Saved outputs to: {out_root}")

    return out_root


def _read_result_csv(csv_path: Path) -> Dict[str, Tuple[int, float]]:
    m: Dict[str, Tuple[int, float]] = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            stem = Path(row["image_id"].strip()).stem  # xxx_CC_R_2
            m[stem] = (int(row["label"]), float(row["confidence_score"]))
    return m


# Not use in pipeline
def _parse_gradcam_name(name: str) -> Optional[Tuple[str, str, str, Optional[int]]]:
    m = GRADCAM_RE.match(name)
    if not m:
        return None
    base = m.group("base")
    view = m.group("view").upper()
    side = m.group("side").upper()
    idx = m.group("idx")
    return base, view, side, (int(idx) if idx else None)


def _blank_image(size=(512, 512)) -> Image.Image:
    w, h = size
    return Image.fromarray(np.ones((h, w, 3), dtype=np.uint8) * 255)


def _load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _compact_title(
    view: str, side: str, label: Optional[int], conf: Optional[float]
) -> str:
    token = f"{view}_{side}"
    if label is None or conf is None:
        return f"{token} | NA"
    status = "Positive" if label == 1 else "Negative"
    return f"{token} | {status} | {conf * 100:.2f}%"


def _find_original_from_gradcam(folder: Path, gradcam_path: Path) -> Optional[Path]:
    """
    gradcam file: <stem>_gradcam.png
    original:     <stem>.png  (prefer), fallback .jpg/.jpeg
    """
    name = gradcam_path.name
    if not name.endswith("_gradcam.png"):
        return None
    stem = name[: -len("_gradcam.png")]  # <stem>
    for ext in [".png", ".jpg", ".jpeg"]:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _concat_horiz(img_left: Image.Image, img_right: Image.Image) -> Image.Image:
    """
    Concatenate two images horizontally (same height by resizing right to left height).
    """
    left = img_left
    right = img_right
    if right.size[1] != left.size[1]:
        new_w = int(right.size[0] * (left.size[1] / right.size[1]))
        right = right.resize((new_w, left.size[1]), Image.Resampling.BILINEAR)

    out = Image.new("RGB", (left.size[0] + right.size[0], left.size[1]))
    out.paste(left, (0, 0))
    out.paste(right, (left.size[0], 0))
    return out


# Not use in pipeline
def viz_sgm_predict(root_folder: str, figsize=(18, 5), compare: bool = False) -> None:
    """
    - Parse patterns ONLY from *_gradcam.png filenames
    - 1-row layout: CC_R, CC_L, MLO_R, MLO_L
    - Title compact (no .png)
    - If compare=True: Positive tiles show [original | gradcam] merged horizontally
    """
    root = Path(root_folder).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")

    folders = sorted(
        [p for p in root.rglob("*") if p.is_dir() and (p / "result.csv").exists()]
    )
    if not folders:
        raise FileNotFoundError(f"No subfolder with result.csv found under: {root}")

    for folder in folders:
        csv_map = _read_result_csv(folder / "result.csv")
        gradcams = sorted(folder.glob("*_gradcam.png"))
        if not gradcams:
            print(f"[WARN] No *_gradcam.png in: {folder}")
            continue

        groups: Dict[str, Dict[Tuple[str, str], Dict]] = {}

        for p in gradcams:
            parsed = _parse_gradcam_name(p.name)
            if parsed is None:
                continue
            base, view, side, idx = parsed
            key = (view, side)

            stem = p.name[: -len("_gradcam.png")]  # xxx_CC_R_2
            label_conf = csv_map.get(stem)
            label = label_conf[0] if label_conf else None
            conf = label_conf[1] if label_conf else None

            cand = {"path": p, "label": label, "conf": conf, "idx": idx}

            groups.setdefault(base, {})
            if key not in groups[base]:
                groups[base][key] = cand
            else:
                prev = groups[base][key]
                prev_conf = prev["conf"]
                cand_conf = cand["conf"]
                if prev_conf is None and cand_conf is not None:
                    groups[base][key] = cand
                elif (
                    prev_conf is not None
                    and cand_conf is not None
                    and cand_conf > prev_conf
                ):
                    groups[base][key] = cand

        if not groups:
            print(f"[WARN] No CC/MLO L/R gradcam pattern matched in: {folder}")
            continue

        for base_id in sorted(groups.keys()):
            g = groups[base_id]

            # derive blank size from first available image
            first_img = None
            for view, side in VIEW_ORDER:
                item = g.get((view, side))
                if item:
                    first_img = _load_rgb(item["path"])
                    break
            blank_size = first_img.size if first_img else (512, 512)

            imgs: List[Image.Image] = []
            titles: List[str] = []

            for view, side in VIEW_ORDER:
                item = g.get((view, side))
                if item is None:
                    imgs.append(_blank_image(blank_size))
                    titles.append(f"{view}_{side} | Missing")
                    continue

                grad_img = _load_rgb(item["path"])
                label = item["label"]
                conf = item["conf"]

                if compare and label == 1:
                    orig_path = _find_original_from_gradcam(folder, item["path"])
                    if orig_path and orig_path.exists():
                        orig_img = _load_rgb(orig_path)
                        merged = _concat_horiz(orig_img, grad_img)  # [orig | gradcam]
                        imgs.append(merged)
                    else:
                        imgs.append(grad_img)  # fallback if original missing
                else:
                    imgs.append(grad_img)

                titles.append(_compact_title(view, side, label, conf))

            fig, axes = plt.subplots(1, 4, figsize=figsize)
            fig.suptitle(f"{folder.name} | {base_id}", fontsize=14)

            for ax, im, ttl in zip(axes, imgs, titles):
                ax.imshow(im)
                ax.set_title(ttl, fontsize=10)
                ax.axis("off")

            plt.tight_layout()
            plt.show()

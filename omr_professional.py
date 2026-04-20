#!/usr/bin/env python3
"""
Profesyonel OMR (Optik İşaret Tanıma) işleme aracı.

Yeni sürüm iyileştirmeleri:
- Sayfa kayması/döndürme için marker tabanlı hizalama (homography)
- Marker bulunamazsa otomatik fallback stratejisi
- Balon merkezlerinde lokal arama ile sıkı (tight) tespit
- Sonuçların CSV + anotasyonlu PDF üretimi
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import yaml
from PIL import Image

LOGGER = logging.getLogger("omr")


@dataclass
class BubbleOption:
    key: str
    x: float
    y: float
    r: float


@dataclass
class FieldDef:
    field_id: str
    label: str
    field_type: str
    options: list[BubbleOption]


@dataclass
class MarkerDef:
    marker_id: str
    x: float
    y: float


@dataclass
class PageConfig:
    fields: list[FieldDef]
    anchors: list[MarkerDef]
    marker_search_window: float = 0.08


@dataclass
class OMRConfig:
    name: str
    pages: dict[int, PageConfig]


@dataclass
class OptionScore:
    key: str
    fill_ratio: float
    z_score: float
    marked: bool


class OMRProcessor:
    def __init__(self, config: OMRConfig, mark_z_threshold: float = 1.0, min_fill_ratio: float = 0.22):
        self.config = config
        self.mark_z_threshold = mark_z_threshold
        self.min_fill_ratio = min_fill_ratio

    @staticmethod
    def pdf_to_images(pdf_path: Path, dpi: int = 300) -> list[np.ndarray]:
        images: list[np.ndarray] = []
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        with fitz.open(pdf_path) as doc:
            for page in doc:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                if pix.n == 4:
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
                else:
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                images.append(arr)
        return images

    @staticmethod
    def preprocess_page(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=40, sigmaSpace=40)

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            35,
            8,
        )
        return enhanced, binary

    @staticmethod
    def _safe_crop(binary: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> tuple[np.ndarray, tuple[int, int]]:
        h, w = binary.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros((1, 1), dtype=np.uint8), (x1, y1)
        return binary[y1:y2, x1:x2], (x1, y1)

    def _detect_marker(self, binary: np.ndarray, exp_x: int, exp_y: int, window: int) -> tuple[int, int] | None:
        roi, (ox, oy) = self._safe_crop(binary, exp_x - window, exp_y - window, exp_x + window, exp_y + window)
        if roi.size == 0:
            return None

        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = -1.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
            if len(approx) != 4:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w <= 3 or h <= 3:
                continue

            ratio = min(w, h) / max(w, h)
            fill = area / float(w * h)
            score = ratio * fill * area
            if score > best_score:
                cx = ox + x + w // 2
                cy = oy + y + h // 2
                best = (cx, cy)
                best_score = score

        return best

    def _align_with_markers(
        self,
        image: np.ndarray,
        binary: np.ndarray,
        page_cfg: PageConfig,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        h, w = binary.shape[:2]
        anchors = page_cfg.anchors
        meta: dict[str, Any] = {"alignment_status": "not_configured", "detected_markers": 0}

        if len(anchors) < 4:
            return image, binary, meta

        exp_pts = []
        det_pts = []
        window = int(page_cfg.marker_search_window * min(w, h))

        for m in anchors:
            ex, ey = int(m.x * w), int(m.y * h)
            found = self._detect_marker(binary, ex, ey, window)
            if found is not None:
                exp_pts.append([ex, ey])
                det_pts.append([found[0], found[1]])

        meta["detected_markers"] = len(det_pts)

        if len(det_pts) < 4:
            meta["alignment_status"] = "insufficient_markers"
            return image, binary, meta

        det_arr = np.array(det_pts, dtype=np.float32)
        exp_arr = np.array(exp_pts, dtype=np.float32)
        H, mask = cv2.findHomography(det_arr, exp_arr, method=cv2.RANSAC, ransacReprojThreshold=3.0)

        if H is None:
            meta["alignment_status"] = "homography_failed"
            return image, binary, meta

        aligned_image = cv2.warpPerspective(image, H, (w, h), flags=cv2.INTER_LINEAR)
        aligned_binary = cv2.warpPerspective(binary, H, (w, h), flags=cv2.INTER_NEAREST)
        inliers = int(mask.sum()) if mask is not None else 0

        meta.update({"alignment_status": "ok", "inliers": inliers})
        return aligned_image, aligned_binary, meta

    @staticmethod
    def _bubble_fill_ratio(binary_img: np.ndarray, cx: int, cy: int, radius: int) -> float:
        h, w = binary_img.shape[:2]
        x1, x2 = max(cx - radius, 0), min(cx + radius, w - 1)
        y1, y2 = max(cy - radius, 0), min(cy + radius, h - 1)

        roi = binary_img[y1 : y2 + 1, x1 : x2 + 1]
        mask = np.zeros_like(roi, dtype=np.uint8)
        center = (cx - x1, cy - y1)
        cv2.circle(mask, center, radius, 255, thickness=-1)

        total_pixels = np.count_nonzero(mask)
        if total_pixels == 0:
            return 0.0

        filled_pixels = np.count_nonzero(cv2.bitwise_and(roi, mask))
        return float(filled_pixels / total_pixels)

    def _refine_bubble_center(self, binary: np.ndarray, cx: int, cy: int, radius: int, delta: int = 6) -> tuple[int, int, float]:
        best_score = -1.0
        best_xy = (cx, cy)

        for dy in range(-delta, delta + 1, 2):
            for dx in range(-delta, delta + 1, 2):
                nx, ny = cx + dx, cy + dy
                score = self._bubble_fill_ratio(binary, nx, ny, radius)
                if score > best_score:
                    best_score = score
                    best_xy = (nx, ny)

        return best_xy[0], best_xy[1], best_score

    def _score_options(
        self,
        binary_img: np.ndarray,
        options: list[BubbleOption],
        width: int,
        height: int,
    ) -> tuple[list[OptionScore], dict[str, tuple[int, int, int]]]:
        fill_values = []
        coords: list[tuple[str, int, int, int, float]] = []
        refined_map: dict[str, tuple[int, int, int]] = {}

        for opt in options:
            cx, cy = int(opt.x * width), int(opt.y * height)
            radius = max(int(opt.r * width), 6)
            rx, ry, ratio = self._refine_bubble_center(binary_img, cx, cy, radius)
            fill_values.append(ratio)
            coords.append((opt.key, rx, ry, radius, ratio))
            refined_map[opt.key] = (rx, ry, radius)

        fills = np.array(fill_values, dtype=np.float32)
        mean = float(np.mean(fills))
        std = float(np.std(fills))
        std = std if std > 1e-6 else 1e-6

        scores: list[OptionScore] = []
        for key, _, _, _, ratio in coords:
            z = (ratio - mean) / std
            marked = z >= self.mark_z_threshold and ratio >= self.min_fill_ratio
            scores.append(OptionScore(key=key, fill_ratio=ratio, z_score=z, marked=marked))

        return scores, refined_map

    @staticmethod
    def _resolve_field_result(field_type: str, scores: list[OptionScore]) -> dict[str, Any]:
        marked = [s for s in scores if s.marked]
        marked_sorted = sorted(marked, key=lambda s: (s.fill_ratio, s.z_score), reverse=True)

        if field_type == "single_choice":
            if len(marked_sorted) == 1:
                return {"value": marked_sorted[0].key, "status": "ok", "confidence": float(marked_sorted[0].fill_ratio)}
            if len(marked_sorted) == 0:
                return {"value": None, "status": "blank", "confidence": 0.0}
            return {
                "value": ",".join([m.key for m in marked_sorted]),
                "status": "multiple",
                "confidence": float(marked_sorted[0].fill_ratio),
            }

        values = [m.key for m in marked_sorted]
        if not values:
            return {"value": None, "status": "blank", "confidence": 0.0}
        confidence = float(np.mean([m.fill_ratio for m in marked_sorted]))
        return {"value": ",".join(values), "status": "ok", "confidence": confidence}

    def process_page(self, image: np.ndarray, page_idx: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
        enhanced, binary = self.preprocess_page(image)
        page_cfg = self.config.pages.get(page_idx, PageConfig(fields=[], anchors=[]))
        aligned_image, aligned_binary, align_meta = self._align_with_markers(image, binary, page_cfg)

        visual = aligned_image.copy()
        h, w = aligned_binary.shape[:2]
        results: list[dict[str, Any]] = []

        cv2.putText(
            visual,
            f"ALIGN: {align_meta.get('alignment_status')} markers={align_meta.get('detected_markers', 0)}",
            (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if page_cfg.anchors:
            for m in page_cfg.anchors:
                ex, ey = int(m.x * w), int(m.y * h)
                cv2.rectangle(visual, (ex - 8, ey - 8), (ex + 8, ey + 8), (255, 255, 0), 2)
                cv2.putText(visual, m.marker_id, (ex + 10, ey - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        for field in page_cfg.fields:
            scores, refined = self._score_options(aligned_binary, field.options, w, h)
            resolved = self._resolve_field_result(field.field_type, scores)
            chosen = set(str(resolved["value"]).split(",")) if resolved["value"] else set()

            for opt in field.options:
                cx, cy, radius = refined[opt.key]
                is_selected = opt.key in chosen
                color = (0, 200, 0) if is_selected else (120, 120, 120)
                cv2.circle(visual, (cx, cy), radius, color, 2)
                if is_selected:
                    cv2.circle(visual, (cx, cy), max(radius // 3, 2), (0, 200, 0), -1)

            label = f"{field.label}: {resolved['value'] or '-'} [{resolved['status']}]"
            first_opt = field.options[0]
            tx, ty = int(first_opt.x * w) + 18, int(first_opt.y * h) - 10
            cv2.putText(
                visual,
                label,
                (tx, max(ty, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 120, 255) if resolved["status"] != "ok" else (30, 220, 30),
                2,
                cv2.LINE_AA,
            )

            results.append(
                {
                    "field_id": field.field_id,
                    "label": field.label,
                    "value": resolved["value"],
                    "status": resolved["status"],
                    "confidence": round(float(resolved["confidence"]), 4),
                    "alignment_status": align_meta.get("alignment_status"),
                    "detected_markers": align_meta.get("detected_markers", 0),
                }
            )

        _ = enhanced
        return visual, results

    def process_pdf(self, pdf_path: Path, output_dir: Path, dpi: int = 300) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)

        pages = self.pdf_to_images(pdf_path, dpi=dpi)
        all_rows: list[dict[str, Any]] = []
        annotated_images: list[Image.Image] = []

        for idx, page_img in enumerate(pages):
            annotated, rows = self.process_page(page_img, idx)
            for row in rows:
                all_rows.append({"source_pdf": pdf_path.name, "page_index": idx, **row})

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            annotated_images.append(Image.fromarray(rgb))

        stem = pdf_path.stem
        csv_path = output_dir / f"{stem}_results.csv"
        out_pdf = output_dir / f"{stem}_annotated.pdf"

        pd.DataFrame(all_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
        if annotated_images:
            doc = fitz.open()
            for pil_img in annotated_images:
                arr = np.array(pil_img)
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                ok, encoded = cv2.imencode(".png", bgr)
                if not ok:
                    continue
                rect = fitz.Rect(0, 0, pil_img.width, pil_img.height)
                page = doc.new_page(width=pil_img.width, height=pil_img.height)
                page.insert_image(rect, stream=encoded.tobytes())
            doc.save(out_pdf)
            doc.close()

        return csv_path, out_pdf


def load_config(path: Path) -> OMRConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    pages: dict[int, PageConfig] = {}
    for page in raw.get("pages", []):
        idx = int(page["page_index"])

        fields: list[FieldDef] = []
        for fdef in page.get("fields", []):
            options = [
                BubbleOption(key=str(opt["key"]), x=float(opt["x"]), y=float(opt["y"]), r=float(opt["r"]))
                for opt in fdef.get("options", [])
            ]
            fields.append(
                FieldDef(
                    field_id=str(fdef["id"]),
                    label=str(fdef.get("label", fdef["id"])),
                    field_type=str(fdef.get("type", "single_choice")),
                    options=options,
                )
            )

        anchors = [
            MarkerDef(marker_id=str(a.get("id", f"m{i}")), x=float(a["x"]), y=float(a["y"]))
            for i, a in enumerate(page.get("alignment_anchors", []))
        ]

        pages[idx] = PageConfig(
            fields=fields,
            anchors=anchors,
            marker_search_window=float(page.get("marker_search_window", 0.08)),
        )

    return OMRConfig(name=str(raw.get("name", "OMR Template")), pages=pages)


def discover_pdfs(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        return [input_path]
    if input_path.is_dir():
        return sorted([p for p in input_path.glob("*.pdf") if p.is_file()])
    return []


def process_pdf_bytes(
    pdf_bytes: bytes,
    config: OMRConfig,
    dpi: int = 300,
    mark_z_threshold: float = 1.0,
    min_fill_ratio: float = 0.22,
) -> tuple[pd.DataFrame, bytes]:
    """
    Streamlit/web gibi ortamlarda diskten bağımsız kullanım için PDF bytes işleme yardımcı fonksiyonu.

    Dönen değerler:
    - results_df: alan bazlı sonuçlar
    - annotated_pdf_bytes: anotasyonlu PDF bayt içeriği (üretilemezse boş bytes)
    """
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if doc.page_count == 0:
            return pd.DataFrame(), b""

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        input_pdf = tmp_dir / "input.pdf"
        output_dir = tmp_dir / "output"
        input_pdf.write_bytes(pdf_bytes)

        processor = OMRProcessor(
            config=config,
            mark_z_threshold=mark_z_threshold,
            min_fill_ratio=min_fill_ratio,
        )
        csv_path, out_pdf = processor.process_pdf(input_pdf, output_dir, dpi=dpi)

        results_df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
        annotated_pdf_bytes = out_pdf.read_bytes() if out_pdf.exists() else b""
        return results_df, annotated_pdf_bytes


def evaluate_with_answer_key(
    results_df: pd.DataFrame,
    answer_key_df: pd.DataFrame,
    min_success_rate: float = 0.8,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Sonuçları cevap anahtarıyla karşılaştırır ve karar özeti üretir."""
    required_cols = {"page_index", "field_id", "value", "status"}
    if not required_cols.issubset(results_df.columns):
        return results_df.copy(), {
            "total_required": 0,
            "correct_required": 0,
            "success_rate": 0.0,
            "ambiguous_count": 0,
            "decision_status": "KARAR: İNCELEME GEREKİR",
        }

    answer_cols = {"page_index", "field_id", "correct_answer", "required"}
    if not answer_cols.issubset(answer_key_df.columns):
        answer_key_df = answer_key_df.copy()
        answer_key_df["correct_answer"] = ""
        answer_key_df["required"] = True

    merged = results_df.merge(
        answer_key_df[["page_index", "field_id", "correct_answer", "required"]],
        on=["page_index", "field_id"],
        how="left",
    )

    normalized_value = merged["value"].fillna("").astype(str).str.replace(" ", "", regex=False).str.upper()
    normalized_key = merged["correct_answer"].fillna("").astype(str).str.replace(" ", "", regex=False).str.upper()

    merged["is_key_provided"] = normalized_key != ""
    merged["is_correct"] = merged["is_key_provided"] & (normalized_value == normalized_key)
    merged["requires_review"] = merged["status"].isin(["multiple", "blank"]) | (
        ~merged["is_correct"] & merged["is_key_provided"]
    )

    required_mask = merged["required"] == True
    total_required = int(merged[required_mask].shape[0])
    correct_required = int(merged[required_mask & merged["is_correct"]].shape[0])
    ambiguous_count = int(merged[merged["status"].isin(["multiple", "blank"])].shape[0])

    success_rate = (correct_required / total_required) if total_required else 0.0
    decision_status = "KARAR: UYGUN"
    if success_rate < min_success_rate or ambiguous_count > 0:
        decision_status = "KARAR: İNCELEME GEREKİR"

    summary = {
        "total_required": total_required,
        "correct_required": correct_required,
        "success_rate": round(success_rate, 4),
        "ambiguous_count": ambiguous_count,
        "decision_status": decision_status,
    }
    return merged, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PDF optik formları okuyup CSV + anotasyonlu PDF üretir.")
    parser.add_argument("--input", required=True, type=Path, help="Tek bir PDF dosyası veya PDF klasörü")
    parser.add_argument("--template", required=True, type=Path, help="YAML şablon dosyası")
    parser.add_argument("--output", default=Path("output"), type=Path, help="Çıktı klasörü")
    parser.add_argument("--dpi", default=300, type=int, help="PDF render DPI (öneri: 300)")
    parser.add_argument("--mark-z-threshold", default=1.0, type=float, help="İşaret z-score eşiği")
    parser.add_argument("--min-fill-ratio", default=0.22, type=float, help="Minimum doluluk oranı")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s - %(message)s")

    if not args.template.exists():
        LOGGER.error("Şablon bulunamadı: %s", args.template)
        return 2

    pdf_files = discover_pdfs(args.input)
    if not pdf_files:
        LOGGER.error("İşlenecek PDF bulunamadı: %s", args.input)
        return 3

    config = load_config(args.template)
    processor = OMRProcessor(config=config, mark_z_threshold=args.mark_z_threshold, min_fill_ratio=args.min_fill_ratio)

    args.output.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []

    for pdf in pdf_files:
        LOGGER.info("İşleniyor: %s", pdf.name)
        csv_path, out_pdf = processor.process_pdf(pdf, args.output, dpi=args.dpi)
        LOGGER.info("Bitti -> CSV: %s | PDF: %s", csv_path.name, out_pdf.name)
        summary_rows.append({"source_pdf": pdf.name, "result_csv": csv_path.name, "annotated_pdf": out_pdf.name})

    summary_csv = args.output / "batch_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    LOGGER.info("Toplu özet: %s", summary_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

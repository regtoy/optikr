#!/usr/bin/env python3
"""Streamlit arayüzü: PDF yükleme, cevap anahtarı girişi, sonuç ve karar durumu gösterimi."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from omr_professional import OMRProcessor, load_config

st.set_page_config(page_title="Optik Form Değerlendirme", layout="wide")
st.title("📄 Optik Form Değerlendirme Arayüzü")
st.caption("PDF yükle, cevap anahtarı gir, sonucu ve karar/uygunluk durumlarını incele.")


@st.cache_data(show_spinner=False)
def parse_template_fields(template_text: str) -> pd.DataFrame:
    raw = yaml.safe_load(template_text)
    rows: list[dict[str, str]] = []

    for page in raw.get("pages", []):
        page_idx = int(page.get("page_index", 0))
        for fdef in page.get("fields", []):
            options = ", ".join(str(opt.get("key")) for opt in fdef.get("options", []))
            rows.append(
                {
                    "page_index": page_idx,
                    "field_id": str(fdef.get("id")),
                    "label": str(fdef.get("label", fdef.get("id"))),
                    "field_type": str(fdef.get("type", "single_choice")),
                    "options": options,
                }
            )

    return pd.DataFrame(rows).sort_values(["page_index", "field_id"]).reset_index(drop=True)


def run_omr(pdf_bytes: bytes, template_bytes: bytes, dpi: int, z_threshold: float, min_fill_ratio: float) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        pdf_path = tmp_dir / "uploaded.pdf"
        template_path = tmp_dir / "template.yaml"
        out_dir = tmp_dir / "output"

        pdf_path.write_bytes(pdf_bytes)
        template_path.write_bytes(template_bytes)

        config = load_config(template_path)
        processor = OMRProcessor(config=config, mark_z_threshold=z_threshold, min_fill_ratio=min_fill_ratio)
        csv_path, _ = processor.process_pdf(pdf_path=pdf_path, output_dir=out_dir, dpi=dpi)
        return pd.read_csv(csv_path)


def build_answer_key_editor(field_df: pd.DataFrame) -> pd.DataFrame:
    key_df = field_df[["page_index", "field_id", "label", "options"]].copy()
    key_df["correct_answer"] = ""
    key_df["required"] = True

    st.subheader("✍️ Cevap Anahtarı (Elle Giriş)")
    st.write("`correct_answer` sütununa beklenen cevabı girin (örn: A veya A,B).")
    edited = st.data_editor(
        key_df,
        num_rows="fixed",
        width="stretch",
        hide_index=True,
        column_config={
            "page_index": st.column_config.NumberColumn("Sayfa", disabled=True),
            "field_id": st.column_config.TextColumn("Alan ID", disabled=True),
            "label": st.column_config.TextColumn("Etiket", disabled=True),
            "options": st.column_config.TextColumn("Seçenekler", disabled=True),
            "correct_answer": st.column_config.TextColumn("Doğru Cevap"),
            "required": st.column_config.CheckboxColumn("Zorunlu"),
        },
    )
    return edited


def evaluate_decisions(results_df: pd.DataFrame, answer_key_df: pd.DataFrame, min_success_rate: float) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    merged = results_df.merge(answer_key_df[["page_index", "field_id", "correct_answer", "required"]], on=["page_index", "field_id"], how="left")

    normalized_value = merged["value"].fillna("").astype(str).str.replace(" ", "", regex=False).str.upper()
    normalized_key = merged["correct_answer"].fillna("").astype(str).str.replace(" ", "", regex=False).str.upper()

    merged["is_key_provided"] = normalized_key != ""
    merged["is_correct"] = merged["is_key_provided"] & (normalized_value == normalized_key)
    merged["requires_review"] = merged["status"].isin(["multiple", "blank"]) | (~merged["is_correct"] & merged["is_key_provided"])

    total_required = int(merged[merged["required"] == True].shape[0])
    correct_required = int(merged[(merged["required"] == True) & (merged["is_correct"])].shape[0])
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


with st.sidebar:
    st.header("Ayarlar")
    dpi = st.number_input("DPI", min_value=150, max_value=600, value=300, step=10)
    z_threshold = st.slider("Mark Z Threshold", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
    min_fill_ratio = st.slider("Min Fill Ratio", min_value=0.05, max_value=0.9, value=0.22, step=0.01)
    min_success_rate = st.slider("Karar için minimum başarı oranı", min_value=0.0, max_value=1.0, value=0.8, step=0.05)

pdf_file = st.file_uploader("1) PDF yükleyin", type=["pdf"])
template_file = st.file_uploader("2) YAML şablon yükleyin", type=["yaml", "yml"])

if template_file is not None:
    template_text = template_file.getvalue().decode("utf-8")
    fields_df = parse_template_fields(template_text)
    st.subheader("🧩 Şablondan Okunan Alanlar")
    st.dataframe(fields_df, use_container_width=True)

    answer_key_df = build_answer_key_editor(fields_df)

    if pdf_file is not None and st.button("3) Değerlendir", type="primary"):
        with st.spinner("PDF işleniyor..."):
            results_df = run_omr(
                pdf_bytes=pdf_file.getvalue(),
                template_bytes=template_text.encode("utf-8"),
                dpi=int(dpi),
                z_threshold=float(z_threshold),
                min_fill_ratio=float(min_fill_ratio),
            )

        evaluated_df, summary = evaluate_decisions(results_df, answer_key_df, min_success_rate=min_success_rate)

        st.subheader("📊 Karar / Gereksinim Uygunluk Durumu")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Zorunlu Soru", summary["total_required"])
        c2.metric("Doğru (Zorunlu)", summary["correct_required"])
        c3.metric("Başarı Oranı", f"%{summary['success_rate'] * 100:.1f}")
        c4.metric("Belirsiz (blank/multiple)", summary["ambiguous_count"])

        if summary["decision_status"].endswith("UYGUN"):
            st.success(summary["decision_status"])
        else:
            st.warning(summary["decision_status"])

        st.subheader("🧾 Detay Sonuçlar")
        st.dataframe(evaluated_df, use_container_width=True)

        csv_bytes = evaluated_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="Sonuçları CSV indir",
            data=io.BytesIO(csv_bytes),
            file_name="streamlit_omr_results.csv",
            mime="text/csv",
        )
else:
    st.info("Devam etmek için önce YAML şablonunu yükleyin.")

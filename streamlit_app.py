#!/usr/bin/env python3
"""Streamlit arayüzü: PDF yükleme, cevap anahtarı girişi, sonuç ve karar durumu gösterimi."""

from __future__ import annotations

import io
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import streamlit as st
import yaml

from omr_professional import evaluate_with_answer_key, load_config, process_pdf_bytes

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


@st.cache_resource(show_spinner=False)
def load_config_cached(template_text: str):
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "template.yaml"
        path.write_text(template_text, encoding="utf-8")
        return load_config(path)


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


def run_batch(
    pdf_files,
    config,
    dpi: int,
    z_threshold: float,
    min_fill_ratio: float,
    answer_key_df: pd.DataFrame,
    min_success_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame, bytes]:
    all_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, str | float | int]] = []

    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, mode="w", compression=ZIP_DEFLATED) as zf:
        for up in pdf_files:
            raw = up.getvalue()
            base_name = Path(up.name).stem

            result_df, annotated_pdf = process_pdf_bytes(
                pdf_bytes=raw,
                config=config,
                dpi=dpi,
                mark_z_threshold=z_threshold,
                min_fill_ratio=min_fill_ratio,
            )
            if result_df.empty:
                continue

            evaluated_df, decision = evaluate_with_answer_key(
                result_df,
                answer_key_df=answer_key_df,
                min_success_rate=min_success_rate,
            )
            evaluated_df.insert(0, "source_pdf", up.name)
            all_rows.append(evaluated_df)

            summary_rows.append(
                {
                    "source_pdf": up.name,
                    "decision_status": decision["decision_status"],
                    "success_rate": decision["success_rate"],
                    "ambiguous_count": decision["ambiguous_count"],
                    "correct_required": decision["correct_required"],
                    "total_required": decision["total_required"],
                }
            )

            zf.writestr(f"{base_name}_evaluated.csv", evaluated_df.to_csv(index=False).encode("utf-8-sig"))
            if annotated_pdf:
                zf.writestr(f"{base_name}_annotated.pdf", annotated_pdf)

    combined_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    return combined_df, summary_df, zip_buffer.getvalue()


with st.sidebar:
    st.header("Ayarlar")
    dpi = st.number_input("DPI", min_value=150, max_value=600, value=300, step=10)
    z_threshold = st.slider("Mark Z Threshold", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
    min_fill_ratio = st.slider("Min Fill Ratio", min_value=0.05, max_value=0.9, value=0.22, step=0.01)
    min_success_rate = st.slider("Karar için minimum başarı oranı", min_value=0.0, max_value=1.0, value=0.8, step=0.05)

pdf_files = st.file_uploader("1) PDF yükleyin (bir veya birden fazla)", type=["pdf"], accept_multiple_files=True)
template_file = st.file_uploader("2) YAML şablon yükleyin", type=["yaml", "yml"])

if template_file is not None:
    template_text = template_file.getvalue().decode("utf-8")
    fields_df = parse_template_fields(template_text)
    st.subheader("🧩 Şablondan Okunan Alanlar")
    st.dataframe(fields_df, use_container_width=True)

    answer_key_df = build_answer_key_editor(fields_df)
    config = load_config_cached(template_text)

    if pdf_files and st.button("3) Değerlendir", type="primary"):
        with st.spinner("PDF(ler) işleniyor..."):
            combined_df, summary_df, zip_bytes = run_batch(
                pdf_files=pdf_files,
                config=config,
                dpi=int(dpi),
                z_threshold=float(z_threshold),
                min_fill_ratio=float(min_fill_ratio),
                answer_key_df=answer_key_df,
                min_success_rate=float(min_success_rate),
            )

        if summary_df.empty:
            st.error("İşlenecek geçerli bir sonuç üretilemedi.")
        else:
            st.subheader("📊 Karar / Gereksinim Uygunluk Özeti")
            st.dataframe(summary_df, use_container_width=True)

            needs_review = int((summary_df["decision_status"] == "KARAR: İNCELEME GEREKİR").sum())
            approved = int((summary_df["decision_status"] == "KARAR: UYGUN").sum())

            c1, c2, c3 = st.columns(3)
            c1.metric("Toplam PDF", int(summary_df.shape[0]))
            c2.metric("Uygun", approved)
            c3.metric("İnceleme Gerekir", needs_review)

            st.subheader("🧾 Detay Sonuçlar")
            st.dataframe(combined_df, use_container_width=True)

            st.download_button(
                label="Tüm çıktılarını ZIP indir",
                data=zip_bytes,
                file_name="streamlit_omr_outputs.zip",
                mime="application/zip",
            )
else:
    st.info("Devam etmek için önce YAML şablonunu yükleyin.")

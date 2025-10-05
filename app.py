"""Streamlit prototype for comparing object detection models."""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from html import escape
from typing import Any, Callable, Dict, List, Literal

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

ModelSource = Literal["huggingface", "ultralytics"]


@dataclass
class Detection:
    label: str
    score: float
    box: Dict[str, float]


def _load_hf_detector(model_name: str):
    try:
        from transformers import pipeline
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "transformers paketi bulunamadı. Lütfen `pip install -r requirements.txt` komutunu çalıştırın."
        ) from exc

    try:
        return pipeline("object-detection", model=model_name)
    except ImportError as exc:  # pragma: no cover - depends on external deps
        if "timm" in str(exc):
            raise RuntimeError(
                "Seçilen model `timm` paketine ihtiyaç duyuyor. Lütfen sanal ortamınızda `pip install timm` "
                "komutunu çalıştırıp uygulamayı yeniden başlatın."
            ) from exc
        raise RuntimeError(f"Hugging Face modeli yüklenemedi: {exc}") from exc
    except OSError as exc:  # pragma: no cover - depends on external deps
        raise RuntimeError(
            "Seçilen Hugging Face modelinin PyTorch veya safetensors ağırlıkları bulunamadı. "
            "Lütfen model kartında desteklenen formatları kontrol edin veya `facebook/detr-resnet-50` gibi "
            "standart bir model tercih edin."
        ) from exc
    except Exception as exc:  # pragma: no cover - depends on external deps
        raise RuntimeError(f"Hugging Face modeli yüklenemedi: {exc}") from exc


def _load_ultralytics_model(model_name: str):
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ultralytics paketi bulunamadı. Lütfen `pip install -r requirements.txt` komutunu çalıştırın."
        ) from exc

    try:
        return YOLO(model_name)
    except Exception as exc:  # pragma: no cover - depends on external deps
        raise RuntimeError(f"Ultralytics modeli yüklenemedi: {exc}") from exc


@st.cache_resource(show_spinner=False)
def get_hf_detector(model_name: str):
    return _load_hf_detector(model_name)


@st.cache_resource(show_spinner=False)
def get_ultralytics_model(model_name: str):
    return _load_ultralytics_model(model_name)


def run_hf_inference(model_name: str, image: Image.Image) -> tuple[List[Detection], float]:
    detector = get_hf_detector(model_name)
    start = time.perf_counter()
    outputs = detector(image)
    elapsed = (time.perf_counter() - start) * 1000

    detections: List[Detection] = []
    for item in outputs:
        box = item.get("box", {})
        detections.append(
            Detection(
                label=item.get("label", "unknown"),
                score=float(item.get("score", 0.0)),
                box={
                    "xmin": float(box.get("xmin", 0.0)),
                    "ymin": float(box.get("ymin", 0.0)),
                    "xmax": float(box.get("xmax", 0.0)),
                    "ymax": float(box.get("ymax", 0.0)),
                },
            )
        )
    return detections, elapsed


def run_ultralytics_inference(model_name: str, image: Image.Image) -> tuple[List[Detection], float]:
    model = get_ultralytics_model(model_name)
    start = time.perf_counter()
    results = model(image, verbose=False)
    elapsed = (time.perf_counter() - start) * 1000

    detections: List[Detection] = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        xyxy = boxes.xyxy.tolist()
        conf = boxes.conf.tolist()
        classes = boxes.cls.tolist()
        names = result.names or {}

        for idx, coords in enumerate(xyxy):
            xmin, ymin, xmax, ymax = coords
            label_idx = int(classes[idx]) if idx < len(classes) else 0
            detections.append(
                Detection(
                    label=names.get(label_idx, str(label_idx)),
                    score=float(conf[idx]) if idx < len(conf) else 0.0,
                    box={
                        "xmin": float(xmin),
                        "ymin": float(ymin),
                        "xmax": float(xmax),
                        "ymax": float(ymax),
                    },
                )
            )
    return detections, elapsed


def draw_detections(base_image: Image.Image, detections: List[Detection]) -> Image.Image:
    image = base_image.copy()
    draw = ImageDraw.Draw(image)

    for det in detections:
        box = det.box
        color = _color_from_label(det.label)
        draw.rectangle(
            [(box["xmin"], box["ymin"]), (box["xmax"], box["ymax"])],
            outline=color,
            width=3,
        )
        text = f"{det.label} {(det.score * 100):.1f}%"
        text_bbox = draw.textbbox((box["xmin"], box["ymin"]), text)
        offset_y = max(0, box["ymin"] - (text_bbox[3] - text_bbox[1]) - 4)
        background_box = [
            (box["xmin"], offset_y),
            (box["xmin"] + (text_bbox[2] - text_bbox[0]) + 8, offset_y + (text_bbox[3] - text_bbox[1]) + 4),
        ]
        draw.rectangle(background_box, fill=color)
        draw.text((box["xmin"] + 4, offset_y + 2), text, fill="white")

    return image


def _color_from_label(label: str) -> str:
    random.seed(label)
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return f"#{r:02x}{g:02x}{b:02x}"


def setup_page() -> None:
    st.set_page_config(page_title="YOLO Model Comparison Prototype", layout="wide")
    _inject_custom_styles()
    _render_sidebar()
    _render_hero_section()


def _resize_for_display(image: Image.Image, *, max_width: int = 900, max_height: int = 550) -> Image.Image:
    if max_width <= 0 or max_height <= 0:
        return image.copy()

    size = (max_width, max_height)
    try:
        return ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)
    except AttributeError:  # Pillow < 9.1 fallback
        return ImageOps.contain(image, size, method=Image.LANCZOS)


def _trigger_rerun() -> None:
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if callable(rerun):
        rerun()


def _download_hf_repository(repo_id: str, update_progress: Callable[[float], None]) -> None:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "huggingface_hub paketi bulunamadı. Lütfen `pip install huggingface_hub` komutunu çalıştırın."
        ) from exc

    api = HfApi()
    try:
        info = api.model_info(repo_id)
    except Exception as exc:  # pragma: no cover - depends on external service
        raise RuntimeError(f"{repo_id} model bilgisi alınamadı: {exc}") from exc

    siblings = [s for s in info.siblings if getattr(s, "rfilename", None)]
    total = sum((getattr(s, "size", 0) or 0) for s in siblings)
    accumulated = 0
    if total <= 0:
        update_progress(1.0)
        return

    for sibling in siblings:
        size = getattr(sibling, "size", 0) or 0
        if sibling.rfilename.endswith("/"):
            continue
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=sibling.rfilename,
                revision=getattr(info, "sha", None),
                resume_download=True,
                local_files_only=False,
            )
        except Exception as exc:  # pragma: no cover - depends on network
            raise RuntimeError(
                f"{repo_id} deposundaki `{sibling.rfilename}` dosyası indirilemedi: {exc}"
            ) from exc
        accumulated += size
        # Hugging Face bazı dosyalar için boyut bilgisi döndürmez; bu durumda adım atlayabilir
        progress = accumulated / total if total else 1.0
        update_progress(min(progress, 1.0))


def _run_model_download(
    session_key: str,
    model_name: str,
    source: ModelSource,
    progress_bar,
    progress_text,
) -> None:
    def update_progress(fraction: float) -> None:
        percent = max(0, min(100, int(fraction * 100)))
        progress_bar.progress(percent)
        progress_text.caption(f"İndirme ilerlemesi: %{percent}")
        st.session_state["download_progress"] = percent
        st.session_state["download_status"][session_key] = {
            "state": "progress",
            "message": f"İndiriliyor... %{percent}",
        }

    update_progress(0.01)

    if source == "huggingface":
        _download_hf_repository(model_name, update_progress)
        # Modeli belleğe alarak ilk kullanım gecikmesini azaltıyoruz.
        get_hf_detector(model_name)
    else:
        try:
            _download_hf_repository(model_name, update_progress)
        except RuntimeError:
            # Depo Hugging Face üzerinde değilse doğrudan Ultralytics'ten indirilecek.
            update_progress(0.1)
        get_ultralytics_model(model_name)
    update_progress(1.0)


def _inject_custom_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --yolo-primary: #2563eb;
                --yolo-primary-hover: #1d4ed8;
                --yolo-surface: #111827;
                --yolo-muted: #1f2937;
                --yolo-border: rgba(148, 163, 184, 0.25);
                --yolo-text: #e2e8f0;
            }

            section.main > div {
                padding-top: 0;
            }

            body {
                background: radial-gradient(circle at top left, #0f172a, #020617);
            }

            .yolo-hero {
                background: linear-gradient(135deg, rgba(37, 99, 235, 0.65), rgba(15, 118, 110, 0.6));
                border-radius: 24px;
                padding: 2.5rem;
                color: white;
                box-shadow: 0 30px 60px rgba(15, 23, 42, 0.35);
                margin-bottom: 2rem;
                border: 1px solid rgba(148, 163, 184, 0.2);
            }

            .yolo-hero h1 {
                margin: 0;
                font-size: 2.4rem;
            }

            .yolo-hero p {
                font-size: 1rem;
                max-width: 640px;
                margin-top: 0.75rem;
                color: rgba(226, 232, 240, 0.92);
            }

            .yolo-hero__meta {
                display: flex;
                gap: 1rem;
                margin-top: 1.5rem;
                flex-wrap: wrap;
                font-weight: 600;
                font-size: 0.95rem;
            }

            .yolo-chip {
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                padding: 0.2rem 0.75rem;
                border-radius: 999px;
                background: rgba(15, 23, 42, 0.3);
                border: 1px solid rgba(226, 232, 240, 0.25);
                font-size: 0.85rem;
                color: var(--yolo-text);
            }

            .yolo-badge {
                background: rgba(17, 24, 39, 0.55);
                border-radius: 999px;
                padding: 0.25rem 0.75rem;
                font-size: 0.75rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                display: inline-block;
                border: 1px solid rgba(226, 232, 240, 0.35);
                margin-bottom: 1rem;
            }

            .yolo-card {
                background: rgba(15, 23, 42, 0.55);
                border-radius: 18px;
                padding: 1.2rem 1.4rem;
                border: 1px solid var(--yolo-border);
                color: var(--yolo-text);
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.3);
            }

            .yolo-card h4 {
                margin-top: 0;
                margin-bottom: 0.6rem;
                font-size: 1.05rem;
            }

            .yolo-card--hint {
                background: rgba(37, 99, 235, 0.14);
                border-color: rgba(59, 130, 246, 0.35);
            }

            .yolo-card--empty {
                border-style: dashed;
                background: rgba(15, 23, 42, 0.35);
                text-align: center;
                padding: 1.8rem 1.4rem;
            }

            .yolo-card--result {
                background: rgba(15, 23, 42, 0.7);
                border: 1px solid rgba(59, 130, 246, 0.25);
            }

            .yolo-card--success {
                border-color: rgba(34, 197, 94, 0.45);
                background: rgba(34, 197, 94, 0.12);
            }

            .yolo-card--error {
                border-color: rgba(248, 113, 113, 0.45);
                background: rgba(248, 113, 113, 0.12);
            }

            .yolo-card__title {
                font-weight: 600;
                font-size: 1rem;
            }

            .stButton > button {
                background: linear-gradient(135deg, var(--yolo-primary), #14b8a6);
                color: white;
                border: none;
                border-radius: 14px;
                padding: 0.6rem 1.2rem;
                font-weight: 600;
                box-shadow: 0 12px 30px rgba(37, 99, 235, 0.35);
            }

            .stButton > button:hover {
                background: linear-gradient(135deg, var(--yolo-primary-hover), #0d9488);
            }

            .stButton > button:disabled {
                background: rgba(148, 163, 184, 0.18);
                color: rgba(226, 232, 240, 0.55);
                box-shadow: none;
            }

            div[data-testid="stMetricValue"] {
                color: white;
            }

            div[data-testid="stMetric"] {
                background: rgba(15, 23, 42, 0.65);
                border-radius: 16px;
                padding: 1rem;
                border: 1px solid rgba(59, 130, 246, 0.2);
            }

            div[data-testid="stFileUploader"] {
                background: rgba(15, 23, 42, 0.45);
                border-radius: 18px;
                border: 1px dashed rgba(148, 163, 184, 0.45);
                padding: 1.6rem 1.4rem;
            }

            div[data-testid="stFileUploader"] section div {
                color: rgba(226, 232, 240, 0.85);
                font-weight: 500;
            }

            .stTabs [data-baseweb="tab-list"] button {
                color: rgba(226, 232, 240, 0.7);
                padding: 0.6rem 1.1rem;
                border-radius: 12px 12px 0 0;
                background: rgba(15, 23, 42, 0.4);
                border: 1px solid transparent;
                margin-right: 0.35rem;
            }

            .stTabs [aria-selected="true"] {
                color: white !important;
                background: rgba(37, 99, 235, 0.35) !important;
                border-color: rgba(59, 130, 246, 0.3) !important;
            }

            div[data-testid="stDataFrame"] > div:nth-of-type(1) {
                border-radius: 18px 18px 0 0;
                overflow: hidden;
            }

            div[data-testid="StyledFullScreenFrame"] {
                background: rgba(15, 23, 42, 0.95);
            }

            .stDataFrame, div[data-testid="stDataFrame"] {
                border-radius: 18px;
                border: 1px solid rgba(148, 163, 184, 0.25);
            }

            div[data-testid="stDataEditor"] {
                background: rgba(15, 23, 42, 0.55);
                border-radius: 18px;
                border: 1px solid rgba(148, 163, 184, 0.25);
                padding: 0.75rem;
            }

            .stDataEditor div[role="columnheader"] {
                background: rgba(37, 99, 235, 0.18) !important;
                color: rgba(226, 232, 240, 0.9) !important;
            }

            .yolo-result-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1rem;
            }

            .yolo-duration {
                font-size: 0.95rem;
                font-weight: 600;
                color: rgba(226, 232, 240, 0.8);
                border: 1px solid rgba(148, 163, 184, 0.35);
                border-radius: 999px;
                padding: 0.25rem 0.9rem;
            }

            .yolo-model-list {
                display: flex;
                flex-direction: column;
                gap: 0.8rem;
            }

            .yolo-model-card {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 1rem;
                background: rgba(15, 23, 42, 0.45);
                border-radius: 16px;
                border: 1px solid rgba(148, 163, 184, 0.25);
                padding: 1rem 1.25rem;
            }

            .yolo-model-card strong {
                font-size: 1rem;
                color: rgba(226, 232, 240, 0.95);
            }

            .yolo-status {
                margin-top: 0.75rem;
                font-size: 0.9rem;
            }

            .yolo-status--success {
                color: #bbf7d0;
            }

            .yolo-status--error {
                color: #fecaca;
            }

            .yolo-status--info {
                color: rgba(226, 232, 240, 0.75);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar() -> None:
    st.sidebar.markdown("### Hızlı Adımlar")
    st.sidebar.markdown(
        "- Bir görsel yükleyin\n"
        "- Karşılaştırma tablosuna modelleri ekleyin\n"
        "- Gerekirse modelleri önceden indirin\n"
        "- `Karşılaştırmayı Çalıştır` butonuna basın"
    )
    st.sidebar.divider()
    st.sidebar.markdown("### Önerilen Modeller")
    st.sidebar.markdown(
        "- `facebook/detr-resnet-50`\n"
        "- `YOLOv8` ailesi (`ultralytics/yolov8n`, `yolov8s`, ... )\n"
        "- `yolov5s` gibi klasik ağırlıklar"
    )
    st.sidebar.caption(
        "Modeller ilk çağrıda önbelleğe alınır, sonraki çalışmalar daha hızlı gerçekleşir."
    )


def _render_hero_section() -> None:
    st.markdown(
        """
        <div class="yolo-hero">
            <div class="yolo-badge">Python prototip</div>
            <h1>YOLO Model Karşılaştırma</h1>
            <p>Hugging Face ve Ultralytics modellerini aynı görsel üzerinde saniyeler içinde karşılaştırın. 
            Sonuçları görsel olarak inceleyin, tespit skorlarını tablo halinde değerlendirin.</p>
            <div class="yolo-hero__meta">
                <span class="yolo-chip">⚡ Tek tıkla indirme</span>
                <span class="yolo-chip">🖼️ Anlık görsel sonuç</span>
                <span class="yolo-chip">📊 Ayrıntılı skor tablosu</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _result_header_html(model_name: str, source: ModelSource, duration: str, variant: str = "default") -> str:
    classes = "yolo-card yolo-card--result"
    if variant == "error":
        classes += " yolo-card--error"
    model_label = escape(model_name)
    source_label = escape(source.title())
    duration_label = escape(duration)
    return (
        f"<div class=\"{classes}\">"
        f"<div class=\"yolo-result-header\">"
        f"<div>"
        f"<div class=\"yolo-card__title\">{model_label}</div>"
        f"<div class=\"yolo-chip\" style=\"margin-top:0.35rem;\">{source_label}</div>"
        f"</div>"
        f"<div class=\"yolo-duration\">{duration_label}</div>"
        f"</div>"
        f"</div>"
    )


def _chunked(items: List[Any], chunk_size: int) -> List[List[Any]]:
    if chunk_size <= 0:
        return [items]
    return [items[idx : idx + chunk_size] for idx in range(0, len(items), chunk_size)]


def _render_results_grid(results: List[Dict[str, Any]], columns: int = 2) -> None:
    if not results:
        return

    for row in _chunked(results, columns):
        cols = st.columns(len(row), gap="large")
        for col, result in zip(cols, row):
            with col:
                st.markdown(
                    _result_header_html(result["name"], result["source"], f"{result['elapsed']:.0f} ms"),
                    unsafe_allow_html=True,
                )

                width = result.get("display_width")
                if width:
                    st.image(
                        result["image"],
                        caption=f"{result['name']} sonuçları",
                        width=width,
                    )
                else:
                    st.image(
                        result["image"],
                        caption=f"{result['name']} sonuçları",
                        use_column_width=True,
                    )

                st.metric("Süre (ms)", f"{result['elapsed']:.0f}")

                if result["detections"]:
                    st.write("**Detaylı Tespitler**")
                    if result["table"] is not None:
                        st.dataframe(result["table"], hide_index=True)
                else:
                    st.info("Herhangi bir nesne tespit edilmedi.")

def init_models_state() -> pd.DataFrame:
    if "models_df" not in st.session_state:
        st.session_state["models_df"] = pd.DataFrame(
            [
                {"name": "facebook/detr-resnet-50", "source": "huggingface"},
                {"name": "ultralytics/yolov8n", "source": "ultralytics"},
            ]
        )
    return st.session_state["models_df"]


def update_models_state(df: pd.DataFrame) -> None:
    st.session_state["models_df"] = df


def get_local_models(directory: str = "ultralytics") -> List[Dict[str, str]]:
    """Scans a directory for .pt files and returns a list of models."""
    if not os.path.isdir(directory):
        return []

    model_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pt")]
    if not model_paths:
        return []

    return [{"name": path, "source": "ultralytics"} for path in model_paths]


def main() -> None:
    setup_page()
    st.session_state.setdefault("active_download", None)
    st.session_state.setdefault("download_progress", 0)
    st.session_state.setdefault("download_runner_busy", False)

    upload_col, info_col = st.columns([3, 2], gap="large")
    with upload_col:
        st.markdown("#### Görsel Seçimi")
        uploaded_file = st.file_uploader(
            "Karşılaştırmak için bir görsel yükleyin",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
        )
        st.caption("PNG, JPG, JPEG veya WEBP formatları desteklenir.")

    with info_col:
        st.markdown(
            """
            <div class="yolo-card yolo-card--hint">
                <h4>İpucu</h4>
                <ul>
                    <li>En az bir Hugging Face ve bir Ultralytics modeli ekleyip hızlarını kıyaslayın.</li>
                    <li>Modelleri önceden indirerek ilk çalıştırmadaki beklemeyi azaltın.</li>
                    <li>Tabloda istediğiniz kadar satır ekleyebilir veya çoğaltabilirsiniz.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        display_image = _resize_for_display(image, max_width=900, max_height=520)
        st.markdown("#### Yüklenen Görsel")
        st.image(display_image, caption="Karşılaştırılacak görsel", width=display_image.width)
    else:
        image = None
        st.markdown(
            """
            <div class="yolo-card yolo-card--empty">
                <strong>Henüz görsel seçilmedi.</strong><br/>
                Örnek olarak bir sokak, iç mekan veya dron görüntüsü yükleyip modellerin çıktısını karşılaştırın.
            </div>
            """,
            unsafe_allow_html=True,
        )

    models_tab, compare_tab = st.tabs(["Model Yönetimi", "Karşılaştırma"])

    valid_models: List[Dict[str, str]] = []

    with models_tab:
        st.markdown("#### Model Tablosu")
        models_df = init_models_state()
        edited_df = st.data_editor(
            models_df,
            num_rows="dynamic",
            hide_index=True,
            column_config={
                "name": st.column_config.TextColumn(
                    "Model Adı",
                    help="Hugging Face model adı veya Ultralytics ağırlığı",
                ),
                "source": st.column_config.SelectboxColumn(
                    "Kaynak",
                    options=("huggingface", "ultralytics"),
                    help="Modelin hangi kütüphane ile çalıştırılacağını seçin",
                ),
            },
        )
        update_models_state(edited_df)

        valid_models = [
            {"name": row["name"].strip(), "source": row["source"]}
            for _, row in edited_df.dropna(subset=["name"]).iterrows()
            if isinstance(row["name"], str) and row["name"].strip()
        ]

        if not valid_models:
            st.info("Karşılaştırma yapmak için tabloya en az bir model ekleyin.")
        else:
            download_status: Dict[str, Dict[str, str]] = st.session_state.setdefault("download_status", {})
            valid_keys = {f"{model['source']}::{model['name']}" for model in valid_models}
            st.session_state["download_status"] = {
                key: value for key, value in download_status.items() if key in valid_keys
            }
            download_status = st.session_state["download_status"]

            st.markdown("#### Model İndirme / Hazırlık")
            for idx, model in enumerate(valid_models):
                model_name = model["name"]
                source: ModelSource = model["source"]  # type: ignore[assignment]
                session_key = f"{source}::{model_name}"
                status = download_status.get(session_key)
                is_active_download = st.session_state.get("active_download") == session_key
                download_busy = st.session_state.get("download_runner_busy", False)

                status_message = "Henüz indirilmedi."
                status_class = "yolo-status yolo-status--info"
                if status:
                    message = status.get("message", "")
                    state = status.get("state")
                    status_message = message
                    if state == "success":
                        status_class = "yolo-status yolo-status--success"
                    elif state == "error":
                        status_class = "yolo-status yolo-status--error"
                    elif state == "progress":
                        status_class = "yolo-status yolo-status--info"

                col_info, col_action = st.columns([4, 1], gap="large")
                with col_info:
                    model_label = escape(model_name)
                    source_label = escape(source.title())
                    status_label = escape(status_message)
                    st.markdown(
                        f"""
                        <div class="yolo-model-card">
                            <div>
                                <strong>{model_label}</strong>
                                <div class="yolo-chip" style="margin-top:0.35rem;">{source_label}</div>
                            </div>
                            <div class="{status_class}">{status_label}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if is_active_download:
                        progress_container = st.container()
                        progress_value = int(st.session_state.get("download_progress", 0))
                        progress_bar = progress_container.progress(progress_value)
                        progress_text = progress_container.empty()
                        if progress_value <= 0:
                            progress_text.caption("İndirme başlatılıyor...")
                        else:
                            progress_text.caption(f"İndirme ilerlemesi: %{progress_value}")

                        if not download_busy:
                            st.session_state["download_runner_busy"] = True
                            try:
                                _run_model_download(
                                    session_key=session_key,
                                    model_name=model_name,
                                    source=source,
                                    progress_bar=progress_bar,
                                    progress_text=progress_text,
                                )
                            except RuntimeError as exc:
                                st.session_state["download_status"][session_key] = {
                                    "state": "error",
                                    "message": str(exc),
                                }
                            else:
                                st.session_state["download_status"][session_key] = {
                                    "state": "success",
                                    "message": "Model indirildi ve kullanılmaya hazır.",
                                }
                            finally:
                                st.session_state["active_download"] = None
                                st.session_state["download_progress"] = 0
                                st.session_state["download_runner_busy"] = False
                                _trigger_rerun()

                button_label = "Modeli indir"
                if col_action.button(
                    button_label,
                    key=f"download_button_{idx}",
                    use_container_width=True,
                    disabled=st.session_state.get("active_download") is not None,
                ):
                    st.session_state["active_download"] = session_key
                    st.session_state["download_progress"] = 0
                    st.session_state["download_runner_busy"] = False
                    st.session_state["download_status"][session_key] = {
                        "state": "progress",
                        "message": "İndiriliyor... %0",
                    }
                    _trigger_rerun()

    with compare_tab:
        st.markdown("#### Yerel Modellerle Karşılaştırma")
        local_models = get_local_models()

        run_button = st.button(
            "Karşılaştırmayı Çalıştır",
            disabled=image is None or not local_models,
            type="primary",
            use_container_width=True,
        )

        if image is None:
            st.info("Önce bir görsel yükleyin.")
        elif not local_models:
            st.info("`ultralytics/` dizininde karşılaştırılacak yerel model (.pt) bulunamadı.")

        if run_button and image is not None and local_models:
            progress_area = st.container()
            results_data: List[Dict[str, Any]] = []
            for model in local_models:
                model_name = model["name"]
                source: ModelSource = "ultralytics"
                display_name = os.path.basename(model_name)

                status_box = progress_area.container()
                status_box.markdown(
                    _result_header_html(display_name, source, "Hazırlanıyor..."),
                    unsafe_allow_html=True,
                )

                try:
                    with st.spinner(f"{display_name} yükleniyor ve çalıştırılıyor..."):
                        detections, elapsed = run_ultralytics_inference(model_name, image)
                except RuntimeError as exc:
                    status_box.markdown(
                        _result_header_html(display_name, source, "Hata", variant="error"),
                        unsafe_allow_html=True,
                    )
                    status_box.error(str(exc))
                    continue

                status_box.empty()

                annotated = draw_detections(image, detections) if detections else image
                detections_table = None
                if detections:
                    detections_table = pd.DataFrame(
                        [
                            {
                                "label": det.label,
                                "score": f"{det.score:.3f}",
                                "xmin": f"{det.box['xmin']:.1f}",
                                "ymin": f"{det.box['ymin']:.1f}",
                                "xmax": f"{det.box['xmax']:.1f}",
                                "ymax": f"{det.box['ymax']:.1f}",
                            }
                            for det in detections
                        ]
                    )

                results_data.append(
                    {
                        "name": display_name,
                        "source": source,
                        "elapsed": elapsed,
                        "detections": detections,
                        "image": annotated,
                        "display_width": 720,
                        "table": detections_table,
                    }
                )

            if results_data:
                st.markdown("#### Sonuçlar")
                _render_results_grid(results_data)


if __name__ == "__main__":
    main()

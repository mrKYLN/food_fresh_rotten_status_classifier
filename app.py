"""
TazeMi? — Domates Tazelik Analizi
===================================
Fotoğraf yükle → CNN-SOM-SVM modeli Taze / Yenebilir / Çürük sınıflandırır.
Opsiyonel: Claude Vision API Türkçe yorum + tarif önerisi ekler.
"""
import os
import base64
import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# ── Sayfa yapılandırması (ilk olmalı) ──────────────────────────────────────
st.set_page_config(
    page_title="TazeMi? 🍅",
    page_icon="🍅",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Stil ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Genel sayfa */
.block-container { max-width: 780px; padding-top: 2rem; }

/* Sonuç kartı */
.result-card {
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin: 1rem 0;
    border: 2px solid;
}
.result-title {
    font-size: 2rem;
    font-weight: 800;
    margin: 0 0 0.2rem 0;
    letter-spacing: -0.5px;
}
.result-subtitle {
    font-size: 1rem;
    font-weight: 400;
    opacity: 0.75;
    margin: 0;
}

/* Güven barları */
.conf-row {
    display: flex;
    align-items: center;
    margin-bottom: 7px;
    gap: 10px;
}
.conf-label {
    width: 80px;
    font-size: 0.87rem;
    font-weight: 500;
    color: #444;
    flex-shrink: 0;
}
.conf-track {
    flex: 1;
    background: #f0f0f0;
    border-radius: 6px;
    height: 14px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.4s ease;
}
.conf-pct {
    width: 42px;
    text-align: right;
    font-size: 0.85rem;
    color: #555;
    flex-shrink: 0;
}

/* Uyarı bant */
.low-conf-badge {
    background: #fff3cd;
    border: 1px solid #ffc107;
    color: #856404;
    border-radius: 8px;
    padding: 0.5rem 0.9rem;
    font-size: 0.9rem;
    margin-top: 0.8rem;
}

/* Öneri kutusu */
.action-box {
    border-radius: 10px;
    padding: 0.8rem 1.1rem;
    font-size: 0.95rem;
    font-weight: 500;
    margin-top: 0.8rem;
}

/* AI yorum kutusu */
.llm-box {
    background: #f8f9fc;
    border: 1px solid #e2e6f0;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-top: 1.2rem;
}
.llm-header {
    font-size: 0.95rem;
    font-weight: 700;
    color: #555;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Gizle uploader etiketi */
label[data-testid="stFileUploaderLabel"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Model yükleme (cached) ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Model yükleniyor…")
def load_model():
    from src.hybrid_model import HybridCNNSOM
    path = Path("models/hybrid_cnn_som.pkl")
    if not path.exists():
        st.error("Model bulunamadı. `python train_hybrid.py` çalıştırın.")
        st.stop()
    return HybridCNNSOM.load(path)

@st.cache_resource(show_spinner="CNN özellik çıkarıcı yükleniyor…")
def load_extractor():
    from src.feature_extractor import MobileNetV2Extractor
    return MobileNetV2Extractor()

# ── Sabitler ────────────────────────────────────────────────────────────────
CLASSES = ["Taze", "Yenebilir", "Çürük"]
CLASS_CONFIG = {
    "Taze":      {"emoji": "✅", "color": "#1a7a3f", "bg": "#edfaf1",
                  "border": "#27ae60", "label": "TAZE", "sublabel": "Fresh"},
    "Yenebilir": {"emoji": "⚠️", "color": "#8a5c00", "bg": "#fffbeb",
                  "border": "#f0b429", "label": "YENEBİLİR", "sublabel": "Edible"},
    "Çürük":     {"emoji": "❌", "color": "#9b1c1c", "bg": "#fef2f2",
                  "border": "#e53e3e", "label": "ÇÜRÜK", "sublabel": "Rotten"},
}
CONF_COLORS = {
    "Taze": "#27ae60",
    "Yenebilir": "#f0b429",
    "Çürük": "#e53e3e",
}
ACTION_TEXT = {
    "Taze":      ("✅", "#edfaf1", "#1a7a3f",
                  "Taze tüketim için ideal. Hemen yiyebilirsiniz."),
    "Yenebilir": ("⏰", "#fffbeb", "#8a5c00",
                  "Bugün-yarın içinde kullanın. Pişirmek en iyi seçenek."),
    "Çürük":     ("🚫", "#fef2f2", "#9b1c1c",
                  "Tüketilmemeli. Sağlık riski oluşturabilir."),
}
CONFIDENCE_THRESHOLD = 0.70   # altında "düşük güven" uyarısı

CNN_INPUT_SIZE = (224, 224)


# ── Yardımcı fonksiyonlar ──────────────────────────────────────────────────

def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize(CNN_INPUT_SIZE, Image.LANCZOS)
    return np.array(img, dtype=np.float32)[np.newaxis] / 255.0


def classify(pil_img: Image.Image):
    features = load_extractor().extract(preprocess_image(pil_img))
    model = load_model()
    pred  = model.predict(features)[0]
    proba = model.confidence_single(features)   # {class: float}
    return pred, proba


def image_to_b64(pil_img: Image.Image, max_size: int = 1024) -> str:
    img = pil_img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def get_llm_comment(pil_img, pred, proba, api_key) -> str:
    """Claude Haiku Vision ile Türkçe yorum + tarif üret."""
    try:
        import anthropic

        cfg = CLASS_CONFIG[pred]
        conf = proba.get(pred, 0) * 100

        class_guidance = {
            "Taze": (
                "Domatesin taze olduğunu belirt. Renk, doku gibi taze görünen özellikleri say. "
                "Çiğ tüketim önerisi ver (salata, söğüş, sandviç gibi). "
                "Çok kısa tut: 2-3 cümle yeterli."
            ),
            "Yenebilir": (
                "Domatesin olgunlaştığını / yenilebilir ama bir an önce tüketilmesi gerektiğini belirt. "
                "Görselde gördüğün işaretleri (renk değişimi, yumuşama vb.) kısaca say. "
                "Ardından **kesinlikle** aşağıdaki formatta tarif önerileri ver:\n\n"
                "🍳 **Bu domatesle bugün yapabileceklerin:**\n"
                "• [tarif 1 — 1 satır açıklama]\n"
                "• [tarif 2 — 1 satır açıklama]\n"
                "• [tarif 3 — 1 satır açıklama]\n\n"
                "Tarifler Türk/Akdeniz mutfağından olsun (menemen, domates çorbası, şakşuka, "
                "domates sosu, türlü, ezme, cacık, köfte sosu vb.). İsraf vurgusunu yap."
            ),
            "Çürük": (
                "Domatesin çürük olduğunu net şekilde belirt. Görselde gördüğün belirtileri say. "
                "Kesinlikle tüketilmemesi gerektiğini vurgula. "
                "Son olarak 1 cümleyle kompost/gübre ipucu ver. "
                "Çok kısa tut: 3 cümle yeterli."
            ),
        }

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=350,
            system=(
                "Sen bir gıda kalitesi uzmanısın. Kullanıcıya domates tazeliği hakkında "
                "Türkçe, sıcak ve pratik bir yorum yapıyorsun. "
                "Yanıtın tamamen Türkçe olsun. Emoji kullanabilirsin. "
                "Markdown (bold, bullet) kullanabilirsin. "
                "Gereksiz tekrar yapma; kısa ve faydalı ol."
            ),
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_to_b64(pil_img),
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Model bu domatesin **{pred}** ({cfg['sublabel']}) olduğunu söylüyor "
                            f"(güven: %{conf:.0f}). "
                            f"{class_guidance[pred]}"
                        ),
                    },
                ],
            }],
        )
        return response.content[0].text
    except Exception as e:
        return f"_(Yorum alınamadı: {e})_"


# ── Başlık ─────────────────────────────────────────────────────────────────
st.markdown("## 🍅 TazeMi?")
st.caption("Fotoğraf yükle · Yapay zeka analiz eder · Gıda israfını önle")

# API anahtarı (küçük, göze batmayan)
with st.expander("🔑 Claude AI yorumu için API anahtarı (opsiyonel)", expanded=False):
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        placeholder="sk-ant-…",
        label_visibility="collapsed",
    )
    if api_key.strip():
        st.success("AI yorumları aktif ✨", icon="✅")
    else:
        st.info("API anahtarı girilirse Claude görsel analizi yaparak Türkçe yorum ve tarif önerir.")

st.markdown("---")

# ── Fotoğraf yükleme ───────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Domates fotoğrafı yükle",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if not uploaded:
    # Büyük, davetkar upload alanı
    st.markdown("""
    <div style="text-align:center; padding: 3rem 1rem; color: #999;
                border: 2px dashed #ddd; border-radius: 16px; margin-top: 1rem;">
        <div style="font-size: 3.5rem; margin-bottom: 0.5rem">📷</div>
        <div style="font-size: 1.1rem; font-weight: 600; color: #555;">
            Domates fotoğrafı sürükleyin veya tıklayın
        </div>
        <div style="font-size: 0.85rem; margin-top: 0.4rem;">
            JPG · PNG · WEBP desteklenir
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    pil_img = Image.open(uploaded)

    # ── Görsel + Sonuçlar yan yana ─────────────────────────────────────
    col_img, col_res = st.columns([1, 1], gap="large")

    with col_img:
        st.image(pil_img, use_container_width=True)

    with col_res:
        with st.spinner("Analiz ediliyor…"):
            pred, proba = classify(pil_img)

        cfg   = CLASS_CONFIG[pred]
        conf  = proba.get(pred, 0)
        low_conf = conf < CONFIDENCE_THRESHOLD

        # Sonuç kartı
        st.markdown(
            f"""<div class="result-card" style="
                background:{cfg['bg']};
                border-color:{cfg['border']};
                color:{cfg['color']};">
              <p class="result-title">{cfg['emoji']} {cfg['label']}</p>
              <p class="result-subtitle">{cfg['sublabel']} · %{conf*100:.0f} güven</p>
            </div>""",
            unsafe_allow_html=True,
        )

        # Güven barları
        bars_html = ""
        for cls in ["Taze", "Yenebilir", "Çürük"]:
            p     = proba.get(cls, 0)
            color = CONF_COLORS[cls]
            bold  = "font-weight:700;" if cls == pred else ""
            bars_html += f"""
            <div class="conf-row">
              <span class="conf-label" style="{bold}">{cls}</span>
              <div class="conf-track">
                <div class="conf-fill" style="width:{p*100:.1f}%; background:{color};"></div>
              </div>
              <span class="conf-pct">{p*100:.0f}%</span>
            </div>"""
        st.markdown(bars_html, unsafe_allow_html=True)

        # Düşük güven uyarısı
        if low_conf:
            st.markdown(
                f"""<div class="low-conf-badge">
                ⚠️ Güven skoru düşük (%{conf*100:.0f}).
                Daha iyi aydınlatılmış, net bir fotoğraf deneyin.
                </div>""",
                unsafe_allow_html=True,
            )

        # Öneri kutusu
        icon, bg, color, text = ACTION_TEXT[pred]
        st.markdown(
            f"""<div class="action-box" style="background:{bg}; color:{color};">
                {icon} {text}
            </div>""",
            unsafe_allow_html=True,
        )

    # ── LLM yorumu (tam genişlik) ──────────────────────────────────────
    if api_key.strip():
        st.markdown(
            "<div class='llm-box'>"
            "<div class='llm-header'>🤖 Yapay Zeka Yorumu</div>",
            unsafe_allow_html=True,
        )
        with st.spinner("Claude analiz yapıyor…"):
            comment = get_llm_comment(pil_img, pred, proba, api_key.strip())
        st.markdown(comment)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='text-align:center; color:#aaa; font-size:0.9rem; "
            "padding:1rem; margin-top:1rem;'>"
            "🔑 API anahtarı girerseniz Claude tarif önerileri ve detaylı analiz ekler."
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Teknik detaylar (collapsible) ──────────────────────────────────
    with st.expander("🔬 Teknik Detaylar — Model ve Görselleştirmeler"):
        c1, c2 = st.columns(2)
        with c1:
            p = Path("outputs/som_map.png")
            if p.exists():
                st.image(str(p), caption="SOM Nöron Dağılımı", use_container_width=True)
        with c2:
            p = Path("outputs/pca_scatter.png")
            if p.exists():
                st.image(str(p), caption="PCA Özellik Uzayı", use_container_width=True)

        st.caption(
            "**Model:** MobileNetV2 (dondurulmuş, 1280-dim) → StandardScaler → PCA(64) "
            "→ MiniSom(3×1) [topoloji] → SVM-RBF [sınıflandırıcı]  \n"
            "**Test doğruluğu:** %97.9  ·  **F1 (macro):** 0.98  ·  "
            "**Veri seti:** 4.287 domates görseli"
        )

# ── Alt bilgi ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#bbb; font-size:0.78rem; margin-top:3rem; padding-top:1rem; border-top: 1px solid #eee;">
TazeMi? &nbsp;·&nbsp; Hybrid CNN · SOM · SVM &nbsp;|&nbsp; MobileNetV2 + MiniSom + SciKit-Learn &nbsp;|&nbsp; Tez Projesi
</div>
""", unsafe_allow_html=True)

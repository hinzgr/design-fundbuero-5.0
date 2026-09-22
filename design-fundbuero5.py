import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os
import base64

# ── Seiteneinstellungen ────────────────────────────────────────────────────────
st.set_page_config(page_title="Fundbüro", page_icon="🎒", layout="centered")

# ── Logo laden (logo.png im selben Ordner) ────────────────────────────────────
def logo_laden():
    logo_pfad = "logo.png"
    if os.path.exists(logo_pfad):
        with open(logo_pfad, "rb") as f:
            daten = f.read()
        b64 = base64.b64encode(daten).decode()
        return f'<img src="data:image/png;base64,{b64}" style="height:70px; display:block; margin-left:auto;">'
    else:
        return (
            '<div style="text-align:right; font-family:\'IM Fell English\',\'Palatino Linotype\',serif;'
            'font-size:13px; font-weight:bold; color:red; line-height:1.2; font-style:italic;">'
            'Katharineum<br>zu Lübeck<br>'
            '<span style="color:black; font-size:11px; font-style:italic;">TU ES</span>'
            '</div>'
        )

LOGO_HTML = logo_laden()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IM+Fell+English:ital@0;1&display=swap');

/* Hintergrund weiß */
.stApp { background-color: white; }

/* ── Layout: volle Breite auf allen Geräten ── */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    max-width: 100vw !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

/* ── ALLE Texte: IM Fell English kursiv ── */
button, .stButton > button, p, div, span, input, textarea, label {
    font-family: 'IM Fell English', 'Palatino Linotype', serif !important;
    font-style: italic !important;
}

/* ── Startseiten-Buttons: volle Breite und Höhe ── */
.grosser-button {
    margin-bottom: 14px !important;
    width: 100% !important;
}
.grosser-button > button {
    font-family: 'IM Fell English', 'Palatino Linotype', serif !important;
    font-size: clamp(24px, 5vw, 48px) !important;
    font-style: italic !important;
    width: 100% !important;
    min-height: 22vh !important;
    height: auto !important;
    background-color: white !important;
    color: black !important;
    border: 2.5px solid black !important;
    border-radius: 6px !important;
    text-align: left !important;
    padding-left: 5vw !important;
    display: flex !important;
    align-items: center !important;
    box-sizing: border-box !important;
    white-space: normal !important;
    word-break: break-word !important;
}
.grosser-button > button:hover {
    background-color: #f5f5f5 !important;
}

/* ── Fertig-Button türkis ── */
.fertig-button > button {
    font-family: 'IM Fell English', 'Palatino Linotype', serif !important;
    font-size: clamp(18px, 4vw, 32px) !important;
    font-style: italic !important;
    background-color: #3dd6b5 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    width: 100% !important;
    min-height: 10vh !important;
    height: auto !important;
    margin-top: 10px !important;
    box-sizing: border-box !important;
}
.fertig-button > button:hover {
    background-color: #2bbfa0 !important;
}

/* ── Zurück-Button ── */
.zurueck-button > button {
    font-family: 'IM Fell English', 'Palatino Linotype', serif !important;
    font-size: clamp(16px, 3.5vw, 28px) !important;
    font-style: italic !important;
    background-color: white !important;
    color: black !important;
    border: 2px solid black !important;
    border-radius: 4px !important;
    width: 100% !important;
    min-height: 8vh !important;
    height: auto !important;
    margin-top: 12px !important;
    box-sizing: border-box !important;
}

/* ── Eingabefelder ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    font-family: 'IM Fell English', 'Palatino Linotype', serif !important;
    font-style: italic !important;
    font-size: clamp(16px, 3.5vw, 24px) !important;
    border: 2.5px solid black !important;
    border-radius: 4px !important;
    background-color: white !important;
    padding: 18px 16px !important;
    height: 70px !important;
    width: 100% !important;
    box-sizing: border-box !important;
}
.stTextArea > div > div > textarea {
    height: 90px !important;
}

/* ── File-Uploader ── */
.stFileUploader > div {
    border: 2.5px solid black !important;
    border-radius: 4px !important;
    background-color: white !important;
    padding: 10px !important;
    width: 100% !important;
    box-sizing: border-box !important;
}
.stFileUploader label,
.stFileUploader span,
.stFileUploader p {
    font-family: 'IM Fell English', 'Palatino Linotype', serif !important;
    font-style: italic !important;
    font-size: clamp(16px, 3.5vw, 24px) !important;
}

/* ── Fundstück-Karte ── */
.karte-name {
    font-family: 'IM Fell English', 'Palatino Linotype', serif;
    font-size: 19px;
    font-style: italic;
    font-weight: bold;
    text-decoration: underline;
    margin-bottom: 4px;
}
.karte-beschreibung {
    font-family: 'IM Fell English', 'Palatino Linotype', serif;
    font-size: 14px;
    font-style: italic;
    color: #222;
}

/* ── Labels & Streamlit-Elemente ausblenden ── */
.stTextInput label, .stTextArea label { display: none !important; }
hr { display: none; }
#MainMenu, footer, header { visibility: hidden; }
div[data-testid="stVerticalBlock"] > div { margin-bottom: 0px !important; }
</style>
""", unsafe_allow_html=True)

# ── Modell laden ───────────────────────────────────────────────────────────────
@st.cache_resource
def modell_laden():
    model = load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return model, labels

model, labels = modell_laden()

# ── KI: Bild erkennen ─────────────────────────────────────────────────────────
def gegenstand_erkennen(bild: Image.Image):
    bild = bild.convert("RGB").resize((224, 224))
    bild_array = np.asarray(bild, dtype=np.float32)
    bild_array = (bild_array / 127.5) - 1
    bild_array = np.expand_dims(bild_array, axis=0)
    vorhersage = model.predict(bild_array)
    index = np.argmax(vorhersage)
    return labels[index], float(vorhersage[0][index])

# ── Session State ──────────────────────────────────────────────────────────────
if "seite" not in st.session_state:
    st.session_state.seite = "start"
if "fundstueck" not in st.session_state:
    st.session_state.fundstueck = {}

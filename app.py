
import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import joblib
import pandas as pd

# Cargar modelos
modelo_bin = joblib.load("modelo_binario.pkl")
modelo_tipo = joblib.load("modelo_tipo_crimen.pkl")
clases_tipo = joblib.load("clases_tipo_crimen.pkl")

st.set_page_config(page_title="SafePath - Mapa de Crimen", layout="centered")
st.title("🗺️ SafePath: Predicción de Crimen por Ubicación")
st.markdown("Haz clic en el mapa para ver si una zona tiene alta probabilidad de crimen.")

# Crear mapa centrado en Boston
m = folium.Map(location=[42.36, -71.05], zoom_start=13)
m.add_child(folium.LatLngPopup())

# Mostrar mapa y capturar clic
output = st_folium(m, height=500, width=700)

if output.get("last_clicked"):
    lat = output["last_clicked"]["lat"]
    lon = output["last_clicked"]["lng"]
    coords = np.array([[lat, lon]])

    prob_crimen = modelo_bin.predict_proba(coords)[0][1]
    st.markdown(f"🔍 **Probabilidad de crimen en ({lat:.4f}, {lon:.4f})**: **{prob_crimen:.2%}**")

    if prob_crimen > 0.5:
        probs_tipo = modelo_tipo.predict_proba(coords)[0]
        idx = np.argmax(probs_tipo)
        tipo = clases_tipo[idx]

        st.success(f"🧠 Crimen más probable: **{tipo}** ({probs_tipo[idx]:.2%})")

        df_probs = pd.DataFrame({
            "Tipo de Crimen": clases_tipo,
            "Probabilidad": probs_tipo
        }).sort_values(by="Probabilidad", ascending=False)

        st.markdown("### 🔎 Top tipos de crimen")
        st.dataframe(df_probs.head(5))
    else:
        st.success("✅ Zona con baja probabilidad de crimen.")

import streamlit as st
import requests
import numpy as np
import pandas as pd
from scipy.special import comb
import plotly.graph_objects as go

# --- FUNCIÓN DE DATOS CON GOLDAPI.IO ---
@st.cache_data(ttl=3600)
def get_market_data_goldapi():
    api_key = st.secrets["GOLD_API_KEY"]
    headers = {
        "x-access-token": api_key,
        "Content-Type": "application/json"
    }
    
    try:
        # 1. Obtener precio actual (XAU/USD)
        response = requests.get("https://www.goldapi.io/api/XAU/USD", headers=headers)
        data = response.json()
        
        if 'price' in data:
            s_actual = float(data['price'])
            
            # 2. Sigma (Volatilidad)
            # GoldAPI gratuito no da series temporales largas. 
            # Recomendación: Usar un sigma estándar de mercado (~0.15 anualizado) 
            # o pedirlo manualmente si la API no lo provee.
            sigma_anual = 0.16 # Volatilidad implícita promedio del oro
            
            return {
                "S": s_actual,
                "sigma": sigma_anual,
                "provider": "GoldAPI.io"
            }
        else:
            st.error(f"Error de API: {data.get('error', 'Desconocido')}")
            return None
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        return None

# --- CONFIGURACIÓN Y UI ---
st.set_page_config(page_title="Valuador Oro", layout="wide")
st.title("🏆 Valuador de Derivados (GoldAPI)")

data = get_market_data_goldapi()

with st.sidebar:
    st.header("Parámetros")
    if data:
        st.info(f"Proveedor: {data['provider']}")
        S = st.number_input("Precio Spot (S)", value=data['S'])
        # Convertimos sigma anual a la varianza que pide tu modelo
        sigma = st.number_input("Varianza (σ)", value=0.00015, format="%.8f")
    else:
        S = st.number_input("Precio Manual (S)", value=2650.0)
        sigma = st.number_input("Varianza (σ)", value=0.00015, format="%.8f")
    
    T = st.number_input("Plazo T (años)", value=1.0)
    alpha = st.number_input("Alpha (α)", value=1.0)
    beta = st.number_input("Beta (β)", value=0.5)
    delta_val = st.slider("Delta (Δ)", 0.01, 1.0, 0.1)
    strike_k = st.number_input("Strike (K)", value=float(S))
    r = st.number_input("Tasa r (SOFR)", value=0.053)

# --- LÓGICA DE CÁLCULO ---
u = np.exp(alpha * (delta_val ** beta))
d = np.exp((sigma * (delta_val ** beta)) / alpha)
M = int(round(T / delta_val))
p = (np.exp(r * delta_val) - d) / (u - d) if (u - d) != 0 else 0.5

# Cálculo de C(m)
m_values = np.arange(0, M + 1)
c_m_results = []
for m in m_values:
    k = np.arange(0, m + 1)
    prices = (u ** k) * (d ** (m - k)) * S
    payoffs = np.maximum(prices - strike_k, 0)
    probs = comb(m, k) * (p ** k) * ((1 - p) ** (m - k))
    c_m = np.exp(-r * (m * delta_val)) * np.sum(payoffs * probs)
    c_m_results.append(c_m)

# --- GRÁFICO ---
fig = go.Figure(go.Scatter(x=m_values, y=c_m_results, line=dict(color='gold')))
fig.update_layout(title="Precio del Call por Pasos", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

st.metric("Resultado Final C(M)", f"{c_m_results[-1]:.4f}")

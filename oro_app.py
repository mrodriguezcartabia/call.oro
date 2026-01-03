import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.special import comb
from datetime import datetime
import requests

# 1. FUNCIÓN CON CACHÉ
# ttl=86400 significa que el dato se guarda por 24 horas (en segundos)
@st.cache_data(ttl=86400)
def fetch_gold_data():
    ticker_symbol = "GC=F"
    headers = {'User-Agent': 'Mozilla/5.0'}
    session = requests.Session()
    session.headers.update(headers)
    
    try:
        gold = yf.Ticker(ticker_symbol, session=session)
        # Descargamos el último año
        hist = gold.history(period="1y")
        if hist.empty: return None
        
        s_actual = hist['Close'].iloc[-1]
        log_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        sigma = log_returns.var()
        options = gold.options
        
        # Guardamos la hora de la descarga
        last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return {
            "S": s_actual,
            "sigma": sigma,
            "options": options,
            "updated": last_update
        }
    except:
        return None

# --- INICIO DE LA APP ---
st.set_page_config(page_title="Valuador Oro Cache", layout="wide")

# Intentar cargar datos del "depósito" (caché)
data = fetch_gold_data()

with st.sidebar:
    st.header("📦 Estado de los Datos")
    if data:
        st.success(f"Datos cargados desde el caché.")
        st.caption(f"Última actualización: {data['updated']}")
        # Botón para forzar la actualización si se desea
        if st.button("🔄 Actualizar datos de Yahoo ahora"):
            st.cache_data.clear()
            st.rerun()
    else:
        st.error("No se pudo obtener datos. Introduce valores manuales.")

    st.divider()
    
    # --- PARÁMETROS ---
    st.header("⚙️ Parámetros")
    # Si hay datos, los usamos como valor por defecto, si no, usamos ceros
    s_val = data['S'] if data else 2650.0
    sig_val = data['sigma'] if data else 0.00015
    
    S = st.number_input("Precio Oro (S)", value=float(s_val))
    sigma = st.number_input("Varianza (σ)", value=float(sig_val), format="%.6f")
    
    # Plazo T
    if data and data['options']:
        selected_t = st.selectbox("Vencimientos disponibles", data['options'])
        dias = (datetime.strptime(selected_t, '%Y-%m-%d') - datetime.now()).days
        T = max(dias / 365, 0.01)
    else:
        T = st.number_input("Plazo T (años)", value=0.5)

    delta_val = st.slider("Delta (Δ)", 0.01, 1.0, 0.1)
    alpha = st.number_input("Alpha (α)", value=1.0)
    beta = st.number_input("Beta (β)", value=0.5)
    r = st.number_input("Tasa r (SOFR)", value=0.053)
    K = st.number_input("Strike K", value=float(round(S, 2)))

# --- CÁLCULOS MATEMÁTICOS ---
u = np.exp(alpha * (delta_val ** beta))
d = np.exp((sigma * (delta_val ** beta)) / alpha)
M = int(round(T / delta_val))
p = (np.exp(r * delta_val) - d) / (u - d) if (u-d) != 0 else 0.5

# Cálculo de C(m)
m_axis = np.arange(0, M + 1)
c_results = []
for m in m_axis:
    k = np.arange(m + 1)
    prices = (u**k) * (d**(m-k)) * S
    payoffs = np.maximum(prices - K, 0)
    probs = comb(m, k) * (p**k) * ((1-p)**(m-k))
    c_m = np.exp(-r * T) * np.sum(payoffs * probs)
    c_results.append(c_m)

# --- VISUALIZACIÓN ---
st.title("🏆 Valuador de Call sobre Oro")
st.info(f"Modelo configurado con **M = {M}** pasos.")

fig = go.Figure()
fig.add_trace(go.Scatter(x=m_axis, y=c_results, line=dict(color="#FFD700", width=3)))
fig.update_layout(title="Curva de Valor C(m)", xaxis_title="Pasos m", yaxis_title="Precio del Call", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
col1.metric("Precio Final Call C(M)", f"${c_results[-1]:.2f}")
col2.metric("Probabilidad p", f"{p:.4f}")

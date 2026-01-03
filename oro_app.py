import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.special import comb
from datetime import datetime
import requests

# 1. FUNCIÓN DE CACHÉ CORREGIDA
@st.cache_data(ttl=86400)
def fetch_gold_data_clean():
    ticker_symbol = "GC=F"
    headers = {'User-Agent': 'Mozilla/5.0'}
    session = requests.Session()
    session.headers.update(headers)
    
    try:
        gold = yf.Ticker(ticker_symbol, session=session)
        hist = gold.history(period="1y")
        
        if hist.empty:
            return None
        
        # Extraemos solo datos primitivos (números y listas de strings)
        s_actual = float(hist['Close'].iloc[-1])
        log_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        sigma_val = float(log_returns.var())
        
        # Guardamos las fechas como una lista simple de strings
        options_list = list(gold.options)
        
        return {
            "S": s_actual,
            "sigma": sigma_val,
            "options": options_list,
            "updated": datetime.now().strftime("%H:%M:%S")
        }
    except Exception as e:
        return None

# --- INTERFAZ ---
st.set_page_config(page_title="Valuador Oro", layout="wide")
st.title("🏆 Valuador de Call sobre Oro")

# Intentar obtener datos
data = fetch_gold_data_clean()

with st.sidebar:
    st.header("📊 Datos de Mercado")
    
    if data:
        st.success(f"Datos actualizados a las {data['updated']}")
        if st.button("🔄 Forzar actualización"):
            st.cache_data.clear()
            st.rerun()
            
        S = st.number_input("Precio Oro (S)", value=data['S'])
        sigma = st.number_input("Varianza (σ)", value=data['sigma'], format="%.6f")
        
        if data['options']:
            sel_date = st.selectbox("Fecha vencimiento (T)", data['options'])
            dias = (datetime.strptime(sel_date, '%Y-%m-%d') - datetime.now()).days
            T = max(dias / 365, 0.01)
        else:
            T = st.number_input("Plazo T (años)", value=0.5)
    else:
        st.warning("⚠️ Usando modo manual (Yahoo no disponible)")
        S = st.number_input("Precio Oro (S)", value=2650.0)
        sigma = st.number_input("Varianza (σ)", value=0.00015, format="%.6f")
        T = st.number_input("Plazo T (años)", value=0.5)

    st.divider()
    st.header("⚙️ Parámetros Modelo")
    delta_val = st.slider("Delta (Δ)", 0.01, 1.0, 0.1)
    alpha = st.number_input("Alpha (α)", value=1.0)
    beta = st.number_input("Beta (β)", value=0.5)
    r = st.number_input("Tasa r (SOFR)", value=0.053)
    K = st.number_input("Strike K", value=float(round(S, 2)))

# --- CÁLCULOS (Pasos 4 al 9) ---
u = np.exp(alpha * (delta_val ** beta))
d = np.exp((sigma * (delta_val ** beta)) / alpha)
M = int(round(T / delta_val))

# Evitar división por cero en p
p = (np.exp(r * delta_val) - d) / (u - d) if (u - d) != 0 else 0.5

# Cálculo de C(m) para el gráfico
m_axis = np.arange(0, M + 1)
c_results = []

for m in m_axis:
    k = np.arange(m + 1)
    # S(m,k) = u^k * d^(m-k) * S
    prices_at_m = (u**k) * (d**(m-k)) * S
    # C(x) = max(x - K, 0)
    payoffs = np.maximum(prices_at_m - K, 0)
    # Sumatoria probabilística
    probs = comb(m, k) * (p**k) * ((1-p)**(m-k))
    c_m = np.exp(-r * T) * np.sum(payoffs * probs)
    c_results.append(c_m)

# --- GRÁFICO (Paso 10) ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=m_axis, y=c_results, line=dict(color="#FFD700", width=3), name="C(m)"))
fig.update_layout(
    title=f"Convergencia del Precio del Call (M={M})",
    xaxis_title="Número de subintervalos (m)",
    yaxis_title="Precio del Derivado C(m)",
    template="plotly_dark"
)
st.plotly_chart(fig, use_container_width=True)

# Métricas finales
c1, c2, c3 = st.columns(3)
c1.metric("Precio Final C(M)", f"${c_results[-1]:.2f}")
c2.metric("Factor Subida (u)", f"{u:.4f}")
c3.metric("Factor Bajada (d)", f"{d:.4f}")

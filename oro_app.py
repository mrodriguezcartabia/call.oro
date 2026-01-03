import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.special import comb
from datetime import datetime
import requests

# --- 1. FUNCIÓN DE DATOS (SOLO DEVUELVE TIPOS SIMPLES) ---
@st.cache_data(ttl=3600)
def get_clean_market_data():
    ticker_symbol = "GC=F"
    try:
        # Configuración para evitar bloqueos
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0'})
        
        gold = yf.Ticker(ticker_symbol, session=session)
        hist = gold.history(period="1y")
        
        if hist.empty:
            return None
        
        # Extraemos solo datos básicos (floats y listas de strings)
        s_actual = float(hist['Close'].iloc[-1])
        # Calculamos la varianza muestral (sigma)
        returns = hist['Close'].pct_change().dropna()
        sigma_val = float(returns.var())
        
        # Lista simple de fechas de opciones
        options_list = list(gold.options)
        
        return {
            "S": s_actual,
            "sigma": sigma_val,
            "options": options_list
        }
    except Exception:
        return None

# --- 2. CONFIGURACIÓN DE LA APP ---
st.set_page_config(page_title="Calculador Call Oro", layout="wide")
st.title("🏆 Valuador de Derivados: Oro (GC=F)")

# Intentamos obtener los datos de la función limpia
market_data = get_clean_market_data()

# --- 3. BARRA LATERAL (Inputs) ---
with st.sidebar:
    st.header("Configuración")
    
    if market_data:
        st.success("Datos de Yahoo Finance cargados")
        S = st.number_input("Precio actual (S)", value=market_data['S'])
        sigma = st.number_input("Varianza (σ)", value=market_data['sigma'], format="%.8f")
        
        if market_data['options']:
            selected_expiry = st.selectbox("Elegir vencimiento (T)", market_data['options'])
            expiry_date = datetime.strptime(selected_expiry, '%Y-%m-%d')
            days_to_expiry = (expiry_date - datetime.now()).days
            T = max(days_to_expiry / 365, 0.001)
        else:
            T = st.number_input("Plazo T (años)", value=1.0)
    else:
        st.error("No se pudo conectar con Yahoo. Usando modo manual.")
        S = st.number_input("Precio actual (S)", value=2650.0)
        sigma = st.number_input("Varianza (σ)", value=0.00015, format="%.8f")
        T = st.number_input("Plazo T (años)", value=1.0)

    st.divider()
    alpha = st.number_input("Alpha (α)", value=1.0)
    beta = st.number_input("Beta (β)", value=0.5)
    delta_val = st.slider("Constante Delta (Δ)", 0.01, 1.0, 0.1)
    strike_k = st.number_input("Precio de ejercicio (K)", value=float(round(S, 2)))
    
    # Tasa SOFR (Punto 6)
    r = st.number_input("Tasa anual r (SOFR)", value=0.053)

# --- 4. CÁLCULOS (Pasos 4 al 9) ---
# Paso 4: u y d
u = np.exp(alpha * (delta_val ** beta))
d = np.exp((sigma * (delta_val ** beta)) / alpha)

# Paso 5: M entero más cercano a T/Delta
M = int(round(T / delta_val))

# Paso 7: p y función C(x)
p = (np.exp(r * delta_val) - d) / (u - d) if (u - d) != 0 else 0.5

# Pasos 8 y 9: Cálculo de C(m) para el gráfico
m_values = np.arange(0, M + 1)
c_m_results = []

for m in m_values:
    k_indices = np.arange(0, m + 1)
    # S(m,k) = u^k * d^(m-k) * S
    prices_at_m = (u ** k_indices) * (d ** (m - k_indices)) * S
    # C(x) = max(x - K, 0)
    payoffs = np.maximum(prices_at_m - strike_k, 0)
    # Probabilidades binomiales
    probs = comb(m, k_indices) * (p ** k_indices) * ((1 - p) ** (m - k_indices))
    # Sumatoria C(m)
    c_m = np.exp(-r * T) * np.sum(payoffs * probs)
    c_m_results.append(c_m)

# --- 5. GRÁFICO (Paso 10) ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=m_values, y=c_m_results, mode='lines+markers', name='C(m)', line=dict(color='gold')))
fig.update_layout(
    title=f"Gráfico de C(m) con M = {M}",
    xaxis_title="m (Pasos)",
    yaxis_title="Precio del Call C(m)",
    template="plotly_dark"
)
st.plotly_chart(fig, use_container_width=True)

# Resultados adicionales
st.write(f"**Resultado Final C(M):** {c_m_results[-1]:.4f}")

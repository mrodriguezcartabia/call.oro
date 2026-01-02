import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.special import comb
from datetime import datetime

# Configuración de la página
st.set_page_config(page_title="Valuador de Call Oro", layout="wide")

st.title("🏆 Valuador de Opciones Call sobre Oro (GC=F)")
st.markdown("""
Esta aplicación calcula el precio de un derivado financiero basado en un modelo binomial personalizado, 
utilizando datos en tiempo real de Yahoo Finance.
""")

# --- PASO 1: Obtención de Datos y Sigma ---
@st.cache_data(ttl=3600)
def load_market_data():
    ticker = "GC=F"
    gold = yf.Ticker(ticker)
    hist = gold.history(period="1y")
    
    if hist.empty:
        return None, None, None, None
    
    # Precio actual (S)
    s_actual = hist['Close'].iloc[-1]
    
    # Varianza muestral del último año (sigma) de los retornos logarítmicos
    log_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
    sigma = log_returns.var()
    
    # Fechas de opciones disponibles
    expiry_dates = gold.options
    
    return s_actual, sigma, expiry_dates, gold

S, sigma, options_dates, gold_obj = load_market_data()

if S is None:
    st.error("No se pudieron cargar los datos del Oro. Intenta más tarde.")
    st.stop()

# --- PASO 2: Entradas del Usuario (Barra Lateral) ---
with st.sidebar:
    st.header("Configuración de Parámetros")
    
    # Constante Delta con barra interactiva
    delta_val = st.slider("Selecciona Delta ($\Delta$)", 0.01, 1.0, 0.1, step=0.01)
    
    # Alpha y Beta
    alpha = st.number_input("Valor de Alpha (α)", value=1.0, step=0.1)
    beta = st.number_input("Valor de Beta (β)", value=0.5, step=0.1)
    
    st.divider()
    
    # Selección de Plazo T
    if options_dates:
        selected_expiry = st.selectbox("Selecciona fecha de vencimiento (T)", options_dates)
        # Calcular T en años
        days_diff = (datetime.strptime(selected_expiry, '%Y-%m-%d') - datetime.now()).days
        T = max(days_diff / 365, 0.01)
    else:
        T = st.number_input("Plazo T (en años)", value=1.0)
    
    # Strike K
    strike_k = st.number_input("Precio de Ejercicio (Strike K)", value=float(round(S, 2)))

# --- PASO 6: Tasa SOFR ---
# Como la web de la Fed puede bloquear scraping, usamos una aproximación dinámica 
# basada en el Treasury Bill a 13 semanas o un input manual.
st.sidebar.divider()
r_input = st.sidebar.number_input("Tasa Anual Continua (r) - Ej. SOFR", value=0.053, format="%.4f")

# --- PASOS 4, 5, 7, 8 y 9: Cálculos Matemáticos ---

# Fórmulas del modelo
u = np.exp(alpha * (delta_val ** beta))
d = np.exp(sigma * (delta_val ** beta) / alpha)
M = int(round(T / delta_val))
p = (np.exp(r_input * delta_val) - d) / (u - d)

def calculate_cm(m_val, S_val, K_val, u_val, d_val, p_val, r_val, T_val):
    # Generar array de k de 0 a m
    k = np.arange(m_val + 1)
    # S(m,k) = u^k * d^(m-k) * S
    prices_at_m = (u_val ** k) * (d_val ** (m_val - k)) * S_val
    # C(S) = max(S - K, 0)
    payoffs = np.maximum(prices_at_m - K_val, 0)
    # Binomial sum: (m choose k) * p^k * (1-p)^(m-k) * payoff
    probabilities = comb(m_val, k) * (p_val ** k) * ((1 - p_val) ** (m_val - k))
    c_m = np.exp(-r_val * T_val) * np.sum(payoffs * probabilities)
    return c_m

# --- PASO 10: Generar datos para el gráfico ---
m_axis = np.arange(0, M + 1)
c_values = [calculate_cm(m, S, strike_k, u, d, p, r_input, T) for m in m_axis]

# --- Visualización de Resultados ---
col1, col2, col3 = st.columns(3)
col1.metric("Precio Oro (S)", f"${S:,.2f}")
col2.metric("Varianza (σ)", f"{sigma:.6f}")
col3.metric("Pasos Totales (M)", M)

# Gráfico con Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=m_axis, y=c_values, mode='lines+markers', name='C(m)',
                         line=dict(color='#FFD700', width=2)))

fig.update_layout(
    title=f"Evolución de C(m) para M = {M}",
    xaxis_title="Número de pasos (m)",
    yaxis_title="Precio del Call C(m)",
    template="plotly_dark",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# Mostrar tabla de datos opcional
if st.checkbox("Ver tabla de valores C(m)"):
    df_results = pd.DataFrame({"m": m_axis, "C(m)": c_values})
    st.dataframe(df_results, use_container_width=True)

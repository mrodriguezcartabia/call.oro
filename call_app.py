import streamlit as st
import requests
import pandas as pd
import pandas_market_calendars as mcal
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.special import comb

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Valuador de Opciones Oro (Binomial)", layout="wide")

# --- FUNCIONES DE OBTENCIÓN DE DATOS ---
@st.cache_data(ttl=3600)
def get_market_data_goldapi():
    try:
        api_key = st.secrets["GOLD_API_KEY"]
        headers = {
            "x-access-token": api_key,
            "Content-Type": "application/json"
        }
        response = requests.get("https://www.goldapi.io/api/XAU/USD", headers=headers)
        data = response.json()
        if 'price' in data:
            return {"S": float(data['price']), "sigma": 0.16}
        else:
            return None
    except Exception:
        return None

@st.cache_data(ttl=3600)
def get_fred_risk_free_rate():
    try:
        api_key = st.secrets["FRED_API_KEY"]
        url = f"https://api.stlouisfed.org/api/series/observations?series_id=DTB4WK&api_key={api_key}&file_type=json&sort_order=desc&limit=1"
        response = requests.get(url)
        data = response.json()
        rate_percent = float(data['observations'][0]['value'])
        return rate_percent / 100
    except Exception:
        return 0.0425  # Tasa de respaldo

@st.cache_data(ttl=86400)
def fecha_vencimiento_oro(year, month):
    try:
        cme = mcal.get_calendar('CME_Total')
        target_month = month - 1 if month > 1 else 12
        target_year = year if month > 1 else year - 1
        last_day = pd.Period(f"{target_year}-{target_month}").to_timestamp(how='end')
        schedule = cme.schedule(start_date=f"{target_year}-{target_month}-01", end_date=last_day)
        vencimiento = schedule.iloc[-4].name 
        return vencimiento.date()
    except Exception:
        return datetime(year, month, 25).date()

# --- MOTOR DE CÁLCULO ---
def calcular_call_crr(S, K, r, T, sigma, beta, paso):
    m = int(round(T / paso))
    if m <= 0: m = 1
    dt = T / m
    u = np.exp((T**0.5) * sigma * (paso**beta))
    d = 1 / u
    a = np.exp(r * dt)
    p = (a - d) / (u - d)
    p = max(min(p, 1.0), 0.0)
    
    suma_binomial = 0
    for k in range(m + 1):
        prob = comb(m, k) * (p**k) * ((1-p)**(m-k))
        st_k = S * (u**(2*k - m))
        payoff = max(st_k - K, 0)
        suma_binomial += prob * payoff
    return np.exp(-r * T) * suma_binomial

# --- LÓGICA DE ESTADO DE SESIÓN ---

VALOR_PASO_ORIGINAL = 0.1

if 'paso_val' not in st.session_state:
    st.session_state.paso_val = VALOR_PASO_ORIGINAL

if 'market_cache' not in st.session_state:
    with st.spinner('Conectando con APIs financieras...'):
        st.session_state.market_cache = get_market_data_goldapi()
        st.session_state.tasa_cache = get_fred_risk_free_rate()

# --- INTERFAZ ---

st.title("📱 App de Valuación de Call sobre Oro")
st.markdown("---")

col_inputs, col_graph = st.columns([1, 1.5]) # Dividimos para que el gráfico esté al lado o arriba

with col_inputs:
    st.subheader("Configuración")
    col1, col2 = st.columns(2)
    
    with col1:
        beta = st.number_input("Beta", value=0.5, step=0.01)
        sigma_default = st.session_state.market_cache['sigma'] if st.session_state.market_cache else 0.16
        sigma = st.number_input("Sigma", value=sigma_default, format="%.2f")
        s_default = st.session_state.market_cache['S'] if st.session_state.market_cache else 2000.0
        precio_s = st.number_input("Precio (S)", value=s_default, format="%.2f")

    with col2:
        strike_init = round(precio_s / 5) * 5
        strike_k_input = st.number_input("Strike (K)", value=float(strike_init), step=5.0)
        tasa_r = st.number_input("Tasa (r)", value=st.session_state.tasa_cache, format="%.4f")
        
        paso_actual = st.number_input("Paso", value=st.session_state.paso_val, format="%.6f")
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            if st.button("x10⁻¹"):
                st.session_state.paso_val = paso_actual * 0.1
                st.rerun()
        with c_btn2:
            if st.button("Reset Paso"):
                st.session_state.paso_val = VALOR_PASO_ORIGINAL
                st.rerun()

    if st.button("RECALCULAR", type="primary", use_container_width=True):
        st.rerun() # Al ser Streamlit, el rerun basta para actualizar con los inputs actuales

# Cálculo de T
hoy = datetime.now()
mes_contrato = hoy.month + 1 if hoy.month < 12 else 1
anio_contrato = hoy.year if hoy.month < 12 else hoy.year + 1
vencimiento_dt = fecha_vencimiento_oro(anio_contrato, mes_contrato)
T = (vencimiento_dt - hoy.date()).days / 365.0

st.sidebar.info(f"📅 Vencimiento: {vencimiento_dt}\nT: {T:.4f} años")

# --- GENERACIÓN DE RESULTADOS (Siempre visible) ---

rango_strikes = np.arange(strike_k_input - 15, strike_k_input + 20, 5)
resultados_c = []

for k_val in rango_strikes:
    c_val = calcular_call_crr(precio_s, k_val, tasa_r, T, sigma, beta, paso_actual)
    resultados_c.append(c_val)

df_res = pd.DataFrame({
    'Strike (K)': rango_strikes,
    'Precio Call (C)': resultados_c
})

with col_graph:
    st.subheader("Análisis de Sensibilidad")
    # Gráfico más pequeño ajustando el figsize
    fig, ax = plt.subplots(figsize=(7, 3.5)) 
    ax.plot(df_res['Strike (K)'], df_res['Precio Call (C)'], marker='o', color='#DAA520', linewidth=2)
    ax.fill_between(df_res['Strike (K)'], df_res['Precio Call (C)'], alpha=0.1, color='#DAA520')
    ax.set_xlabel("Strike (K)")
    ax.set_ylabel("Precio Call (C)")
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

st.markdown("---")
st.table(df_res.style.format({'Precio Call (C)': '{:.2f}'}))

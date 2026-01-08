import streamlit as st
import requests
import pandas as pd
import pandas_market_calendars as mcal
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.special import comb

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Valuador de call de oro", layout="wide")

# --- FUNCIONES DE OBTENCIÓN DE DATOS ---
@st.cache_data(ttl=3600)
def get_market_data_goldapi():
    try:
        api_key = st.secrets["GOLD_API_KEY"]
        headers = {"x-access-token": api_key, "Content-Type": "application/json"}
        response = requests.get("https://www.goldapi.io/api/XAU/USD", headers=headers)
        data = response.json()
        if 'price' in data:
            return float(data['price'])
    except:
        return 4000 #mensaje alerta

@st.cache_data(ttl=86400)
def fecha_vencimiento_oro(year, month):
    try:
        cme = mcal.get_calendar('CME_Total')
        last_day = pd.Period(f"{year}-{month}").to_timestamp(how='end')
        schedule = cme.schedule(start_date=f"{year}-{month}-01", end_date=last_day)
        return schedule.iloc[-4].name.date()
    except:
        return datetime(year, month, 25).date() #agregar alerta

@st.cache_data(ttl=3600)
def get_fred_risk_free_rate():
    try:
        api_key = st.secrets["FRED_API_KEY"]
        url = f"https://api.stlouisfed.org/api/series/observations?series_id=DTB4WK&api_key={api_key}&file_type=json&sort_order=desc&limit=1"
        response = requests.get(url)
        data = response.json()
        return float(data['observations'][0]['value']) / 100
    except:
        return 0.0425

# --- FECHA DE VENCIMIENTO (para mes siguiente debemos ver el mes actual) --- 
hoy = datetime.now()
candidato1 = fecha_vencimiento_oro(hoy.year, hoy.month)
if hoy.date() < candidato1:
    vencimiento = candidato1
else:
    mes_mas_uno = hoy.month + 1 if hoy.month < 12 else 1
    anio = hoy.year if hoy.month < 12 else hoy.year + 1
    vencimiento = fecha_vencimiento_oro(anio, mes_mas_uno)

# --- MOTOR DE CÁLCULO ---
@st.cache_data
def calcular_call(S, K, r, T, sigma, beta, paso, param_a):
    m = int(round(T / paso))
    if m <= 0: m = 1
    dt = T / m
    u = np.exp(param_a * (T**0.5) * sigma * (paso**beta))
    d = u**(-1/param_a**2)
    tasa = np.exp(r * dt)
    p = (tasa - d) / (u - d)
    p = max(min(p, 1.0), 0.0)
    suma_binomial = 0
    for k in range(m + 1):
        prob = comb(m, k) * (p**k) * ((1-p)**(m-k))
        st_k = S * (u**(2*k - m))
        payoff = max(st_k - K, 0)
        suma_binomial += prob * payoff
    return np.exp(-r * T) * suma_binomial

# --- ESTADO DE SESIÓN ---
VALOR_PASO_ORIGINAL = 0.1

if 'paso_val' not in st.session_state:
    st.session_state.paso_val = VALOR_PASO_ORIGINAL
if 'market_cache' not in st.session_state:
    st.session_state.market_cache = get_market_data_goldapi()
    st.session_state.tasa_cache = get_fred_risk_free_rate()
if 'data_grafico' not in st.session_state:
    st.session_state.data_grafico = None

# --- INTERFAZ ---
dias = (vencimiento - hoy.date()).days 
T = dias/ 365.0
precio_s = st.session_state.market_cache
strike = round(precio_s / 5) * 5

col1, col2 = st.columns(2)
with col1:
    beta = st.number_input("Beta", value=0.5, step=0.01)
    st.caption("ℹ️ Este valor se corresponde con el modelo de Black-Scholes")
    sigma_def = 0.16
    sigma = st.number_input("Sigma", value=sigma_def, format="%.2f")
    st.caption("ℹ️ volatilidad: valor conservador basado en datos pasados")

with col2:
    param_a_def = 1.0
    param_a = st.number_input("a", value=param_a_def, step=0.01)
    st.caption("ℹ️ valor de a")
    # strike_init = round(precio_s / 5) * 5
    # strike_k_input = st.number_input("Strike", value=float(strike_init), step=5.0)
    # st.caption("ℹ️ at the money")
    #s_def = st.session_state.market_cache['S'] if st.session_state.market_cache else 2000.0
    #precio_s = st.number_input("Precio", value=s_def, format="%.2f")
    st.caption("ℹ️ datos tomados de GoldAPI")
    tasa_r = st.number_input("Tasa", value=st.session_state.tasa_cache, format="%.4f")
    st.caption("ℹ️ fuente: FRED")
    # st.text_input("Paso", value=f"{st.session_state.paso_val:.8f}", disabled=True)
    # st.caption("ℹ️ use los botones inferiores para modificar el paso")

st.info(f" Vencimiento en {dias} días ({vencimiento})")

# Botones de control
c_p1, c_p2, c_rec = st.columns([1, 1, 2])
with c_p1:
    if st.button("x10⁻¹"):
        st.session_state.paso_val *= 0.1
        st.rerun()
with c_p2:
    if st.button("Reset Paso"):
        st.session_state.paso_val = VALOR_PASO_ORIGINAL
        st.rerun()
with c_rec:
    btn_recalcular = st.button("RECALCULAR", type="primary", use_container_width=True)

# --- LÓGICA DE CÁLCULO BAJO DEMANDA ---

if st.session_state.data_grafico is None or btn_recalcular:
    # Indicador de carga activo durante el proceso matemático
    with st.spinner('Ejecutando modelo binomial...'):
        rango_strikes = np.arange(strike - 35, strike + 40, 5)
        valores_c = []
        for k in rango_strikes:
            c = calcular_call(precio_s, k, tasa_r, T, sigma, beta, st.session_state.paso_val, param_a)
            valores_c.append(c)
        st.session_state.data_grafico = (rango_strikes, valores_c)
    # Mensaje temporal de éxito
    if btn_recalcular:
        st.toast("¡Cálculo finalizado!", icon="✅")

# --- GRÁFICO ---
strikes, calls = st.session_state.data_grafico

st.subheader("Gráfico de Precio de Call (C) vs Strike (K)")
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(strikes, calls, marker='o', color='#DAA520', linewidth=2)
ax.fill_between(strikes, calls, alpha=0.1, color='#DAA520')
ax.set_xlabel("Strike (K)")
ax.set_ylabel("Precio de la Opción (C)")
ax.grid(True, linestyle='--', alpha=0.6)
st.pyplot(fig)

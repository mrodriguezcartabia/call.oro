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

def get_market_data_goldapi():
    """Obtiene el precio actual del oro (XAU/USD) desde GoldAPI."""
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

def get_fred_risk_free_rate():
    """Obtiene la tasa libre de riesgo desde FRED."""
    try:
        api_key = st.secrets["FRED_API_KEY"]
        url = f"https://api.stlouisfed.org/api/series/observations?series_id=DTB4WK&api_key={api_key}&file_type=json&sort_order=desc&limit=1"
        response = requests.get(url)
        data = response.json()
        rate_percent = float(data['observations'][0]['value'])
        return rate_percent / 100
    except Exception:
        return 0.0425  # Tasa de respaldo

def fecha_vencimiento_oro(year, month):
    """Calcula el 4to día hábil antes del fin de mes (Regla CME para Oro)."""
    try:
        cme = mcal.get_calendar('CME_Total')
        # Ir al mes anterior al contrato (asumimos el contrato siguiente al mes actual)
        target_month = month - 1 if month > 1 else 12
        target_year = year if month > 1 else year - 1
        
        last_day = pd.Period(f"{target_year}-{target_month}").to_timestamp(how='end')
        schedule = cme.schedule(start_date=f"{target_year}-{target_month}-01", end_date=last_day)
        
        # 4to día hábil antes del fin de mes
        vencimiento = schedule.iloc[-4].name 
        return vencimiento.date()
    except Exception:
        # Respaldo simple: fin del mes actual
        return datetime(year, month, 25).date()

# --- LÓGICA DE ESTADO DE SESIÓN ---

if 'paso_val' not in st.session_state:
    st.session_state.paso_val = 0.1

if 'market_cache' not in st.session_state:
    with st.spinner('Conectando con APIs financieras...'):
        st.session_state.market_cache = get_market_data_goldapi()
        st.session_state.tasa_cache = get_fred_risk_free_rate()

# --- INTERFAZ ---

st.title("📱 App de Valuación de Call sobre Oro")
st.markdown("---")

# 1. Definición de Variables de Ingreso
col1, col2 = st.columns(2)

with col1:
    beta = st.number_input("Beta", value=0.5, step=0.01)
    st.caption("ℹ️ Este valor se corresponde con el modelo de Black-Scholes")

    sigma_default = st.session_state.market_cache['sigma'] if st.session_state.market_cache else 0.16
    sigma = st.number_input("Sigma", value=sigma_default, format="%.2f")
    st.caption("ℹ️ volatilidad: valor conservador basado en datos pasados")

    s_default = st.session_state.market_cache['S'] if st.session_state.market_cache else 2000.0
    precio_s = st.number_input("Precio", value=s_default, format="%.2f")
    st.caption("ℹ️ datos tomados de GoldAPI")

with col2:
    # Strike: múltiplo de 5 más cercano al precio
    strike_init = round(precio_s / 5) * 5
    strike_k_input = st.number_input("Strike", value=float(strike_init), step=5.0)
    st.caption("ℹ️ at the money")

    tasa_r = st.number_input("Tasa", value=st.session_state.tasa_cache, format="%.4f")
    st.caption("ℹ️ fuente: FRED")

    # Fila para Paso y Botón multiplicador
    c_p1, c_p2 = st.columns([3, 1])
    with c_p1:
        paso_actual = st.number_input("Paso", value=st.session_state.paso_val, format="%.6f")
    with c_p2:
        st.write("") # Alineación
        if st.button("x10⁻¹"):
            st.session_state.paso_val = paso_actual * 0.1
            st.rerun()

# 3. Variable T (Vencimiento)
hoy = datetime.now()
# Buscamos el contrato de oro del mes siguiente
mes_contrato = hoy.month + 1 if hoy.month < 12 else 1
anio_contrato = hoy.year if hoy.month < 12 else hoy.year + 1
vencimiento_dt = fecha_vencimiento_oro(anio_contrato, mes_contrato)
T = (vencimiento_dt - hoy.date()).days / 365.0

st.info(f"📅 **Vencimiento estimado:** {vencimiento_dt} | **T:** {T:.4f} años")

# --- MOTOR DE CÁLCULO ---

def calcular_call_crr(S, K, r, T, sigma, beta, paso):
    # m = entero más cercano a T/Paso
    m = int(round(T / paso))
    if m <= 0: m = 1
    
    dt = T / m  # Tiempo por paso
    
    # u = exp( T^(1/2) * sigma * Paso^beta )
    u = np.exp((T**0.5) * sigma * (paso**beta))
    d = 1 / u
    
    # p = (exp(r * dt) - d) / (u - d)
    a = np.exp(r * dt)
    p = (a - d) / (u - d)
    
    # Asegurar estabilidad de p (arbitraje)
    p = max(min(p, 1.0), 0.0)
    
    suma_binomial = 0
    for k in range(m + 1):
        # Probabilidad de k éxitos en m pasos
        prob = comb(m, k) * (p**k) * ((1-p)**(m-k))
        # Nodo de precio final: S * u^(2k - m)
        st_k = S * (u**(2*k - m))
        # Payoff de la call: max(S_T - K, 0)
        payoff = max(st_k - K, 0)
        suma_binomial += prob * payoff
        
    # C = exp(-r * T) * Suma
    return np.exp(-r * T) * suma_binomial

# --- ACCIÓN: CALCULAR ---

if st.button("CALCULAR", type="primary"):
    # Variar K de 5 en 5 entre Strike-15 y Strike+15
    rango_strikes = np.arange(strike_k_input - 15, strike_k_input + 20, 5)
    resultados_c = []
    
    for k_val in rango_strikes:
        c_val = calcular_call_crr(precio_s, k_val, tasa_r, T, sigma, beta, paso_actual)
        resultados_c.append(c_val)
    
    df_res = pd.DataFrame({
        'Strike (K)': rango_strikes,
        'Precio Call (C)': resultados_c
    })

    # Mostrar Gráfico
    st.subheader("Gráfico de Precio de Call (C) vs Strike (K)")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_res['Strike (K)'], df_res['Precio Call (C)'], marker='o', color='#DAA520', linewidth=2)
    ax.fill_between(df_res['Strike (K)'], df_res['Precio Call (C)'], alpha=0.1, color='#DAA520')
    ax.set_xlabel("Strike (K)")
    ax.set_ylabel("Precio de la Opciones (C)")
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)
    
    # Mostrar Tabla
    st.table(df_res.style.format({'Precio Call (C)': '{:.2f}'}))

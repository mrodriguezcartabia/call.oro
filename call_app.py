import streamlit as st
import requests
import pandas as pd
import pandas_market_calendars as mcal
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.special import comb
from scipy.optimize import minimize_scalar

# Debe estar al comienzo

def guardar_manual():
    if st.session_state.input_manual_temp:
        st.session_state.market_cache = st.session_state.input_manual_temp     

# --- LÓGICA DE IDIOMA ---
params = st.query_params
idioma = params.get("lang", "en") # Por defecto inglés

texts = {
    "en": {
        "title": "Gold Call Valuator",
        "beta_lbl": "Beta",
        "beta_cap": "ℹ️ This value corresponds to the Black-Scholes model",
        "sigma_lbl": "Sigma (Volatility)",
        "sigma_cap": "ℹ️ Conservative value based on past data",
        "alpha_lbl": "Alpha",
        "fuente_precio": "ℹ️ Data from GoldAPI",
        "tasa_lbl": "Risk-Free Rate",
        "fuente_tasa": "ℹ️ Source: FRED",
        "venc_msg": "Expires in {} days ({})",
        "val_act": "Current Price",
        "strike_atm": "Strike (At-the-money)",
        "paso_temp": "Time Step",
        "reset": "Reset to initial value",
        "recalc": "RECALCULATE",
        "msg_loading": "Running binomial model...",
        "msg_success": "Calculation complete!",
        "graph_title": "Call Price (C) vs Strike (K)",
        "graph_y": "Call Price",
        "info_init": "Click RECALCULATE to generate the visualization.",
        "lbl_ingresar": "Enter market data",
        "lbl_cerrar": "Close to save",
        "lbl_hallar": "Find sigma",
        "lbl_res": "Sigma found",
        "lbl_mkt_info": "Enter market prices for each Strike:",
        "Mercado": "Price market",
        "msg_error_api": "No connection to GoldAPI",
        "msg_manual_price": "Please enter the price manually to continue.",
    },
    "es": {
        "title": "Valuador de Call de Oro",
        "beta_lbl": "Beta",
        "beta_cap": "ℹ️ Este valor corresponde al modelo de Black-Scholes",
        "sigma_lbl": "Sigma (Volatilidad)",
        "sigma_cap": "ℹ️ Valor conservador basado en datos pasados",
        "alpha_lbl": "Alfa",
        "fuente_precio": "ℹ️ Datos de GoldAPI",
        "tasa_lbl": "Tasa Libre de Riesgo",
        "fuente_tasa": "ℹ️ Fuente: FRED",
        "venc_msg": "Vencimiento en {} días ({})",
        "val_act": "Valor Actual",
        "strike_atm": "Strike At-the-money",
        "paso_temp": "Paso Temporal",
        "reset": "Reestablecer al valor inicial",
        "recalc": "RECALCULAR",
        "msg_loading": "Ejecutando modelo binomial...",
        "msg_success": "¡Cálculo finalizado!",
        "graph_title": "Gráfico de Precio de Call (C) vs Strike (K)",
        "graph_y": "Precio de la opción",
        "info_init": "Presiona RECALCULAR para generar la visualización.",
        "lbl_ingresar": "Ingresar datos de mercado",
        "lbl_cerrar": "Cerrar para guardar",
        "lbl_hallar": "Hallar sigma",
        "lbl_res": "Sigma hallado",
        "lbl_mkt_info": "Introduce los precios de mercado para cada Strike:",
        "Mercado": "Valor de mercado",
        "msg_error_api": "Sin conexión con GoldAPI",
        "msg_manual_price": "Por favor, coloque el precio manualmente para continuar.",
    },
    "pt": {
        "title": "Valiador de Call de Ouro",
        "beta_lbl": "Beta",
        "beta_cap": "ℹ️ Este valor corresponde ao modelo Black-Scholes",
        "sigma_lbl": "Sigma (Volatilidade)",
        "sigma_cap": "ℹ️ Valor conservador baseado em dados passados",
        "alpha_lbl": "Alfa",
        "fuente_precio": "ℹ️ Dados da GoldAPI",
        "tasa_lbl": "Taxa Livre de Risco",
        "fuente_tasa": "ℹ️ Fonte: FRED",
        "venc_msg": "Expira em {} dias ({})",
        "val_act": "Preço Atual",
        "strike_atm": "Strike At-the-money",
        "paso_temp": "Passo Temporal",
        "reset": "Redefinir para o valor inicial",
        "recalc": "RECALCULAR",
        "msg_loading": "Executando modelo binomial...",
        "msg_success": "Cálculo concluído!",
        "graph_title": "Gráfico de Preço da Call (C) vs Strike (K)",
        "graph_y": "Preço da opção",
        "info_init": "Clique em RECALCULAR para gerar a visualização.",
        "lbl_ingresar": "Insira os dados de mercado",
        "lbl_cerrar": "Fechar para salvar",
        "lbl_hallar": "Encontre sigma",
        "lbl_res": "Sigma encontrado",
        "lbl_mkt_info": "Insira os preços de mercado para cada Strike:",
        "Mercado": "Mercado de preços",
        "msg_error_api": "Sem conexão com a GoldAPI",
        "msg_manual_price": "Por favor, insira o preço manualmente para continuar.",
    }
}

t = texts.get(idioma, texts["en"])

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title=t["title"], layout="wide")

# Función para cargar el CSS  
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass # Por si el archivo aún no se sube o falla la lectura

local_css("style.css")

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
        return None
    except:
        return None #mensaje alerta

@st.cache_data(ttl=86400)
def fecha_vencimiento_oro(year, month):
    try:
        cme = mcal.get_calendar('CME_Total')
        last_day = pd.Period(f"{year}-{month}").to_timestamp(how='end')
        schedule = cme.schedule(start_date=f"{year}-{month}-01", end_date=last_day)
        return schedule.iloc[-4].name.date()
    except:
        return datetime(year, month, 25).date() #agregar alerta

@st.cache_data(ttl=86400)
def get_fred_risk_free_rate():
    try:
        api_key = st.secrets["FRED_API_KEY"]
        url = f"https://api.stlouisfed.org/api/series/observations?series_id=DTB4WK&api_key={api_key}&file_type=json&sort_order=desc&limit=1"
        response = requests.get(url)
        data = response.json()
        return float(data['observations'][0]['value']) / 100
    except:
        return 0.0425
        
def hallar_sigma_optimo(precios_mercado, strikes, S, r, T, beta, paso, param_a):
    def error_cuadratico(sigma_test):
        if sigma_test <= 0: return 1e10
        err = 0
        for i, k in enumerate(strikes):
            # Calculamos el precio del modelo para cada strike con el sigma de prueba
            c_mod = calcular_call(S, k, r, T, sigma_test, beta, paso, param_a)
            err += (c_mod - precios_mercado[i])**2
        return err
    
    # Optimizamos una sola variable (sigma) en un rango de 1% a 200%
    res = minimize_scalar(error_cuadratico, bounds=(0.01, 2.0), method='bounded')
    return res.x 

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
if 'mostrar_editor' not in st.session_state:
    st.session_state.mostrar_editor = False
if 'sigma_hallado' not in st.session_state:
    st.session_state.sigma_hallado = None
if 'precios_mercado' not in st.session_state:
    # Inicializamos con 15 ceros (asumiendo el rango de strikes por defecto)
    st.session_state.precios_mercado = [0.0] * 6

# --- INTERFAZ ---
# Intentamos obtener el precio de la sesión o de la API
        
if st.session_state.market_cache is None:
    _, center_col, _ = st.columns([1, 2, 1])
    
    with center_col:
        st.markdown(f"""
            <div class="overlay-card-static">
                <h2 style="color: #DAA520; text-align: center;">{t['msg_error_api']}</h2>
                <p style="color: white; text-align: center;">{t['msg_manual_price']}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # 1. Usamos una clave diferente para el input temporal
        # 2. Al pulsar Enter, guardamos el valor inmediatamente
        precio_temp = st.number_input(t["val_act"], value=None, placeholder="", key="input_manual_temp", on_change=guardar_manual)
        
        if st.button(t["recalc"], key="btn_start_manual", use_container_width=True, type="primary"):
            guardar_manual()
            if precio_temp is not None and  precio_temp > 0:
                st.rerun()
            else:
                st.warning(t["msg_manual_price"])
    
    st.stop()
    
# Si llegamos aquí, ya hay un precio (sea por API o manual)
dias = (vencimiento - hoy.date()).days 
T = dias/ 365.0
precio_s = float(st.session_state.market_cache) if st.session_state.market_cache is not None else 0.0      
strike = round(precio_s / 5) * 5

col1, col2 = st.columns(2)
with col1:
    param_a_def = 1.0
    param_a = st.number_input(t["alpha_lbl"], value=param_a_def, step=0.01)
    sigma_def = 0.16
    sigma = st.number_input(t["sigma_lbl"], value=sigma_def, format="%.2f")
    st.caption(t["sigma_cap"])

with col2:
    beta = st.number_input("Beta", value=0.5, step=0.01)
    tasa_r = st.number_input(t["tasa_lbl"], value=st.session_state.tasa_cache, format="%.4f")
    st.caption(t["fuente_tasa"])

# --- BOTONES DE CONTROL Y GRÁFICO ---
herramientas, grafico = st.columns([1, 3])
with herramientas:
    st.info(t["venc_msg"].format(dias, vencimiento))
    st.markdown(f"""
        <div class="custom-metric-container">
            <span class="metric-label">{t["val_act"]}:</span>
            <span class="metric-value-small">{precio_s}</span>
        </div>
    """, unsafe_allow_html=True)
    st.caption(t["fuente_precio"])
    st.markdown(f"""
        <div class="custom-metric-container">
            <span class="metric-label">{t["paso_temp"]}:</span>
            <span class="metric-value-small">{st.session_state.paso_val:.8f}</span>
        </div>
    """, unsafe_allow_html=True)

    # Botones de paso temporal
    boton1, boton2 = st.columns([1, 2])
    with boton1:
        if st.button("x10⁻¹"):
            st.session_state.paso_val *= 0.1
            st.rerun()
    with boton2:
        if st.button(t["reset"], key="btn-reset"):
            st.session_state.paso_val = VALOR_PASO_ORIGINAL
            st.rerun()
            
    # Código para buscar sigma        
    btn_recalcular = st.button(t["recalc"], type="primary", use_container_width=True)

    with st.popover(t["lbl_ingresar"], use_container_width=True):
        st.write(t["lbl_mkt_info"])
        
        # 1. Creamos un formulario interno
        with st.form("form_mercado"):
            rango_edicion = np.arange(strike - 15, strike + 15, 5)
            
            # Sincronización inicial
            if len(st.session_state.precios_mercado) != len(rango_edicion):
                st.session_state.precios_mercado = [0.0] * len(rango_edicion)
                
            df_editor = pd.DataFrame({
                "Strike": rango_edicion, 
                "Precio Call Mercado": st.session_state.precios_mercado
            })
            
            # El editor dentro del formulario no dispara re-ejecuciones automáticas
            edited_df = st.data_editor(
                df_editor, 
                hide_index=True, 
                use_container_width=True,
                num_rows="fixed",
                column_config={"Strike": st.column_config.NumberColumn(disabled=True)}
            )
            
            # 2. Botón para confirmar los cambios
            submit_save = st.form_submit_button(t["lbl_cerrar"], use_container_width=True)
            
            if submit_save:
                # Solo aquí guardamos los datos en el estado global
                st.session_state.precios_mercado = edited_df["Precio Call Mercado"].tolist()
                st.rerun() # Esto refresca el gráfico con los nuevos puntos rojos

    # Botón para hallar sigma (ahora queda debajo del popover)
    if st.button(t["lbl_hallar"], type="primary", use_container_width=True):
        strikes_actuales = np.arange(strike - 15, strike + 15, 5)
        sigma_fit = hallar_sigma_optimo(
            st.session_state.precios_mercado, 
            strikes_actuales, precio_s, tasa_r, T, beta, 
            st.session_state.paso_val, param_a
        )
        st.session_state.sigma_hallado = sigma_fit
        st.session_state.data_grafico = (strikes_actuales, [
            calcular_call(precio_s, k, tasa_r, T, sigma_fit, beta, st.session_state.paso_val, param_a) 
            for k in strikes_actuales
        ])
        st.rerun()

    # Resultado del Sigma hallado
    valor_sigma = f"{st.session_state.sigma_hallado:.5f}" if st.session_state.sigma_hallado else ""
    st.metric(label=t["lbl_res"], value=valor_sigma)

# --- LÓGICA DE CÁLCULO BAJO DEMANDA ---
if st.session_state.data_grafico is None or btn_recalcular:
    # Indicador de carga activo durante el proceso matemático
    with st.spinner(t['msg_loading']):
        rango_strikes = np.arange(strike - 15, strike + 15, 5)
        valores_c = []
        for k in rango_strikes:
            c = calcular_call(precio_s, k, tasa_r, T, sigma, beta, st.session_state.paso_val, param_a)
            valores_c.append(c)
        st.session_state.data_grafico = (rango_strikes, valores_c)
    # Mensaje temporal de éxito
    if btn_recalcular:
        st.toast(t["msg_success"])
        
# --- COLOCAMOS EL GRÁFICO ---
with grafico:
    strikes, calls = st.session_state.data_grafico

    #st.subheader("Gráfico de Precio de Call (C) vs Strike (K)")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor('#e2e8f0') 
    ax.set_facecolor('#e2e8f0')
    
    # Curva del Modelo
    ax.plot(strikes, calls, marker='o', color='#B8860B', linewidth=2)
    ax.fill_between(strikes, calls, alpha=0.1, color='#B8860B', label='Call')

    # Curva de Mercado   - Solo si el usuario ingresó algún valor > 0
    if any(p > 0 for p in st.session_state.precios_mercado):
        ax.plot(strikes, st.session_state.precios_mercado, marker='o', color='#000000', linewidth=2)
        ax.fill_between(strikes, st.session_state.precios_mercado, alpha=0.1, color='#000000', label=t['Mercado'])
        
    ax.set_xlabel("Strike")
    ax.set_ylabel(t["graph_y"])
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

    # Eliminar bordes innecesarios
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

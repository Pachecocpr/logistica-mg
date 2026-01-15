import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
from io import BytesIO
import numpy as np
from geopy.geocoders import Nominatim
import time
import random

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Logística Pro - Otimizador", page_icon="🚚", layout="wide")

# Função de busca com múltiplas tentativas (ajuda na validação da origem)
def buscar_coordenadas_robusta(endereco):
    # Geramos um user_agent aleatório para evitar bloqueios de API
    user_agent = f"logistica_mg_{random.randint(1000, 9999)}"
    geolocator = Nominatim(user_agent=user_agent, timeout=10)
    
    try:
        # Tentativa 1: Endereço completo
        location = geolocator.geocode(endereco)
        if location:
            return location.latitude, location.longitude
        
        # Tentativa 2: Apenas as últimas partes (Cidade, Estado, País)
        partes = endereco.split(',')
        if len(partes) > 1:
            resumo = ", ".join(partes[-2:]) # Pega as últimas duas partes
            location = geolocator.geocode(resumo)
            if location:
                return location.latitude, location.longitude
                
        return None, None
    except Exception as e:
        st.sidebar.warning(f"Erro de conexão com o mapa: {e}")
        return None, None

def calcular_distancia(lat1, lon1, lat2, lon2):
    r = 6371 
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlambda = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return (2 * r * np.arcsin(np.sqrt(a))) * 1.3

def formatar_tempo(km):
    horas = km / 60
    return f"{int(horas//1)}h {int((horas%1)*60)}min"

st.title("🔄 Otimizador Logístico: Trechos Detalhados")
modo = st.sidebar.selectbox("Modo de Operação:", ["Base Fixa (MG)", "Importação Manual (CSV)"])

st.sidebar.divider()
st.sidebar.header("📍 Ponto de Partida")

# Dica para o usuário
st.sidebar.info("Dica: Use o formato 'Rua, Número, Cidade, MG'")
endereco_origem = st.sidebar.text_input("Endereço de Saída:", value="", placeholder="Ex: Av. Amazonas, 100, Belo Horizonte, MG")

if st.sidebar.button("🔍 Validar Origem"):
    if endereco_origem:
        with st.spinner('Validando endereço...'):
            lat_o, lon_o = buscar_coordenadas_robusta(endereco_origem)
            if lat_o:
                st.session_state['lat_o'], st.session_state['lon_o'] = lat_o, lon_o
                st.sidebar.success(f"✅ Localizado: {lat_o:.4f}, {lon_o:.4f}")
            else:
                st.sidebar.error("❌ Endereço não encontrado. Tente apenas 'Cidade, MG' para testar.")
    else:
        st.sidebar.warning("Digite um endereço.")

if 'lat_o' not in st.session_state:
    st.info("👋 Por favor, valide seu endereço de origem para começar.")
    st.stop()

lat_p, lon_p = st.session_state['lat_o'], st.session_state['lon_o']

# --- LÓGICA DE DADOS ---
df_final = pd.DataFrame()

if modo == "Base Fixa (MG)":
    try:
        df_base = pd.read_csv('municipios_mg.csv')
        regioes = sorted(df_base['regiao'].dropna().unique())
        selecao = st.sidebar.multiselect("Regiões de Minas:", options=regioes)
        if selecao:
            df_final = df_base[df_base['regiao'].isin(selecao)].copy()
            df_final['nome_exibicao'] = df_final['cidade']
        else:
            st.warning("Selecione uma região.")
            st.stop()
    except:
        st.error("Erro no arquivo municipios_mg.csv")
        st.stop()
else:
    arquivo = st.sidebar.file_uploader("Suba seu CSV:", type=["csv"])
    if arquivo:
        df_import = pd.read_csv(arquivo, encoding='iso-8859-1', sep=None, engine='python')
        df_final = pd.DataFrame()
        df_final['nome_exibicao'] = df_import.iloc[:, 7].astype(str) # Município
        df_final['rua'] = df_import.iloc[:, 2].astype(str)
        df_final['num'] = df_import.iloc[:, 3].astype(str)
        
        st.info("Buscando coordenadas dos destinos...")
        prog = st.progress(0)
        lats, lons = [], []
        for i, row in df_final.iterrows():
            end = f"{row['rua']}, {row['num']}, {row['nome_exibicao']}, MG"
            lt, ln = buscar_coordenadas_robusta(end)
            lats.append(lt); lons.append(ln)
            prog.progress((i+1)/len(df_final))
            time.sleep(0.8) # Pausa para evitar bloqueio de API
        df_final['lat'], df_final['lon'] = lats, lons
        df_final = df_final.dropna(subset=['lat', 'lon'])
    else:
        st.stop()

# --- ROTEIRIZAÇÃO E RELATÓRIO ---
if not df_final.empty:
    capacidade = st.sidebar.slider("Paradas por caminhão:", 2, 15, 5)
    n_clusters = max(1, len(df_final) // capacidade)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_final['ID_Rota'] = kmeans.fit_predict(df_final[['lat', 'lon']])

    trechos = []
    for id_r, grupo in df_final.groupby('ID_Rota'):
        grupo = grupo.reset_index()
        rota_num = id_r + 1
        
        # 1. Origem -> Primeira Cidade
        d = calcular_distancia(lat_p, lon_p, grupo.loc[0, 'lat'], grupo.loc[0, 'lon'])
        trechos.append({
            'Rota': rota_num, 'De': 'ORIGEM', 'Para': grupo.loc[0, 'nome_exibicao'],
            'KM Trecho': round(d, 1), 'Tempo': formatar_tempo(d)
        })
        
        # 2. Trechos intermediários
        for j in range(len(grupo)-1):
            d = calcular_distancia(grupo.loc[j, 'lat'], grupo.loc[j, 'lon'], grupo.loc[j+1, 'lat'], grupo.loc[j+1, 'lon'])
            trechos.append({
                'Rota': rota_num, 'De': grupo.loc[j, 'nome_exibicao'], 'Para': grupo.loc[j+1, 'nome_exibicao'],
                'KM Trecho': round(d, 1), 'Tempo': formatar_tempo(d)
            })
        
        # 3. Última Cidade -> Origem
        d = calcular_distancia(grupo.loc[len(grupo)-1, 'lat'], grupo.loc[len(grupo)-1, 'lon'], lat_p, lon_p)
        trechos.append({
            'Rota': rota_num, 'De': grupo.loc[len(grupo)-1, 'nome_exibicao'], 'Para': 'RETORNO À ORIGEM',
            'KM Trecho': round(d, 1), 'Tempo': formatar_tempo(d)
        })

    df_rel_trechos = pd.DataFrame(trechos)
    st.subheader("📊 Itinerário Passo a Passo")
    st.dataframe(df_rel_trechos, use_container_width=True)
    
    fig = px.scatter_mapbox(df_final, lat="lat", lon="lon", color="ID_Rota", zoom=6)
    fig.add_scattermapbox(lat=[lat_p], lon=[lon_p], marker=dict(size=14, color='red'), name="ORIGEM")
    fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_rel_trechos.to_excel(writer, index=False)
    st.download_button("📥 Baixar Planilha", output.getvalue(), "itinerario.xlsx")

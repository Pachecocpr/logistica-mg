import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
from io import BytesIO
import numpy as np
from geopy.geocoders import Nominatim
import time

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Logística Pro - Roteirizador por Endereço", page_icon="🚚", layout="wide")

geolocator = Nominatim(user_agent="logistica_pacheco_v5")

# Função para buscar coordenadas (Geocodificação)
def buscar_coordenadas(local):
    try:
        location = geolocator.geocode(local)
        if location:
            return location.latitude, location.longitude
        return None, None
    except:
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

st.title("🚚 Otimizador Logístico por Endereços de CSV")

# --- BARRA LATERAL ---
st.sidebar.header("📂 Importação de Dados")
arquivo_upload = st.sidebar.file_uploader("Suba seu arquivo CSV com as 8 colunas:", type=["csv"])

st.sidebar.divider()
st.sidebar.header("📍 Ponto de Partida (Origem)")
endereco_origem = st.sidebar.text_input("Seu Endereço de Saída:", placeholder="Ex: Rua Simão Antonio, 149, Contagem, MG")

if st.sidebar.button("🔍 Validar Origem"):
    lat_o, lon_o = buscar_coordenadas(endereco_origem)
    if lat_o:
        st.session_state['lat_o'], st.session_state['lon_o'] = lat_o, lon_o
        st.sidebar.success("Origem Validada!")
    else:
        st.sidebar.error("Origem não encontrada.")

# --- PROCESSAMENTO ---
if arquivo_upload and 'lat_o' in st.session_state:
    try:
        # Lê o CSV sem assumir nomes de colunas fixos
        df_import = pd.read_csv(arquivo_upload)
        
        # Seleção por ÍNDICE (conforme sua solicitação)
        # Col 3 = index 2 | Col 4 = index 3 | Col 8 = index 7
        df_process = pd.DataFrame()
        df_process['rua'] = df_import.iloc[:, 2].astype(str)
        df_process['numero'] = df_import.iloc[:, 3].astype(str)
        df_process['cidade'] = df_import.iloc[:, 7].astype(str)
        
        # Cria a string completa para o GPS
        df_process['endereco_completo'] = df_process['rua'] + ", " + df_process['numero'] + ", " + df_process['cidade'] + ", MG"

        st.write(f"### 📍 Processando {len(df_process)} destinos do arquivo...")
        
        progresso = st.progress(0)
        lats, lons = [], []
        
        # Loop de Geocodificação (transforma endereço em Lat/Lon)
        for i, endereco in enumerate(df_process['endereco_completo']):
            lat, lon = buscar_coordenadas(endereco)
            # Se não achar o endereço com número, tenta buscar só a cidade/município
            if not lat:
                lat, lon = buscar_coordenadas(df_process.iloc[i]['cidade'] + ", MG")
                
            lats.append(lat)
            lons.append(lon)
            progresso.progress((i + 1) / len(df_process))
            time.sleep(1) # Respeita o limite do servidor de mapas

        df_process['lat'] = lats
        df_process['lon'] = lons
        df_process = df_process.dropna(subset=['lat', 'lon']) # Remove falhas

        # --- ROTEIRIZAÇÃO ---
        cidades_por_rota = st.sidebar.slider("Paradas por caminhão:", 2, 15, 5)
        n_clusters = max(1, len(df_process) // cidades_por_rota)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_process['ID_Rota'] = kmeans.fit_predict(df_process[['lat', 'lon']])

        relatorio = []
        lat_p, lon_p = st.session_state['lat_o'], st.session_state['lon_o']

        for id_r, grupo in df_process.groupby('ID_Rota'):
            grupo = grupo.reset_index()
            d_ida = calcular_distancia(lat_p, lon_p, grupo.loc[0, 'lat'], grupo.loc[0, 'lon'])
            for j in range(len(grupo)-1):
                d_ida += calcular_distancia(grupo.loc[j, 'lat'], grupo.loc[j, 'lon'], grupo.loc[j+1, 'lat'], grupo.loc[j+1, 'lon'])
            
            d_ret = calcular_distancia(grupo.loc[len(grupo)-1, 'lat'], grupo.loc[len(grupo)-1, 'lon'], lat_p, lon_p)
            km_total = d_ida + d_ret
            
            relatorio.append({
                'Rota': id_r + 1,
                'Itinerário': ' ➔ '.join(grupo['cidade'].unique()),
                'KM Ida': round(d_ida, 1),
                'KM Retorno': round(d_ret, 1),
                'KM TOTAL': round(km_total, 1),
                'TEMPO TOTAL': formatar_tempo(km_total)
            })

        # --- EXIBIÇÃO ---
        st.subheader("📊 Planejamento de Rotas Otimizado")
        st.dataframe(pd.DataFrame(relatorio), use_container_width=True)
        
        # Mapa interativo
        fig = px.scatter_mapbox(df_process, lat="lat", lon="lon", color="ID_Rota", 
                                hover_name="endereco_completo", zoom=6)
        fig.add_scattermapbox(lat=[lat_p], lon=[lon_p], marker=dict(size=14, color='red'), name="SUA ORIGEM")
        fig.update_layout(mapbox_style="carto-darkmatter", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

        # Download do Excel
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            pd.DataFrame(relatorio).to_excel(writer, index=False)
        st.download_button("📥 Baixar Relatório das Rotas", output.getvalue(), "rotas_detalhadas.xlsx")

    except Exception as e:
        st.error(f"Erro ao ler colunas do CSV: {e}")
else:
    st.info("Aguardando upload do CSV e validação da Origem.")

import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
from io import BytesIO
import numpy as np
from geopy.geocoders import Nominatim
import time

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Logística Pro - Relatório por Trecho", page_icon="🚚", layout="wide")

geolocator = Nominatim(user_agent="logistica_pacheco_v8")

# --- FUNÇÕES ---
def buscar_coordenadas(local):
    try:
        location = geolocator.geocode(local)
        if location: return location.latitude, location.longitude
        return None, None
    except: return None, None

def calcular_distancia(lat1, lon1, lat2, lon2):
    r = 6371 
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlambda = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return (2 * r * np.arcsin(np.sqrt(a))) * 1.3

def formatar_tempo(km):
    horas = km / 60
    return f"{int(horas//1)}h {int((horas%1)*60)}min"

st.title("🔄 Otimizador Logístico: Relatório Detalhado por Trecho")
modo = st.sidebar.selectbox("Modo de Operação:", ["Base Fixa (MG)", "Importação Manual (CSV)"])

st.sidebar.divider()
st.sidebar.header("📍 Ponto de Partida")
endereco_origem = st.sidebar.text_input("Endereço de Saída:", value="", placeholder="Rua, Número, Cidade, MG")

if st.sidebar.button("🔍 Validar Origem"):
    lat_o, lon_o = buscar_coordenadas(endereco_origem)
    if lat_o:
        st.session_state['lat_o'], st.session_state['lon_o'] = lat_o, lon_o
        st.sidebar.success("Origem Validada!")
    else:
        st.sidebar.error("Origem não encontrada.")

if 'lat_o' not in st.session_state or endereco_origem == "":
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
        
        st.info("Buscando coordenadas...")
        prog = st.progress(0)
        lats, lons = [], []
        for i, row in df_final.iterrows():
            end = f"{row['rua']}, {row['num']}, {row['nome_exibicao']}, MG"
            lt, ln = buscar_coordenadas(end)
            if not lt: lt, ln = buscar_coordenadas(f"{row['nome_exibicao']}, MG")
            lats.append(lt); lons.append(ln)
            prog.progress((i+1)/len(df_final))
            time.sleep(0.5)
        df_final['lat'], df_final['lon'] = lats, lons
        df_final = df_final.dropna(subset=['lat', 'lon'])
    else:
        st.stop()

# --- ROTEIRIZAÇÃO E RELATÓRIO POR TRECHO ---
if not df_final.empty:
    capacidade = st.sidebar.slider("Paradas por caminhão:", 2, 15, 5)
    n_clusters = max(1, len(df_final) // capacidade)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_final['ID_Rota'] = kmeans.fit_predict(df_final[['lat', 'lon']])

    trechos = []
    for id_r, grupo in df_final.groupby('ID_Rota'):
        grupo = grupo.reset_index()
        rota_num = id_r + 1
        
        # 1. Primeiro Trecho: Origem -> Primeira Cidade
        d = calcular_distancia(lat_p, lon_p, grupo.loc[0, 'lat'], grupo.loc[0, 'lon'])
        trechos.append({
            'Rota': rota_num, 'De': 'ORIGEM (Ponto de Partida)', 'Para': grupo.loc[0, 'nome_exibicao'],
            'KM Trecho': round(d, 1), 'Tempo Trecho': formatar_tempo(d), 'Tipo': 'IDA'
        })
        
        # 2. Trechos entre Cidades
        for j in range(len(grupo)-1):
            d = calcular_distancia(grupo.loc[j, 'lat'], grupo.loc[j, 'lon'], grupo.loc[j+1, 'lat'], grupo.loc[j+1, 'lon'])
            trechos.append({
                'Rota': rota_num, 'De': grupo.loc[j, 'nome_exibicao'], 'Para': grupo.loc[j+1, 'nome_exibicao'],
                'KM Trecho': round(d, 1), 'Tempo Trecho': formatar_tempo(d), 'Tipo': 'ENTRE CIDADES'
            })
        
        # 3. Último Trecho: Última Cidade -> Origem
        d = calcular_distancia(grupo.loc[len(grupo)-1, 'lat'], grupo.loc[len(grupo)-1, 'lon'], lat_p, lon_p)
        trechos.append({
            'Rota': rota_num, 'De': grupo.loc[len(grupo)-1, 'nome_exibicao'], 'Para': 'ORIGEM (Retorno)',
            'KM Trecho': round(d, 1), 'Tempo Trecho': formatar_tempo(d), 'Tipo': 'RETORNO'
        })

    df_rel_trechos = pd.DataFrame(trechos)

    # --- EXIBIÇÃO ---
    st.subheader("📊 Itinerário Detalhado (Trecho a Trecho)")
    st.dataframe(df_rel_trechos, use_container_width=True)
    
    # Mapa
    fig = px.scatter_mapbox(df_final, lat="lat", lon="lon", color="ID_Rota", hover_name="nome_exibicao", zoom=6)
    fig.add_scattermapbox(lat=[lat_p], lon=[lon_p], marker=dict(size=14, color='red'), name="ORIGEM")
    fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

    # Download
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_rel_trechos.to_excel(writer, index=False)
    st.download_button("📥 Baixar Planilha de Trechos", output.getvalue(), "itinerario_detalhado.xlsx")

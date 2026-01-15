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
st.set_page_config(page_title="Logística Pro v11", page_icon="🚚", layout="wide")

# --- FUNÇÃO DE BUSCA ULTRA REFORÇADA ---
def buscar_coordenadas(local):
    """Busca com identificador aleatório para evitar bloqueios."""
    if not local: return None, None
    
    # Lista de IDs para rotacionar e evitar bloqueio de IP
    user_agents = [f"mg_log_app_{random.randint(1, 99999)}" for _ in range(5)]
    
    try:
        geolocator = Nominatim(user_agent=random.choice(user_agents), timeout=20)
        # Tenta buscar reforçando que é em Minas Gerais, Brasil
        query = f"{local}, Minas Gerais, Brazil"
        location = geolocator.geocode(query)
        
        if location:
            return location.latitude, location.longitude
        return None, None
    except Exception:
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

# --- INTERFACE ---
st.title("🔄 Otimizador Logístico: Versão com Contingência")

# Carregar base de cidades para o menu de backup
try:
    df_cidades_mg = pd.read_csv('municipios_mg.csv')
    lista_cidades = sorted(df_cidades_mg['cidade'].unique())
except:
    lista_cidades = ["Belo Horizonte", "Contagem", "Uberlândia", "Juiz de Fora", "Betim"]

st.sidebar.header("📍 Ponto de Partida")
metodo_origem = st.sidebar.radio("Como deseja informar a origem?", ["Digitar Endereço", "Selecionar Cidade"])

if metodo_origem == "Digitar Endereço":
    endereco_origem = st.sidebar.text_input("Endereço (Rua, Número, Cidade):")
else:
    endereco_origem = st.sidebar.selectbox("Escolha a cidade de saída:", options=lista_cidades)

if st.sidebar.button("🔍 Validar Origem"):
    with st.spinner('Conectando ao servidor de mapas...'):
        lat, lon = buscar_coordenadas(endereco_origem)
        if lat:
            st.session_state['lat_o'], st.session_state['lon_o'] = lat, lon
            st.session_state['nome_origem'] = endereco_origem
            st.sidebar.success(f"✅ Sucesso! Origem fixada.")
        else:
            st.sidebar.error("❌ Erro de conexão ou endereço não encontrado.")

if 'lat_o' not in st.session_state:
    st.warning("⚠️ Valide o Ponto de Partida na lateral antes de prosseguir.")
    st.stop()

# --- MODO DE OPERAÇÃO ---
st.sidebar.divider()
modo = st.sidebar.selectbox("Modo de Operação:", ["Base Fixa (Regiões de MG)", "Importar CSV Customizado"])

df_final = pd.DataFrame()

if modo == "Base Fixa (Regiões de MG)":
    regioes = sorted(df_cidades_mg['regiao'].dropna().unique())
    selecao = st.sidebar.multiselect("Filtre por Região:", options=regioes)
    if selecao:
        df_final = df_cidades_mg[df_cidades_mg['regiao'].isin(selecao)].copy()
        df_final['label'] = df_final['cidade']
    else: st.stop()

else: # MODO IMPORTAÇÃO
    arquivo = st.sidebar.file_uploader("Suba o CSV (Colunas 3, 4 e 8)", type=["csv"])
    if arquivo:
        try:
            df_import = pd.read_csv(arquivo, encoding='iso-8859-1', sep=None, engine='python')
            df_final = pd.DataFrame()
            df_final['rua'] = df_import.iloc[:, 2].astype(str)
            df_final['num'] = df_import.iloc[:, 3].astype(str)
            df_final['cid'] = df_import.iloc[:, 7].astype(str)
            df_final['label'] = df_final['cid']
            
            st.info("📍 Buscando coordenadas... (1 segundo por endereço)")
            barra = st.progress(0)
            lats, lons = [], []
            for i, r in df_final.iterrows():
                lt, ln = buscar_coordenadas(f"{r['rua']}, {r['num']}, {r['cid']}")
                if not lt: lt, ln = buscar_coordenadas(r['cid'])
                lats.append(lt); lons.append(ln)
                barra.progress((i+1)/len(df_final))
                time.sleep(1.2) # Pausa de segurança
            df_final['lat'], df_final['lon'] = lats, lons
        except: st.error("Erro no CSV."); st.stop()
    else: st.stop()

# --- RESULTADOS ---
df_final = df_final.dropna(subset=['lat', 'lon'])
if not df_final.empty:
    qtd = st.sidebar.slider("Paradas por veículo:", 2, 15, 5)
    n_clusters = max(1, len(df_final) // qtd)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_final['ID_Rota'] = kmeans.fit_predict(df_final[['lat', 'lon']])

    resumo = []
    lat_p, lon_p = st.session_state['lat_o'], st.session_state['lon_o']
    for id_r, gp in df_final.groupby('ID_Rota'):
        gp = gp.reset_index()
        d_ida = calcular_distancia(lat_p, lon_p, gp.loc[0, 'lat'], gp.loc[0, 'lon'])
        for j in range(len(gp)-1):
            d_ida += calcular_distancia(gp.loc[j, 'lat'], gp.loc[j, 'lon'], gp.loc[j+1, 'lat'], gp.loc[j+1, 'lon'])
        d_ret = calcular_distancia(gp.loc[len(gp)-1, 'lat'], gp.loc[len(gp)-1, 'lon'], lat_p, lon_p)
        total = d_ida + d_ret
        resumo.append({'Rota': id_r+1, 'Itinerário': ' ➔ '.join(gp['label'].unique()), 'KM TOTAL': round(total, 1), 'TEMPO': formatar_tempo(total)})

    st.write(f"### 📍 Saindo de: {st.session_state['nome_origem']}")
    st.dataframe(pd.DataFrame(resumo), use_container_width=True)
    fig = px.scatter_mapbox(df_final, lat="lat", lon="lon", color="ID_Rota", zoom=5.5)
    fig.add_scattermapbox(lat=[lat_p], lon=[lon_p], marker=dict(size=15, color='red'), name="BASE")
    fig.update_layout(mapbox_style="open-stret-map", margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

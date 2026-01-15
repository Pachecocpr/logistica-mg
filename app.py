import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
from io import BytesIO
import numpy as np
from geopy.geocoders import Nominatim
import time

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Logística Pro v9", page_icon="🚚", layout="wide")

# --- FUNÇÕES TÉCNICAS ---

def buscar_coordenadas(local):
    """Busca coordenadas com reforço de localização para Minas Gerais."""
    try:
        # User-agent único e timeout longo para estabilidade
        geolocator = Nominatim(user_agent="logistica_mg_pacheco_v9", timeout=15)
        
        # Reforça a busca para dentro de Minas Gerais se não houver país/estado especificado
        if "Minas Gerais" not in local:
            local = f"{local}, Minas Gerais, Brazil"
            
        location = geolocator.geocode(local)
        if location:
            return location.latitude, location.longitude
        return None, None
    except Exception:
        return None, None

def calcular_distancia(lat1, lon1, lat2, lon2):
    """Haversine com 30% de margem para estradas."""
    r = 6371 
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlambda = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return (2 * r * np.arcsin(np.sqrt(a))) * 1.3

def formatar_tempo(km):
    horas = km / 60
    return f"{int(horas//1)}h {int((horas%1)*60)}min"

# --- INTERFACE ---
st.title("🔄 Otimizador Logístico: Minas Gerais")
modo = st.sidebar.selectbox("Modo de Trabalho:", ["Base Fixa (MG)", "Importar CSV Customizado"])

st.sidebar.divider()

# --- ORIGEM ---
st.sidebar.header("📍 Ponto de Partida")
endereco_origem = st.sidebar.text_input("Endereço de Saída:", placeholder="Ex: Rua Simão Antonio, 149, Contagem, MG")

if st.sidebar.button("🔍 Validar e Fixar Origem"):
    if endereco_origem:
        with st.spinner('Localizando ponto de partida...'):
            lat_o, lon_o = buscar_coordenadas(endereco_origem)
            if lat_o:
                st.session_state['lat_o'], st.session_state['lon_o'] = lat_o, lon_o
                st.sidebar.success("Origem fixada!")
            else:
                st.sidebar.error("Origem não encontrada. Tente: 'Cidade, MG'")
    else:
        st.sidebar.warning("Digite a origem.")

if 'lat_o' not in st.session_state:
    st.info("👋 **Ação Necessária:** Valide o endereço de saída na barra lateral.")
    st.stop()

lat_p, lon_p = st.session_state['lat_o'], st.session_state['lon_o']

# --- DADOS ---
df_final = pd.DataFrame()

if modo == "Base Fixa (MG)":
    try:
        df_base = pd.read_csv('municipios_mg.csv')
        st.sidebar.header("🗺️ Filtros de Região")
        regioes = sorted(df_base['regiao'].dropna().unique())
        selecao = st.sidebar.multiselect("Selecione as Regiões:", options=regioes)
        
        if selecao:
            df_final = df_base[df_base['regiao'].isin(selecao)].copy()
            df_final['label_cidade'] = df_final['cidade'].astype(str).str.strip()
            
            # Se a base fixa não tiver Lat/Lon, vamos buscar agora
            if 'lat' not in df_final.columns or df_final['lat'].isnull().any():
                st.info("Buscando coordenadas das cidades selecionadas...")
                lats, lons = [], []
                barra = st.progress(0)
                for i, cid in enumerate(df_final['label_cidade']):
                    lt, ln = buscar_coordenadas(cid)
                    lats.append(lt); lons.append(ln)
                    barra.progress((i+1)/len(df_final))
                    time.sleep(1.1)
                df_final['lat'], df_final['lon'] = lats, lons
        else:
            st.stop()
    except Exception as e:
        st.error(f"Erro ao carregar base fixa: {e}")
        st.stop()

else: # MODO IMPORTAÇÃO
    arquivo = st.sidebar.file_uploader("Suba o CSV (Colunas 3, 4 e 8)", type=["csv"])
    if arquivo:
        try:
            df_import = pd.read_csv(arquivo, encoding='iso-8859-1', sep=None, engine='python')
            df_final = pd.DataFrame()
            df_final['rua'] = df_import.iloc[:, 2].astype(str).str.strip()
            df_final['numero'] = df_import.iloc[:, 3].astype(str).str.strip()
            df_final['municipio'] = df_import.iloc[:, 7].astype(str).str.strip()
            df_final['label_cidade'] = df_final['municipio']
            
            st.info("📍 Mapeando endereços do arquivo...")
            barra = st.progress(0)
            lats, lons = [], []
            for i, row in df_final.iterrows():
                # Tenta Endereço Completo
                lt, ln = buscar_coordenadas(f"{row['rua']}, {row['numero']}, {row['municipio']}")
                if not lt: # Tenta só Cidade
                    lt, ln = buscar_coordenadas(row['municipio'])
                lats.append(lt); lons.append(ln)
                barra.progress((i+1)/len(df_final))
                time.sleep(1.1)
            
            df_final['lat'], df_final['lon'] = lats, lons
        except Exception as e:
            st.error(f"Erro: {e}"); st.stop()
    else:
        st.stop()

# --- ROTAS ---
df_final = df_final.dropna(subset=['lat', 'lon'])

if not df_final.empty:
    st.sidebar.divider()
    cidades_por_caminhao = st.sidebar.slider("Cidades por Rota:", 2, 15, 5)
    n_clusters = max(1, len(df_final) // cidades_por_caminhao)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_final['ID_Rota'] = kmeans.fit_predict(df_final[['lat', 'lon']])

    relatorio_dados = []
    for id_r, grupo in df_final.groupby('ID_Rota'):
        grupo = grupo.reset_index()
        d_ida = calcular_distancia(lat_p, lon_p, grupo.loc[0, 'lat'], grupo.loc[0, 'lon'])
        for j in range(len(grupo)-1):
            d_ida += calcular_distancia(grupo.loc[j, 'lat'], grupo.loc[j, 'lon'], grupo.loc[j+1, 'lat'], grupo.loc[j+1, 'lon'])
        d_ret = calcular_distancia(grupo.loc[len(grupo)-1, 'lat'], grupo.loc[len(grupo)-1, 'lon'], lat_p, lon_p)
        km_s = d_ida + d_ret
        
        relatorio_dados.append({
            'Rota': id_r + 1,
            'Itinerário': ' ➔ '.join(grupo['label_cidade'].unique()),
            'KM TOTAL': round(km_s, 1),
            'TEMPO TOTAL': formatar_tempo(km_s)
        })

    st.subheader("📊 Relatório Final")
    st.dataframe(pd.DataFrame(relatorio_dados), use_container_width=True)
    
    fig = px.scatter_mapbox(df_final, lat="lat", lon="lon", color="ID_Rota", hover_name="label_cidade", zoom=5.5)
    fig.add_scattermapbox(lat=[lat_p], lon=[lon_p], marker=dict(size=14, color='red'), name="BASE")
    fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

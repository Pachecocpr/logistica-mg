import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
from io import BytesIO
import numpy as np
from geopy.geocoders import Nominatim
import time

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Logística MG Pro v2", page_icon="🚚", layout="wide")

geolocator = Nominatim(user_agent="logistica_pacheco_v7")

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

# --- TÍTULO E MENU ---
st.title("🚚 Sistema Integrado de Otimização Logística")
modo = st.sidebar.selectbox("Selecione o Modo de Operação:", ["Base Fixa (MG)", "Importação Manual (CSV)"])

st.sidebar.divider()

# --- CONFIGURAÇÃO DE ORIGEM ---
st.sidebar.header("📍 Ponto de Partida")
endereco_origem = st.sidebar.text_input("Endereço de Saída:", placeholder="Rua, Número, Cidade, MG")

if st.sidebar.button("🔍 Validar Origem"):
    lat_o, lon_o = buscar_coordenadas(endereco_origem)
    if lat_o:
        st.session_state['lat_o'], st.session_state['lon_o'] = lat_o, lon_o
        st.sidebar.success("Origem Validada!")
    else:
        st.sidebar.error("Origem não encontrada.")

if 'lat_o' not in st.session_state:
    st.info("👋 **Bem-vindo!** Para começar, valide seu endereço de origem na barra lateral.")
    st.stop()

lat_p, lon_p = st.session_state['lat_o'], st.session_state['lon_o']

# --- LÓGICA DE DADOS ---
df_final = pd.DataFrame()

if modo == "Base Fixa (MG)":
    try:
        df_base = pd.read_csv('municipios_mg.csv')
        st.sidebar.header("🗺️ Filtros MG")
        regioes = sorted(df_base['regiao'].dropna().unique())
        selecao = st.sidebar.multiselect("Regiões de Minas:", options=regioes)
        
        if selecao:
            df_final = df_base[df_base['regiao'].isin(selecao)].copy()
            df_final['nome_exibicao'] = df_final['cidade']
        else:
            st.warning("Selecione ao menos uma região para carregar os dados.")
            st.stop()
    except:
        st.error("Erro ao carregar 'municipios_mg.csv'. Verifique se o arquivo está no GitHub.")
        st.stop()

else: # MODO IMPORTAÇÃO
    arquivo = st.sidebar.file_uploader("Suba seu CSV (Colunas 3, 4 e 8):", type=["csv"])
    if arquivo:
        try:
            df_import = pd.read_csv(arquivo, encoding='iso-8859-1', sep=None, engine='python')
            df_final = pd.DataFrame()
            # Pega Rua (2), Numero (3) e Município (7)
            df_final['rua'] = df_import.iloc[:, 2].astype(str)
            df_final['numero'] = df_import.iloc[:, 3].astype(str)
            df_final['municipio'] = df_import.iloc[:, 7].astype(str)
            df_final['nome_exibicao'] = df_final['municipio']
            
            # Geocodificação Automática
            st.info(f"Buscando coordenadas para {len(df_final)} destinos...")
            prog = st.progress(0)
            lats, lons = [], []
            for i, row in df_final.iterrows():
                end = f"{row['rua']}, {row['numero']}, {row['municipio']}, MG"
                lt, ln = buscar_coordenadas(end)
                if not lt: lt, ln = buscar_coordenadas(f"{row['municipio']}, MG")
                lats.append(lt); lons.append(ln)
                prog.progress((i+1)/len(df_final))
                time.sleep(0.5) # Delay menor para agilizar
            
            df_final['lat'], df_final['lon'] = lats, lons
            df_final = df_final.dropna(subset=['lat', 'lon'])
        except Exception as e:
            st.error(f"Erro na importação: {e}")
            st.stop()
    else:
        st.info("Aguardando upload do arquivo CSV...")
        st.stop()

# --- CÁLCULO DE ROTAS (COMUM PARA OS DOIS MODOS) ---
if not df_final.empty:
    st.sidebar.divider()
    capacidade = st.sidebar.slider("Paradas por caminhão:", 2, 15, 5)
    
    n_clusters = max(1, len(df_final) // capacidade)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_final['ID_Rota'] = kmeans.fit_predict(df_final[['lat', 'lon']])

    relatorio = []
    for id_r, grupo in df_final.groupby('ID_Rota'):
        grupo = grupo.reset_index()
        d_ida = calcular_distancia(lat_p, lon_p, grupo.loc[0, 'lat'], grupo.loc[0, 'lon'])
        for j in range(len(grupo)-1):
            d_ida += calcular_distancia(grupo.loc[j, 'lat'], grupo.loc[j, 'lon'], grupo.loc[j+1, 'lat'], grupo.loc[j+1, 'lon'])
        
        d_ret = calcular_distancia(grupo.loc[len(grupo)-1, 'lat'], grupo.loc[len(grupo)-1, 'lon'], lat_p, lon_p)
        km_t = d_ida + d_ret
        
        relatorio.append({
            'Rota': id_r + 1,
            'Itinerário': ' ➔ '.join(grupo['nome_exibicao'].unique()),
            'KM Ida': round(d_ida, 1),
            'KM Retorno': round(d_ret, 1),
            'KM TOTAL': round(km_t, 1),
            'TEMPO TOTAL': formatar_tempo(km_t)
        })

    # --- EXIBIÇÃO ---
    st.subheader(f"📊 Relatório Otimizado - Partida: {endereco_origem}")
    st.dataframe(pd.DataFrame(relatorio), use_container_width=True)
    
    fig = px.scatter_mapbox(df_final, lat="lat", lon="lon", color="ID_Rota", hover_name="nome_exibicao", zoom=6)
    fig.add_scattermapbox(lat=[lat_p], lon=[lon_p], marker=dict(size=14, color='red'), name="ORIGEM")
    fig.update_layout(mapbox_style="carto-darkmatter", margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

    # Download Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        pd.DataFrame(relatorio).to_excel(writer, index=False)
    st.download_button("📥 Baixar Relatório (Excel)", output.getvalue(), "relatorio_logistica.xlsx")

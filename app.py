import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
from io import BytesIO
import numpy as np
from geopy.geocoders import Nominatim # Nova biblioteca para buscar coordenadas

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Logística MG Pro", page_icon="🔄", layout="wide")

# Função para buscar Latitude e Longitude automaticamente
def buscar_coordenadas(endereco):
    try:
        geolocator = Nominatim(user_agent="logistica_app_mg")
        location = geolocator.geocode(endereco)
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

st.title("🔄 Otimizador Logístico: Saída e Retorno")

# --- BARRA LATERAL (SIDEBAR) ---
st.sidebar.header("📍 Configurações de Origem")

# Campo de endereço
endereco_input = st.sidebar.text_input("Digite o Endereço de Partida:", placeholder="Ex: Rua Simão Antonio, 149, Contagem, MG")

# Botão para buscar coordenadas automaticamente
if st.sidebar.button("🔍 Buscar Coordenadas Automaticamente"):
    if endereco_input:
        with st.spinner('Buscando localização...'):
            lat_buscada, lon_buscada = buscar_coordenadas(endereco_input)
            if lat_buscada:
                st.session_state['lat_f'] = lat_buscada
                st.session_state['lon_f'] = lon_buscada
                st.sidebar.success("Localização encontrada!")
            else:
                st.sidebar.error("Endereço não encontrado. Tente ser mais específico (inclua cidade e estado).")
    else:
        st.sidebar.warning("Digite um endereço primeiro.")

# Campos de coordenadas (preenchidos automaticamente ou manualmente)
col_lat, col_lon = st.sidebar.columns(2)
lat_p = col_lat.number_input("Lat Origem:", value=st.session_state.get('lat_f', 0.0), format="%.4f")
lon_p = col_lon.number_input("Lon Origem:", value=st.session_state.get('lon_f', 0.0), format="%.4f")

st.sidebar.divider()

# --- TRAVA DE SEGURANÇA ---
if endereco_input == "" or lat_p == 0.0:
    st.info("👋 **Bem-vindo!** Digite o endereço acima e clique em **Buscar Coordenadas** para iniciar o planejamento das rotas.")
    st.stop()

# --- INÍCIO DO PROCESSAMENTO ---
try:
    df_base = pd.read_csv('municipios_mg.csv')

    st.sidebar.header("🗺️ Filtros Geográficos")
    todas_regioes = sorted([str(r) for r in df_base['regiao'].dropna().unique()])
    regioes_selecionadas = st.sidebar.multiselect("Filtrar por Região:", options=todas_regioes)
    cidades_por_rota = st.sidebar.slider("Cidades por rota:", 2, 10, 3)

    df = df_base.copy()
    if regioes_selecionadas:
        df = df[df['regiao'].isin(regioes_selecionadas)].copy()

    if not df.empty:
        n_clusters = max(1, len(df) // cidades_por_rota)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['ID_Rota'] = kmeans.fit_predict(df[['lat', 'lon']])

        relatorio = []
        for id_rota, grupo in df.groupby('ID_Rota'):
            grupo = grupo.reset_index()
            dist_ida = calcular_distancia(lat_p, lon_p, grupo.loc[0, 'lat'], grupo.loc[0, 'lon'])
            for i in range(len(grupo) - 1):
                dist_ida += calcular_distancia(grupo.loc[i, 'lat'], grupo.loc[i, 'lon'], grupo.loc[i+1, 'lat'], grupo.loc[i+1, 'lon'])
            
            dist_retorno = calcular_distancia(grupo.loc[len(grupo)-1, 'lat'], grupo.loc[len(grupo)-1, 'lon'], lat_p, lon_p)
            
            relatorio.append({
                'ID_Rota': id_rota + 1,
                'Região': grupo['regiao'].iloc[0],
                'Itinerário': ' ➔ '.join(grupo['cidade']),
                'Km Ida (Total)': round(dist_ida, 1),
                'Tempo Ida': f"{int((dist_ida/60)//1)}h {int((dist_ida/60%1)*60)}min",
                'Km Retorno': round(dist_retorno, 1),
                'Tempo Retorno': f"{int((dist_retorno/60)//1)}h {int((dist_retorno/60%1)*60)}min"
            })

        df_rel = pd.DataFrame(relatorio)
        st.subheader(f"Logística partindo de: {endereco_input}")
        
        fig = px.scatter_mapbox(df, lat="lat", lon="lon", color="regiao", hover_name="cidade", zoom=5.5)
        fig.add_scattermapbox(lat=[lat_p], lon=[lon_p], marker=dict(size=15, color='red'), name="ORIGEM")
        fig.update_layout(mapbox_style="carto-darkmatter", paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df_rel, use_container_width=True)
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_rel.to_excel(writer, index=False)
        st.download_button("📥 Baixar Planilha Excel", output.getvalue(), "rotas_logistica.xlsx")
    else:
        st.warning("Selecione uma região para processar.")

except Exception as e:
    st.error(f"Erro: {e}")import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
from io import BytesIO
import numpy as np
from geopy.geocoders import Nominatim # Nova biblioteca para buscar coordenadas

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Logística MG Pro", page_icon="🔄", layout="wide")

# Função para buscar Latitude e Longitude automaticamente
def buscar_coordenadas(endereco):
    try:
        geolocator = Nominatim(user_agent="logistica_app_mg")
        location = geolocator.geocode(endereco)
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

st.title("🔄 Otimizador Logístico: Saída e Retorno")

# --- BARRA LATERAL (SIDEBAR) ---
st.sidebar.header("📍 Configurações de Origem")

# Campo de endereço
endereco_input = st.sidebar.text_input("Digite o Endereço de Partida:", placeholder="Ex: Rua Simão Antonio, 149, Contagem, MG")

# Botão para buscar coordenadas automaticamente
if st.sidebar.button("🔍 Buscar Coordenadas Automaticamente"):
    if endereco_input:
        with st.spinner('Buscando localização...'):
            lat_buscada, lon_buscada = buscar_coordenadas(endereco_input)
            if lat_buscada:
                st.session_state['lat_f'] = lat_buscada
                st.session_state['lon_f'] = lon_buscada
                st.sidebar.success("Localização encontrada!")
            else:
                st.sidebar.error("Endereço não encontrado. Tente ser mais específico (inclua cidade e estado).")
    else:
        st.sidebar.warning("Digite um endereço primeiro.")

# Campos de coordenadas (preenchidos automaticamente ou manualmente)
col_lat, col_lon = st.sidebar.columns(2)
lat_p = col_lat.number_input("Lat Origem:", value=st.session_state.get('lat_f', 0.0), format="%.4f")
lon_p = col_lon.number_input("Lon Origem:", value=st.session_state.get('lon_f', 0.0), format="%.4f")

st.sidebar.divider()

# --- TRAVA DE SEGURANÇA ---
if endereco_input == "" or lat_p == 0.0:
    st.info("👋 **Bem-vindo!** Digite o endereço acima e clique em **Buscar Coordenadas** para iniciar o planejamento das rotas.")
    st.stop()

# --- INÍCIO DO PROCESSAMENTO ---
try:
    df_base = pd.read_csv('municipios_mg.csv')

    st.sidebar.header("🗺️ Filtros Geográficos")
    todas_regioes = sorted([str(r) for r in df_base['regiao'].dropna().unique()])
    regioes_selecionadas = st.sidebar.multiselect("Filtrar por Região:", options=todas_regioes)
    cidades_por_rota = st.sidebar.slider("Cidades por rota:", 2, 10, 3)

    df = df_base.copy()
    if regioes_selecionadas:
        df = df[df['regiao'].isin(regioes_selecionadas)].copy()

    if not df.empty:
        n_clusters = max(1, len(df) // cidades_por_rota)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['ID_Rota'] = kmeans.fit_predict(df[['lat', 'lon']])

        relatorio = []
        for id_rota, grupo in df.groupby('ID_Rota'):
            grupo = grupo.reset_index()
            dist_ida = calcular_distancia(lat_p, lon_p, grupo.loc[0, 'lat'], grupo.loc[0, 'lon'])
            for i in range(len(grupo) - 1):
                dist_ida += calcular_distancia(grupo.loc[i, 'lat'], grupo.loc[i, 'lon'], grupo.loc[i+1, 'lat'], grupo.loc[i+1, 'lon'])
            
            dist_retorno = calcular_distancia(grupo.loc[len(grupo)-1, 'lat'], grupo.loc[len(grupo)-1, 'lon'], lat_p, lon_p)
            
            relatorio.append({
                'ID_Rota': id_rota + 1,
                'Região': grupo['regiao'].iloc[0],
                'Itinerário': ' ➔ '.join(grupo['cidade']),
                'Km Ida (Total)': round(dist_ida, 1),
                'Tempo Ida': f"{int((dist_ida/60)//1)}h {int((dist_ida/60%1)*60)}min",
                'Km Retorno': round(dist_retorno, 1),
                'Tempo Retorno': f"{int((dist_retorno/60)//1)}h {int((dist_retorno/60%1)*60)}min"
            })

        df_rel = pd.DataFrame(relatorio)
        st.subheader(f"Logística partindo de: {endereco_input}")
        
        fig = px.scatter_mapbox(df, lat="lat", lon="lon", color="regiao", hover_name="cidade", zoom=5.5)
        fig.add_scattermapbox(lat=[lat_p], lon=[lon_p], marker=dict(size=15, color='red'), name="ORIGEM")
        fig.update_layout(mapbox_style="open-street-map", paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df_rel, use_container_width=True)
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_rel.to_excel(writer, index=False)
        st.download_button("📥 Baixar Planilha Excel", output.getvalue(), "rotas_logistica.xlsx")
    else:
        st.warning("Selecione uma região para processar.")

except Exception as e:
    st.error(f"Erro: {e}")

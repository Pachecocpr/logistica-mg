import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
from io import BytesIO
import numpy as np
from geopy.geocoders import Nominatim

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Logística MG Pro", page_icon="🔄", layout="wide")

# Função para buscar coordenadas
def buscar_coordenadas(endereco):
    try:
        geolocator = Nominatim(user_agent="logistica_mg_pacheco_v3")
        location = geolocator.geocode(endereco)
        if location:
            return location.latitude, location.longitude
        return None, None
    except:
        return None, None

# Função de distância (Haversine + 30% de margem para estradas)
def calcular_distancia(lat1, lon1, lat2, lon2):
    r = 6371 
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlambda = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return (2 * r * np.arcsin(np.sqrt(a))) * 1.3

st.title("🔄 Otimizador Logístico:")

# --- BARRA LATERAL (SIDEBAR) ---
st.sidebar.header("📍 Configurações de Origem")

endereco_input = st.sidebar.text_input("Digite o Endereço de Partida:", placeholder="Ex: Rua Simão Antonio, 149, Contagem, MG")

if st.sidebar.button("🔍 Buscar Coordenadas Automaticamente"):
    if endereco_input:
        with st.spinner('Buscando localização...'):
            lat_buscada, lon_buscada = buscar_coordenadas(endereco_input)
            if lat_buscada:
                st.session_state['lat_f'] = lat_buscada
                st.session_state['lon_f'] = lon_buscada
                st.sidebar.success("Localização encontrada!")
            else:
                st.sidebar.error("Endereço não encontrado.")
    else:
        st.sidebar.warning("Digite um endereço primeiro.")

col_lat, col_lon = st.sidebar.columns(2)
lat_p = col_lat.number_input("Lat Origem:", value=st.session_state.get('lat_f', 0.0), format="%.4f")
lon_p = col_lon.number_input("Lon Origem:", value=st.session_state.get('lon_f', 0.0), format="%.4f")

st.sidebar.divider()

if endereco_input == "" or lat_p == 0.0:
    st.info("👋 **Bem-vindo!** Insira a origem, em seguida selecione a região, a quantidade de cidades por rota desejada para o relatório de viagem.")
    st.stop()

# --- PROCESSAMENTO ---
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
            
            # 1. CÁLCULO DE IDA (Origem -> Cidades -> Cidades)
            dist_ida = calcular_distancia(lat_p, lon_p, grupo.loc[0, 'lat'], grupo.loc[0, 'lon'])
            for i in range(len(grupo) - 1):
                dist_ida += calcular_distancia(grupo.loc[i, 'lat'], grupo.loc[i, 'lon'], grupo.loc[i+1, 'lat'], grupo.loc[i+1, 'lon'])
            
            # 2. CÁLCULO DE RETORNO (Última Cidade -> Origem)
            dist_retorno = calcular_distancia(grupo.loc[len(grupo)-1, 'lat'], grupo.loc[len(grupo)-1, 'lon'], lat_p, lon_p)
            
            # 3. CÁLCULOS TOTAIS (SOMAS)
            km_total = dist_ida + dist_retorno
            
            # Formatação de Tempo (Base 60km/h)
            def formatar_tempo(km):
                horas = km / 60
                return f"{int(horas//1)}h {int((horas%1)*60)}min"

            relatorio.append({
                'ID_Rota': id_rota + 1,
                'Região': grupo['regiao'].iloc[0],
                'Itinerário': ' ➔ '.join(grupo['cidade']),
                'Km Ida': round(dist_ida, 1),
                'Tempo Ida': formatar_tempo(dist_ida),
                'Km Retorno': round(dist_retorno, 1),
                'Tempo Retorno': formatar_tempo(dist_retorno),
                'KM TOTAL (SOMA)': round(km_total, 1),
                'TEMPO TOTAL (SOMA)': formatar_tempo(km_total)
            })

        df_rel = pd.DataFrame(relatorio)
        
        # Interface Visual
        st.subheader(f"📍 Origem: {endereco_input}")
        
        fig = px.scatter_mapbox(df, lat="lat", lon="lon", color="regiao", hover_name="cidade", zoom=5.5)
        fig.add_scattermapbox(lat=[lat_p], lon=[lon_p], marker=dict(size=15, color='red'), name="ORIGEM")
        fig.update_layout(mapbox_style="open-street-map", paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

        # Exibição da Tabela com Tudo
        st.write("### Detalhamento das Rotas")
        st.dataframe(df_rel, use_container_width=True)
        
        # Download
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_rel.to_excel(writer, index=False)
        st.download_button("📥 Baixar Planilha Completa (Excel)", output.getvalue(), "logistica_completa_mg.xlsx")
    else:
        st.warning("Selecione uma região para gerar os dados.")

except Exception as e:
    st.error(f"Erro ao processar: {e}")



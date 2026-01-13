import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
from io import BytesIO
import numpy as np

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Logística MG Pro", page_icon="🔄", layout="wide")

# Função de cálculo de distância
def calcular_distancia(lat1, lon1, lat2, lon2):
    r = 6371 
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlambda = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return (2 * r * np.arcsin(np.sqrt(a))) * 1.3 # Fator de correção estradas

st.title("🔄 Otimizador Logístico MG:")

try:
    # Carregar dados
    df_base = pd.read_csv('municipios_mg.csv')

    # --- BARRA LATERAL (SIDEBAR) ---
    st.sidebar.header("📍 Configurações de Origem")
    
    # Campo de Endereço de Origem (Apenas rótulo para o Excel)
    endereco_origem = st.sidebar.text_input("Endereço de Partida:", "Rua Simão Antonio, 149, Contagem - MG")
    
    # Coordenadas de Origem (Onde o cálculo realmente acontece)
    col_lat, col_lon = st.sidebar.columns(2)
    lat_p = col_lat.number_input("Lat Origem:", value=-19.9203, format="%.4f")
    lon_p = col_lon.number_input("Lon Origem:", value=-44.0466, format="%.4f")

    st.sidebar.divider()

    st.sidebar.header("🗺️ Filtros Geográficos")
    # Correção do Erro de Ordenação (Removendo NaNs e convertendo para String)
    todas_regioes = sorted([str(r) for r in df_base['regiao'].dropna().unique()])
    regioes_selecionadas = st.sidebar.multiselect(
        "Filtrar por Região (Opcional):", 
        options=todas_regioes
    )

    cidades_por_rota = st.sidebar.slider("Cidades por rota:", 2, 10, 3)

    # --- FILTRAGEM DOS DADOS ---
    df = df_base.copy()
    if regioes_selecionadas:
        df = df[df['regiao'].isin(regioes_selecionadas)].copy()

    if not df.empty:
        # Lógica de Agrupamento
        n_clusters = max(1, len(df) // cidades_por_rota)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['ID_Rota'] = kmeans.fit_predict(df[['lat', 'lon']])

        relatorio = []
        for id_rota, grupo in df.groupby('ID_Rota'):
            grupo = grupo.reset_index()
            
            # Cálculo de IDA (Origem -> Cidades)
            dist_ida = calcular_distancia(lat_p, lon_p, grupo.loc[0, 'lat'], grupo.loc[0, 'lon'])
            for i in range(len(grupo) - 1):
                dist_ida += calcular_distancia(grupo.loc[i, 'lat'], grupo.loc[i, 'lon'], 
                                               grupo.loc[i+1, 'lat'], grupo.loc[i+1, 'lon'])
            
            # Cálculo de RETORNO (Última Cidade -> Origem)
            ultima_lat = grupo.loc[len(grupo)-1, 'lat']
            ultima_lon = grupo.loc[len(grupo)-1, 'lon']
            dist_retorno = calcular_distancia(ultima_lat, ultima_lon, lat_p, lon_p)
            
            relatorio.append({
                'ID_Rota': id_rota + 1,
                'Região': grupo['regiao'].iloc[0],
                'Origem': endereco_origem,
                'Itinerário': ' ➔ '.join(grupo['cidade']),
                'Km Ida (Total)': round(dist_ida, 1),
                'Tempo Ida': f"{int((dist_ida/60)//1)}h {int((dist_ida/60%1)*60)}min",
                'Km Retorno': round(dist_retorno, 1),
                'Tempo Retorno': f"{int((dist_retorno/60)//1)}h {int((dist_retorno/60%1)*60)}min"
            })

        df_rel = pd.DataFrame(relatorio)

        # --- VISUALIZAÇÃO ---
        st.subheader(f"Logística partindo de: {endereco_origem}")
        
        # Mapa Darkmatter
        fig = px.scatter_mapbox(df, lat="lat", lon="lon", color="regiao", hover_name="cidade", zoom=5.5)
        # Adicionar o ponto de origem no mapa (Ponto Vermelho)
        fig.add_scattermapbox(lat=[lat_p], lon=[lon_p], marker=dict(size=15, color='red'), name="ORIGEM")
        fig.update_layout(mapbox_style="open-street-map", paper_bgcolor="#0e1117", font_color="white", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        # Tabela e Exportação
        st.dataframe(df_rel, use_container_width=True)
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_rel.to_excel(writer, index=False, sheet_name='Rotas_MG')
        
        st.download_button(
            label="📥 Baixar Planilha para Excel",
            data=output.getvalue(),
            file_name="planejamento_logistico_mg.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("Selecione uma região ou verifique se o arquivo de dados está correto.")

except Exception as e:

    st.error(f"Ocorreu um erro: {e}")

import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
from io import BytesIO
import numpy as np
from geopy.geocoders import Nominatim
import time

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Logística Pro v8", page_icon="🚚", layout="wide")

# --- FUNÇÕES TÉCNICAS ---

def buscar_coordenadas(local):
    """Busca coordenadas com User-Agent exclusivo e Timeout estendido para evitar erros de conexão."""
    try:
        # User-agent único para evitar bloqueio por 'Connection Refused'
        geolocator = Nominatim(user_agent="logistica_mg_pacheco_final_v8", timeout=10)
        location = geolocator.geocode(local)
        if location:
            return location.latitude, location.longitude
        return None, None
    except Exception as e:
        # Silencia o erro técnico para não travar a interface
        return None, None

def calcular_distancia(lat1, lon1, lat2, lon2):
    """Cálculo de Haversine com margem de 30% para curvas de estrada."""
    r = 6371 
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlambda = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return (2 * r * np.arcsin(np.sqrt(a))) * 1.3

def formatar_tempo(km):
    """Converte KM em tempo baseado em média de 60km/h."""
    horas = km / 60
    return f"{int(horas//1)}h {int((horas%1)*60)}min"

# --- INTERFACE PRINCIPAL ---
st.title("🔄 Otimizador Logístico Inteligente")
modo = st.sidebar.selectbox("Escolha o Modo de Trabalho:", ["Base Fixa (MG)", "Importar CSV Customizado"])

st.sidebar.divider()

# --- CONFIGURAÇÃO DE ORIGEM ---
st.sidebar.header("📍 Ponto de Partida")
endereco_origem = st.sidebar.text_input("Endereço de Saída:", placeholder="Ex: Rua Simão Antonio, 149, Contagem, MG")

if st.sidebar.button("🔍 Validar e Fixar Origem"):
    if endereco_origem:
        with st.spinner('Validando endereço no mapa...'):
            lat_o, lon_o = buscar_coordenadas(endereco_origem)
            if lat_o:
                st.session_state['lat_o'] = lat_o
                st.session_state['lon_o'] = lon_o
                st.sidebar.success("Origem fixada com sucesso!")
            else:
                st.sidebar.error("Não encontramos este endereço. Tente 'Cidade, MG'.")
    else:
        st.sidebar.warning("Digite um endereço primeiro.")

# Bloqueio de segurança se não houver origem
if 'lat_o' not in st.session_state:
    st.info("👋 **Para começar:** Informe o endereço de partida na barra lateral e clique em **Validar**.")
    st.stop()

lat_p, lon_p = st.session_state['lat_o'], st.session_state['lon_o']

# --- PREPARAÇÃO DOS DADOS ---
df_final = pd.DataFrame()

if modo == "Base Fixa (MG)":
    try:
        df_base = pd.read_csv('municipios_mg.csv')
        st.sidebar.header("🗺️ Filtros de Região")
        regioes = sorted(df_base['regiao'].dropna().unique())
        selecao = st.sidebar.multiselect("Selecione as Regiões:", options=regioes)
        
        if selecao:
            df_final = df_base[df_base['regiao'].isin(selecao)].copy()
            df_final['label_cidade'] = df_final['cidade']
        else:
            st.warning("Selecione uma ou mais regiões para gerar as rotas.")
            st.stop()
    except:
        st.error("Arquivo 'municipios_mg.csv' não encontrado no repositório.")
        st.stop()

else: # MODO IMPORTAÇÃO MANUAL
    st.sidebar.header("📂 Upload de Arquivo")
    arquivo = st.sidebar.file_uploader("Suba o CSV (Colunas 3, 4 e 8)", type=["csv"])
    if arquivo:
        try:
            # Tratamento de encoding para arquivos Excel/Windows
            df_import = pd.read_csv(arquivo, encoding='iso-8859-1', sep=None, engine='python')
            
            # Mapeia colunas solicitadas (Rua: index 2, Num: index 3, Cidade: index 7)
            df_final = pd.DataFrame()
            df_final['rua'] = df_import.iloc[:, 2].astype(str)
            df_final['numero'] = df_import.iloc[:, 3].astype(str)
            df_final['cidade'] = df_import.iloc[:, 7].astype(str)
            df_final['label_cidade'] = df_final['cidade']
            
            # Geocodificação com barra de progresso
            st.info(f"📍 Buscando coordenadas para {len(df_final)} endereços. Isso pode levar alguns segundos...")
            barra = st.progress(0)
            lats, lons = [], []
            
            for i, row in df_final.iterrows():
                # Tenta endereço completo
                endereco_full = f"{row['rua']}, {row['numero']}, {row['cidade']}, MG"
                lt, ln = buscar_coordenadas(endereco_full)
                
                # Se falhar, tenta apenas a cidade
                if not lt:
                    lt, ln = buscar_coordenadas(f"{row['cidade']}, MG")
                
                lats.append(lt)
                lons.append(ln)
                barra.progress((i + 1) / len(df_final))
                time.sleep(1.1) # Pausa obrigatória para não ser bloqueado pelo servidor Nominatim
            
            df_final['lat'], df_final['lon'] = lats, lons
            df_final = df_final.dropna(subset=['lat', 'lon'])
            
            if df_final.empty:
                st.error("Não foi possível localizar nenhum endereço no mapa.")
                st.stop()
                
        except Exception as e:
            st.error(f"Erro ao processar CSV: {e}")
            st.stop()
    else:
        st.info("Aguardando o upload do seu arquivo CSV...")
        st.stop()

# --- GERAÇÃO DE ROTAS (NÚCLEO DO APP) ---
if not df_final.empty:
    st.sidebar.divider()
    cidades_por_caminhao = st.sidebar.slider("Cidades/Paradas por Caminhão:", 2, 15, 5)
    
    # Agrupamento Geográfico
    n_clusters = max(1, len(df_final) // cidades_por_caminhao)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_final['ID_Rota'] = kmeans.fit_predict(df_final[['lat', 'lon']])

    relatorio_dados = []
    for id_r, grupo in df_final.groupby('ID_Rota'):
        grupo = grupo.reset_index()
        
        # Cálculo de IDA
        d_ida = calcular_distancia(lat_p, lon_p, grupo.loc[0, 'lat'], grupo.loc[0, 'lon'])
        for j in range(len(grupo)-1):
            d_ida += calcular_distancia(grupo.loc[j, 'lat'], grupo.loc[j, 'lon'], grupo.loc[j+1, 'lat'], grupo.loc[j+1, 'lon'])
        
        # Cálculo de RETORNO
        d_ret = calcular_distancia(grupo.loc[len(grupo)-1, 'lat'], grupo.loc[len(grupo)-1, 'lon'], lat_p, lon_p)
        km_soma = d_ida + d_ret
        
        relatorio_dados.append({
            'Rota': id_r + 1,
            'Itinerário': ' ➔ '.join(grupo['label_cidade'].unique()),
            'KM Detalhado Ida': round(d_ida, 1),
            'Tempo Estimado Ida': formatar_tempo(d_ida),
            'KM Detalhado Retorno': round(d_ret, 1),
            'Tempo Estimado Retorno': formatar_tempo(d_ret),
            'KM TOTAL VIAGEM': round(km_soma, 1),
            'TEMPO TOTAL VIAGEM': formatar_tempo(km_soma)
        })

    # --- RESULTADOS VISUAIS ---
    st.subheader(f"📊 Relatório de Planejamento - Saída: {endereco_origem}")
    df_rel = pd.DataFrame(relatorio_dados)
    st.dataframe(df_rel, use_container_width=True)
    
    # Mapa
    fig = px.scatter_mapbox(df_final, lat="lat", lon="lon", color="ID_Rota", 
                            hover_name="label_cidade", zoom=5.5)
    fig.add_scattermapbox(lat=[lat_p], lon=[lon_p], marker=dict(size=14, color='red'), name="BASE ORIGEM")
    fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

    # Exportação para Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_rel.to_excel(writer, index=False)
    st.download_button("📥 Baixar Planejamento (Excel)", output.getvalue(), "logistica_mg_completo.xlsx")

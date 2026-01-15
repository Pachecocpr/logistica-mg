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
st.set_page_config(page_title="Logística Pro - Saída Fixa", page_icon="🚚", layout="wide")

# --- COORDENADAS FIXAS (Rua Simão Antônio, 149, Contagem) ---
LAT_ORIGEM = -19.9203
LON_ORIGEM = -44.0466
ENDERECO_FIXO = "Rua Simão Antônio, 149, Contagem - MG"

# --- FUNÇÕES TÉCNICAS ---
def buscar_coordenadas(local):
    """Busca apenas para os destinos do CSV."""
    if not local: return None, None
    try:
        # Rotação de User-Agent para evitar bloqueio nos destinos
        agente = f"log_mg_pacheco_{random.randint(1000, 9999)}"
        geolocator = Nominatim(user_agent=agente, timeout=10)
        location = geolocator.geocode(f"{local}, Minas Gerais, Brazil")
        if location:
            return location.latitude, location.longitude
    except:
        pass
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
st.title("🔄 Otimizador Logístico - Unidade Contagem")

# Exibição fixa da origem na barra lateral
st.sidebar.success(f"📍 **Origem Fixa:**\n{ENDERECO_FIXO}")
st.sidebar.info(f"Coordenadas: {LAT_ORIGEM}, {LON_ORIGEM}")

st.sidebar.divider()
modo = st.sidebar.selectbox("Fonte de Destinos:", ["Importar CSV das Entregas", "Base Fixa (MG)"])

df_final = pd.DataFrame()

if modo == "Importar CSV das Entregas":
    arquivo = st.sidebar.file_uploader("Suba o CSV (Colunas 3, 4 e 8)", type=["csv"])
    if arquivo:
        try:
            df_import = pd.read_csv(arquivo, encoding='iso-8859-1', sep=None, engine='python')
            df_final = pd.DataFrame()
            # Mapeamento conforme sua estrutura: Col 3 (Rua), 4 (Num), 8 (Cidade)
            df_final['rua'] = df_import.iloc[:, 2].astype(str)
            df_final['num'] = df_import.iloc[:, 3].astype(str)
            df_final['cid'] = df_import.iloc[:, 7].astype(str)
            df_final['label'] = df_final['cid']
            
            st.info("📍 Localizando destinos no mapa... (Aguarde)")
            barra = st.progress(0)
            lats, lons = [], []
            for i, r in df_final.iterrows():
                # Tenta endereço completo, senão tenta só a cidade
                lt, ln = buscar_coordenadas(f"{r['rua']}, {r['num']}, {r['cid']}")
                if not lt: lt, ln = buscar_coordenadas(r['cid'])
                lats.append(lt); lons.append(ln)
                barra.progress((i+1)/len(df_final))
                time.sleep(1.1) # Respeita limite do servidor para destinos
            
            df_final['lat'], df_final['lon'] = lats, lons
        except Exception as e:
            st.error(f"Erro no processamento: {e}")
            st.stop()
    else:
        st.info("Aguardando upload do arquivo para iniciar.")
        st.stop()

else: # MODO BASE MG
    try:
        df_base = pd.read_csv('municipios_mg.csv')
        regioes = sorted(df_base['regiao'].dropna().unique())
        selecao = st.sidebar.multiselect("Selecione as Regiões:", options=regioes)
        if selecao:
            df_final = df_base[df_base['regiao'].isin(selecao)].copy()
            df_final['label'] = df_final['cidade']
        else:
            st.stop()
    except:
        st.error("Arquivo municipios_mg.csv não encontrado.")
        st.stop()

# --- PROCESSAMENTO DE ROTAS ---
df_final = df_final.dropna(subset=['lat', 'lon'])

if not df_final.empty:
    n_v = st.sidebar.slider("Cidades/Entregas por veículo:", 2, 15, 5)
    n_clusters = max(1, len(df_final) // n_v)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_final['ID_Rota'] = kmeans.fit_predict(df_final[['lat', 'lon']])

    resumo = []
    for id_r, gp in df_final.groupby('ID_Rota'):
        gp = gp.reset_index()
        # Distância Ida (Saindo de Contagem)
        dist = calcular_distancia(LAT_ORIGEM, LON_ORIGEM, gp.loc[0, 'lat'], gp.loc[0, 'lon'])
        # Entre destinos
        for j in range(len(gp)-1):
            dist += calcular_distancia(gp.loc[j, 'lat'], gp.loc[j, 'lon'], gp.loc[j+1, 'lat'], gp.loc[j+1, 'lon'])
        # Retorno (Voltando para Contagem)
        dist += calcular_distancia(gp.loc[len(gp)-1, 'lat'], gp.loc[len(gp)-1, 'lon'], LAT_ORIGEM, LON_ORIGEM)
        
        resumo.append({
            'Rota': id_r + 1,
            'KM Total (Ida/Volta)': round(dist, 1),
            'Tempo Estimado': formatar_tempo(dist),
            'Itinerário': ' > '.join(gp['label'].unique())
        })

    st.write("### 📊 Relatório de Rotas Otimizadas")
    st.dataframe(pd.DataFrame(resumo), use_container_width=True)
    
    # Mapa
    fig = px.scatter_mapbox(df_final, lat="lat", lon="lon", color="ID_Rota", 
                            hover_name="label", zoom=6)
    fig.add_scattermapbox(lat=[LAT_ORIGEM], lon=[LON_ORIGEM], 
                          marker=dict(size=15, color='red'), name="BASE CONTAGEM")
    fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

    # Download Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        pd.DataFrame(resumo).to_excel(writer, index=False)
    st.download_button("📥 Baixar Planilha de Rotas", output.getvalue(), "rotas_contagem.xlsx")

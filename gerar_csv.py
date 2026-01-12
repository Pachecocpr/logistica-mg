import pandas as pd

# URL com dados que incluem as microrregiões/mesorregiões
url = "https://raw.githubusercontent.com/kelvins/municipios-brasileiros/main/csv/municipios.csv"
df_brasil = pd.read_csv(url)

# Filtrar MG (Código 31)
df_mg = df_brasil[df_brasil['codigo_uf'] == 31].copy()

# Dicionário manual para simplificar (Mesorregiões de MG)
# Para um sistema real, usaríamos a API do IBGE, mas aqui vamos simular a coluna
# com base em faixas de latitude/longitude para o teste ou carregar de uma base completa.
# NOTA: O arquivo do Kelvins não tem 'regiao', então vamos baixar do IBGE diretamente:

url_ibge = "https://servicodados.ibge.gov.br/api/v1/localidades/estados/31/municipios"
import requests
res = requests.get(url_ibge).json()
regioes = {item['nome']: item['microrregiao']['mesorregiao']['nome'] for item in res}

df_mg['regiao'] = df_mg['nome'].map(regioes)
df_final = df_mg[['nome', 'latitude', 'longitude', 'regiao']].rename(
    columns={'nome': 'cidade', 'latitude': 'lat', 'longitude': 'lon'}
)

df_final.to_csv('municipios_mg.csv', index=False)
print("Dados com regiões atualizados!")
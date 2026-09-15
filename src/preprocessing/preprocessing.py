import pandas as pd
from pathlib import Path
import pyarrow, fastparquet

# Caminho base para os arquivos de dados
BASE_DATA = Path(__file__).resolve().parent.parent.parent / "data"

# Método para incrementar a tabela resultante da camada gold 'ml_aluno' com dados socioeconomicos,
# buscando caracterizar melhor municípios e UFs brasileiras.
def constroi_df_resultante(df, df_estatisticas_escolares, df_dados_socioeconomicos):
  print(f"[INFO:PREPROC] Integrando dados da camada Gold com indicadores socioeconômicos.")
  df_dados_socioeconomicos.loc[df_dados_socioeconomicos['gini_uf'] > 1, 'gini_uf'] /= 1000

  df_estatisticas_escolares = df_estatisticas_escolares \
  .filter(items=["ano", "id_escola", "total_alunos", "total_alunos_presentes", "percentual_faltantes", "desvio_padrao_proficiencia"]) \
  .rename(columns={
      "total_alunos": "total_participantes_escola",
      "total_alunos_presentes": "total_presentes_escola",
      "percentual_faltantes": "percentual_faltantes_escola",
      "desvio_padrao_proficiencia": "desvio_padrao_proficiencia_escola"})

  df = df.merge(df_estatisticas_escolares, on=["id_escola", "ano"], how="left")

  df['nome_municipio'] = df['nome_municipio'].str.upper()
  df_dados_socioeconomicos = df_dados_socioeconomicos.rename(columns={"municipio": "nome_municipio"})

  df_dados_socioeconomicos['nome_municipio'] = df_dados_socioeconomicos['nome_municipio'].replace('´', "'", regex=True)

  df = df.merge(df_dados_socioeconomicos, on=["nome_municipio", "sigla_uf", "ano"], how="left")

  df['feat_rate_escola_mun'] = df['feat_media_proficiencia_escola'] / df['feat_media_portugues_municipio']
  df['feat_rate_mun_estado'] = df['feat_media_portugues_municipio'] / df['feat_media_portugues_estado']

  return df.drop(columns=["_gold_processed_at"])

"""# Encontramos alguns valores faltantes para média de português municipal e estadual, que não haviam sido percebidos durante a construção do pipeline de dados.

###**Estadual**:
    Foram encontrados valores faltantes para sigla_uf = ("Não encontrado", "DF", "TO")
      
        Não encontrado (489 valores): não houve match entre o id do município reportado para aquela escola e o mapeamento de ids municipais fornecido.
        Decisão - descartar.
        
        DF (22111 valores): ao consultar a tabela de dados brutos estaduais do Tech Challenge 2, percebemos que ela não continha dados para o Distrito Federal.
        Decisão - utilizar a média do único município que compõe o estado (Brasília).

        TO (24 valores): uma investigação mais detalhada no pipeline é necessária para entender o motivo pelo qual registros da rede privada da cidade de Gurupi, em 2024, ficaram com valores nulos para média municipal e estadual.
        Decisão - popular com o valor conhecido de média estadual para TO em 2024 (742.86).

###**Municipal**
    Para os municípios, a estratégia foi atribuir aos registros faltantes o valor da média das outras redes de ensino naquele mesmo ano.
    Quando não disponível, foi atribuída a média estadual naquele ano.
"""

def imputa_faltantes(df):
  print(f"[INFO:PREPROC] Imputando valores faltantes.")
  # Descartando não encontrados.
  df = df[df['sigla_uf'] != "Não encontrado"]

  # Atribuindo o valor conhecido de média estadual para TO em 2024.
  MEDIA_TO_2024 = 742.86
  df.loc[df['feat_media_portugues_estado'].isna() & (df['sigla_uf'] == 'TO'), 'feat_media_portugues_estado'] = MEDIA_TO_2024

  # Atribuindo a média ao Distrito Federal.
  MEDIA_BRASILIA_2024 = 743.01
  df.loc[(df['sigla_uf'] == 'DF') & (df['ano'] == 2024), 'feat_media_portugues_estado'] = MEDIA_BRASILIA_2024

  # Olhando para a media municipal.
  mun_medias_faltantes = df[df['feat_media_portugues_municipio'].isna() == True]["nome_municipio"].unique().tolist()

  for municipio in mun_medias_faltantes:
    for ano in [2023, 2024]:

      total_municipais_nulas = df.loc[(df['nome_municipio'] == municipio) & (df['ano'] == ano), 'feat_media_portugues_municipio'].isna().sum()

      # checa primeiro se existe algum registro não nulo para média do município naquele ano.
      # caso todos sejam nulos, é necessario atribuir a média estadual.
      if df.loc[(df['nome_municipio'] == municipio) & (df['ano'] == ano), 'feat_media_portugues_municipio'].count() == 0:
        df.loc[(df['nome_municipio'] == municipio) & (df['ano'] == ano), 'feat_media_portugues_municipio'] = \
          df.loc[(df['nome_municipio'] == municipio) & (df['ano'] == ano), 'feat_media_portugues_estado'].unique().mean()

      elif total_municipais_nulas > 0:

        df.loc[
            (df['feat_media_portugues_municipio'].isna() == True) & (df['nome_municipio'] == municipio) & (df['ano'] == ano), 'feat_media_portugues_municipio'] = \
            df.loc[(df['feat_media_portugues_municipio'].isna() == False) & (df['nome_municipio'] == municipio) & \
            (df['ano'] == ano), 'feat_media_portugues_municipio'].unique().mean()

  return df

def run_preprocessing(df_ml_aluno, df_estatisticas_escolares, df_dados_socioeconomicos, filter):
  
  df_ml_aluno = imputa_faltantes(df_ml_aluno)

  df_resultante = constroi_df_resultante(df_ml_aluno, df_estatisticas_escolares, df_dados_socioeconomicos)

  if filter:
    df_resultante = df_resultante.filter(items=['ano', 'sigla_uf', 'feat_rede_encoded', 'feat_peso_aluno', 'feat_media_proficiencia_escola', 'desvio_padrao_proficiencia_escola', \
                                            'feat_rate_escola_mun', 'total_participantes_escola', 'total_presentes_escola', 'percentual_faltantes_escola', 'feat_media_portugues_municipio', 'feat_rate_mun_estado', \
                                            'feat_media_portugues_estado', 'proporcao_pbf_municipio', 'ideb_medio_municipio', \
                                            'tendencia_ideb_municipio', 'gini_uf', 'target_alfabetizado']) \
  
  df_resultante = df_resultante.rename(columns={'desvio_padrao_proficiencia_escola': 'feat_std_proficiencia_escola', 'total_participantes_escola': 'feat_total_participantes_escola', \
    'total_presentes_escola': 'feat_total_presentes_escola','percentual_faltantes_escola': 'feat_percentual_faltantes_escola', 'proporcao_pbf_municipio': 'feat_proporcao_pbf_municipio', \
      'ideb_medio_municipio': 'feat_ideb_medio_municipio', 'tendencia_ideb_municipio': 'feat_tendencia_ideb_municipio', \
        'gini_uf': 'feat_gini_uf', 'sigla_uf': 'feat_uf'})

  df_resultante['feat_coef_variacao_escola'] = df_resultante['feat_std_proficiencia_escola'] / df_resultante['feat_media_proficiencia_escola']

  return df_resultante

def preprocess(filter = True):
    print(f"[INFO:PREPROC] Iniciando leitura dos dados de entrada.")
    try:
      df_ml_aluno = pd.read_parquet(f'{BASE_DATA}/ml_aluno', engine='pyarrow')
      df_estatisticas_escolares = pd.read_parquet(f'{BASE_DATA}/estatisticas_escolares', engine='pyarrow')
      df_dados_socioeconomicos = pd.read_csv(f"{BASE_DATA}/br_dados_socioeconomicos.csv", delimiter=';')
    except Exception as e:
      print(f"[ERROR:PREPROC] Erro na leitura dos dados de entrada: {str(e)}")
      raise e
    print(f"[INFO:PREPROC] Leitura dos dados completa. Prosseguindo com pré-processamento.")
    df_resultante = run_preprocessing(df_ml_aluno, df_estatisticas_escolares, df_dados_socioeconomicos, filter)
    print(f"[INFO:PREPROC] Pré-processamento concluído.\n")
    return df_resultante
# Predição de Alfabetização Infantil e Análise de Risco Educacional

Projeto desenvolvido para a **Fase 3 do Tech Challenge da Pós-Tech**, focada na construção de um pipeline preditivo e interpretável em **Machine Learning** utilizando dados de desempenho escolar. O objetivo é identificar padrões de alfabetização de alunos no 2º ano do Ensino Fundamental, fornecendo estrutura para o direcionamento de recursos educacionais e formulação de políticas públicas.



## **Estrutura do Repositório**

```text
├── data/                         # Arquivos de dados
├── notebooks/                    # Análise Exploratória de Dados (EDA)
├── src/                          # Módulos em Python da solução
│   ├── preprocessing/            # Limpeza, imputações e engenharia de atributos
│   ├── modeling/                 # Pipeline de treino, divisão temporal e Optuna
│   ├── evaluation/               # Avaliação e geração de métricas
│   └── visualization/            # Gráficos de SHAP Values, Matriz de Confusão e ROC-AUC
├── reports/                      # Relatório com análise da EDA e dos resultados
├── images/                       # Visualizações de dados
├── requirements.txt              
└── README.md

```

## **Engenharia de Atributos e Prevenção de Data Leakage**

### **1. Divisão Out-of-Time (Validação Temporal)**

Visando evitar *Data Leakage*, a divisão de dados foi feita como segue:

* **Base de Desenvolvimento (2023):** Dividida em $80\%$ treino e $20\%$ validação interna estratificada.


* **Base Holdout de Teste (2024):** $1.851.630$ alunos reservados exclusivamente para a avaliação final de métricas.



### **2. Engenharia de Recursos e Imputações Tratadas**

* **Tratamento de Exceções Regionais:** Imputação de dados faltantes com médias estaduais históricas para casos como o Distrito Federal (devido à ausência de dados estaduais na fonte primária) e reparos pontuais em municípios específicos.


* **Razões Hierárquicas:**
* `feat_rate_escola_mun`: Desempenho relativo da escola em relação ao município (`média_escola / média_município`).


* `feat_rate_mun_estado`: Desempenho relativo do município em relação ao estado (`média_município / média_estado`).



* **Desigualdade Escolar:** `feat_coef_variacao_escola` (Desvio / Média), capturando a dispersão do desempenho dentro do mesmo colégio.

* Além dos dados obtidos da camada Gold do **Tech Challenge 2**, a base de dados utilizada para treinamento foi incrementada com dados referentes ao **IDEB municipal**, proporção de inscritos no programa **Bolsa Família** por município, e **Índice Gini** estadual. Essas informações foram incluídas visando fornecer uma melhor visão socioeconômica de cada estudante.

* Referência dos dados adicionais utilizados:

        Índice de Gini Estadual (acesso em Set 2026) -
         https://www.ipeadata.gov.br/ExibeSerieR.aspx?stub=1&serid=2096726935&MINDATA=2012&MAXDATA=2030&TNIVID=2&TPAID=1&module=S

        IDEB municipal (acesso em Set 2026) - https://basedosdados.org/dataset/96eab476-5d30-459b-82be-f888d4d0d6b9?table=8873f899-fdca-4ae4-808a-cdaaa1735d6a

        Dados Bolsa Família (acesso em Set 2026) - https://aplicacoes.cidadania.gov.br/vis/data3/data-explorer.php#

        População brasileira (acesso em Set 2026) - https://basedosdados.org/dataset/d30222ad-7a5c-4778-a1ec-f0785371d1ca?table=0c279444-165b-41da-92cd-50fd7e66baa1



## **Modelagem Preditiva e Otimização**

### **Por que o LightGBM?**

A escolha do **LightGBM (Light Gradient Boosting Machine)** como algoritmo baseou-se nas seguintes vantagens estruturais:

1. **Escalabilidade com Alta Volumetria:** Trata com  eficiência o grande volume de dados do conjunto (mais de $1.85$ milhão de alunos na base de teste de 2024), garantindo tempos de treino reduzidos.


2. **Suporte Nativo a Variáveis Categóricas:** Trata variáveis como `feat_uf` e `feat_rede_encoded` sem a necessidade de expansão por *One-Hot Encoding*, otimizando o consumo de memória.


3. **Explicabilidade via TreeSHAP:** Permite auditoria completa e rápida do comportamento das variáveis por meio de *TreeExplainer*, para interpretabilidade do modelo.



### **Estratégia de Otimização ($F_{0.5}\text{-Score}$)**

A otimização de hiperparâmetros foi conduzida via **Optuna** com 30 *trials*.

* **Função Objetivo:** Maximização do $F_{0.5}\text{-Score}$.


* **Justificativa ($F_{0.5}$):** O ajuste da métrica $F_{0.5}$ penaliza Falsos Positivos com maior rigor, garantindo que a lista de alunos sinalizados como alfabetizados possua alta precisão. Rotular um aluno como "Alfabetizado" quando ele na verdade necessita de apoio (Falso Positivo) muito provavelmente acarretará a despriorização indevida desse estudante em programas de reforço, portanto a escolha por otimizar a precisão do modelo visa reduzir as chances de que isso aconteça.



## **Resultados no Conjunto de Teste (Ano 2024)**

Com a aplicação do limiar ótimo encontrado na validação ($0.5796$), o modelo atingiu os seguintes resultados no conjunto de teste de 2024:

| Métrica | Valor Obtido |
| --- | --- |
| **Limiar de Decisão Aplicado** | `0.5796`<br> |
| **Precisão (Classe 1 - Alfabetizado)** | **74.69%**<br> |
| **Recall (Classe 1 - Alfabetizado)** | **65.91%**<br> |
| **$F_{0.5}\text{-Score}$** | **0.7275**<br> |
| **Acurácia Global** | **66.27%**<br> |
| **ROC-AUC Score** | **0.6635**<br> |

### **Matriz de Confusão Final**

* **Verdadeiros Negativos (VN):** 497.424


* **Falsos Positivos (FP):** 247.228


* **Falsos Negativos (FN):** 377.372


* **Verdadeiros Positivos (VP):** 729.606



## **Interpretação e Explicabilidade (SHAP Values)**

A análise de explicabilidade via **SHAP (`TreeExplainer`)** foi calculada sobre uma amostra representativa de $100.000$ alunos do conjunto de teste:

* A variável de maior impacto preditivo isolado no modelo é a média de proficiência da escola (`feat_media_proficiencia_escola`), confirmando que o ambiente imediato da escola é o principal determinante.
* A dispersão e desvio de aprendizado dentro do próprio colégio (`feat_coef_variacao_escola` e `feat_std_proficiencia_escola`) demonstraram ser fatores relevantes para diferenciar os estudantes.
* **Fatores Regionais:** O estado de localização (`feat_uf`) e o desempenho municipal (`feat_media_portugues_municipio`) atuam como condicionantes estruturais secundários de forte influência.


## **Como Executar o Projeto**

### **1. Pré-requisitos**

Certifique-se de extrair os dados contidos em `./data/data_files.zip`, e que esses dados estejam dispostos dentro do diretório `./data`.

### **2. Instalação das Dependências**

```bash
pip install -r requirements.txt
```

### **3. Execução da Avaliação e Gráficos**

Para executar o pipeline completo de avaliação nos dados de teste e gerar os relatórios e gráficos:

```bash
python ./src/evaluation/evaluation.py
```
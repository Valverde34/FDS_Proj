# FDS_Proj
Projeto de Fundamentos de Ciências de Dados - Previsão de Doenças Cardíacas em Crianças

## Progresso do Projeto

### 1. Raw Data (Dados Brutos)
- [x] **Script:** `raw_data_analysis.py`
- [x] Descrição dos tipos de variáveis (nominal, ordinal, interval, ratio, binary)
  - [x] `raw_01_tipos_variaveis.csv`
- [x] Análise preliminar dos dados originais (UCMF.xls)
- [x] Histogramas dos dados brutos
  - [x] `raw_04_histograms_raw.png`
- [x] Boxplots dos dados brutos
  - [x] `raw_05_boxplots_raw.png`
- [x] Lista de problemas identificados
  - [x] `raw_02_problemas_identificados.csv`
- [x] Estatísticas descritivas dos dados brutos
  - [x] `raw_03_statistics_raw.csv`
- [x] Distribuição TARGET (dados brutos)
  - [x] `raw_06_target_distribution_raw.png`

### 2. Preprocessed Data (Pré-processamento)
- [x] **Script:** `p_process.py`
- [x] Remoção de duplicados (ID)
- [x] Limpeza de idades (0-19 anos)
- [x] Limpeza de Peso, Altura, IMC
- [x] Limpeza de Pressão Arterial e Frequência Cardíaca
- [x] Padronização de variáveis categóricas (SEXO, TARGET, PULSOS, SOPRO, B2)
- [x] Remoção de colunas irrelevantes (HDA2, datas)
- [x] **Output:** `UCMF_CLEAN.csv` 


### 3. Basic Description (Análise Descritiva - Dados Limpos)
- [x] **Script:** `eda_basic.py`
- [x] Estatísticas descritivas completas
  - [x] `01_statistics_summary.csv`
- [x] Histogramas com curvas de densidade
  - [x] `02_histograms_density.png`
- [x] Boxplots para identificar outliers
  - [x] `03_boxplots.png`
- [x] Medidas de dispersão (Range, IQR, Variance, CV, MAD)
  - [x] `04_spread_measures.csv`
- [x] Distribuição da variável TARGET
  - [x] `05_target_distribution.png`
- [x] Distribuições de variáveis categóricas
  - [x] `06_categorical_distributions.png`
- [x] Testes de normalidade
  - [x] `07_normality_tests.csv`

### 4. Analysis (Análise)

#### 4.1 Bivariate Analysis (Análise Bivariada)
- [ ] Correlações (Pearson, Spearman, Kendall)
- [ ] Matriz de correlação (heatmap)
- [ ] Regressão simples
- [ ] Comparações entre Normal vs Anormal (t-tests)

#### 4.2 Multivariate Analysis (Análise Multivariada)
- [ ] Regressão múltipla
- [ ] Mutual Information
- [ ] Cluster Analysis
- [ ] PCA (Principal Component Analysis)

#### 4.3 Feature Selection
- [ ] Seleção de variáveis relevantes e não-redundantes
- [ ] Feature importance ranking

### 5. Predictive Models (Modelação Preditiva)
- [ ] Decision Trees
- [ ] Support Vector Machines (SVM)
- [ ] Bayesian Networks
- [ ] Random Forest (opcional)
- [ ] Neural Networks (opcional)

### 6. Evaluation & Comparison
- [ ] Métricas de avaliação:
  - [ ] Accuracy, Precision, Recall, F1-Score
  - [ ] ROC Curves e AUC
  - [ ] Confusion Matrix
- [ ] Comparação de modelos
- [ ] Seleção do melhor modelo

### 7. Report & Discussion
- [ ] Relatório final (PDF)
- [ ] Discussão de resultados
- [ ] Conclusões principais

# FDS_Proj
Projeto de Fundamentos de Ciências de Dados - Previsão de Doenças Cardíacas em Crianças

## Progresso do Projeto

### 1. Raw Data (Dados Brutos) 
- [x] **Script:** `raw_data_analysis.py` 
- [x] Descrição dos tipos de variáveis (nominal, ordinal, interval, ratio, binary)
  - [x] `raw_01_tipos_variaveis.csv` 
- [x] Análise preliminar dos dados originais (UCMF.xls)
- [x] Identificação de problemas
  - [x] `raw_02_problemas_identificados.csv`
  - [x] `raw_02b_missing_values.csv` 
  - [x] `raw_02c_impossible_physiological_values.csv` 
- [x] Estatísticas descritivas completas (15 métricas)
  - [x] `raw_03_statistics_raw.csv` 
- [x] Visualizações numéricas
  - [x] `raw_04_histograms_raw.png` 
  - [x] `raw_05_boxplots_raw.png` 
  - [x] `raw_07_density_kde.png`
- [x] Análise de variáveis categóricas (TODAS as 10 variáveis)
  - [x] `raw_08_categorical_analysis.csv` 
  - [x] `raw_09_rare_categories.csv` 
  - [x] `raw_10_categorical_distributions.png` 
- [x] Distribuição TARGET (dados brutos)
  - [x] `raw_06_target_distribution_raw.png` 


### 2. Preprocessed Data (Pré-processamento)
- [x] **Script:** `pre_process.py`
- [x] Remoção de duplicados (ID)
- [x] Limpeza de idades (0-19 anos)
- [x] Limpeza de Peso, Altura, IMC
- [x] Limpeza de Pressão Arterial e Frequência Cardíaca
- [x] Padronização de variáveis categóricas (SEXO, TARGET, PULSOS, SOPRO, B2)
- [x] Remoção de colunas irrelevantes (HDA2, datas)
- [x] **Output:** `UCMF_CLEAN.xlsx` 


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
- [x] **Script:** `bivariate_analysis.py` 
- [x] Correlações (Pearson, Spearman, Kendall)
  - [x] `08_correlation_pearson.csv`
  - [x] `09_correlation_spearman.csv`
  - [x] `10_correlation_kendall.csv`
- [x] Matriz de correlação (heatmap)
  - [x] `11_correlation_heatmaps.png`
- [x] Regressão simples
  - [x] `12_strong_correlations.csv`
  - [x] `13_simple_regressions.csv`
  - [x] `14_scatterplots_correlations.png`
- [x] Comparações entre Normal vs Anormal (t-tests)
  - [x] `15_ttest_normal_vs_anormal.csv`
  - [x] `16_boxplots_normal_vs_anormal.png`
  - [x] `17_violinplots_normal_vs_anormal.png`
- [x] Análise Numérica vs Categórica *(NOVO)*
  - [x] `17b_anova_categorical_vs_numeric.csv` 
- [x] Correlação Parcial (controlando IDADE) 
  - [x] `17c_partial_correlations.csv`
- [x] Pairplot (visão geral de todas as relações) 
  - [x] `17d_pairplot_all_variables.png`

#### 4.2 Multivariate Analysis (Análise Multivariada)
- [x] **Script:** `multivariate_analysis.py`
- [x] Regressão Logística (binária)
  - [x] `18_logistic_regression_coefficients.csv`
  - [x] `19_logistic_regression_plot.png`
- [x] Mutual Information
  - [x] `20_mutual_information.csv`
  - [x] `21_mutual_information_plot.png`
- [x] PCA (Principal Component Analysis)
  - [x] `22_pca_variance.csv`
  - [x] `23_pca_variance_plots.png`
  - [x] `24_pca_loadings.csv`
  - [x] `25_pca_biplot.png`
- [x] Cluster Analysis
  - [x] `26_cluster_metrics.csv`
  - [x] `27_cluster_evaluation.png`
  - [x] `28_cluster_vs_target.csv`
  - [x] `29_clusters_vs_target_pca.png`
  - [x] `30_cluster_profiles.csv`
  - [x] `31_cluster_profiles_heatmap.png`

#### 4.3 Feature Selection
- [x] **Script:** `feature_selection.py`
- [x] Seleção de variáveis relevantes e não-redundantes
  - [x] `31_multicolinearidade_analysis.csv`
  - [x] `32_feature_variances.csv`
  - [x] `33_univariate_ftest.csv`
  - [x] `34_mutual_information_scores.csv`
  - [x] `35_random_forest_importance.csv`
  - [x] `36_gradient_boosting_importance.csv`
  - [x] `37_rfe_rankings.csv`
  - [x] `38_consolidated_feature_ranking.csv`
- [x] Feature importance ranking
  - [x] `39_feature_selection_comparison.png`
  - [x] `40_feature_selection_heatmap.png`
  - [x] `41_cv_feature_selection.csv`
  - [x] `42_cv_performance_vs_features.png`

### 5. Predictive Models (Modelação Preditiva)
- [x] **Script:** `predictive_models.py`
- [x] Decision Trees
- [x] Support Vector Machines (SVM)
- [x] Bayesian Networks (Naive Bayes)
- [x] Random Forest
- [x] Neural Networks (MLP)

### 6. Evaluation & Comparison
- [x] **Integrado em:** `predictive_models.py`
- [x] Métricas de avaliação:
  - [x] Accuracy, Precision, Recall, F1-Score
    - [x] `46_metrics_comparison.png`
  - [x] ROC Curves e AUC
    - [x] `45_roc_curves.png`
  - [x] Confusion Matrix
    - [x] `44_confusion_matrices.png`
  - [x] Classification Reports
    - [x] `49_classification_reports.csv`
  - [x] Comparação de modelos
    - [x] `43_model_comparison.csv`
  - [x] Cross-validation
    - [x] `50_cross_validation_results.csv`
  - [x] Feature Importance
    - [x] `47_feature_importance_models.png`
    - [x] `48_decision_tree_visualization.png`

### 7. Report & Discussion
- [ ] Relatório final (PDF)
- [ ] Discussão de resultados
- [ ] Conclusões principais

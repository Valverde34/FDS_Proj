# FDS_Proj
Projeto de Fundamentos de Ciências de Dados - Previsão de Doenças Cardíacas em Crianças

## Pipeline de Análise

### 1. Raw Data Analysis → `raw_data_analysis.py`

Análise exploratória dos dados brutos (UCMF.xls) antes de qualquer tratamento.

**Outputs gerados:**
```
raw_01_tipos_variaveis.csv              # Classificação de variáveis (nominal, ordinal, etc)
raw_02_problemas_identificados.csv      # Lista de problemas encontrados
raw_02b_missing_values.csv              # Análise de valores ausentes
raw_02c_impossible_physiological_values.csv  # Valores fisiologicamente impossíveis
raw_03_statistics_raw.csv               # 15 métricas estatísticas descritivas
raw_04_histograms_raw.png               # Histogramas das variáveis numéricas
raw_05_boxplots_raw.png                 # Boxplots para deteção de outliers
raw_06_target_distribution_raw.png      # Distribuição da variável TARGET
raw_07_density_kde.png                  # Curvas de densidade KDE
raw_08_categorical_analysis.csv         # Análise completa de 10 variáveis categóricas
raw_09_rare_categories.csv              # Categorias raras identificadas
raw_10_categorical_distributions.png    # Gráficos de distribuições categóricas
```

---

### 2. Data Preprocessing → `pre_process.py`

Limpeza e preparação dos dados para análise.

**Transformações aplicadas:**
- Remoção de duplicados (coluna ID)
- Validação de idades (0-19 anos)
- Limpeza de Peso, Altura, IMC
- Validação de PA Sistólica, PA Diastólica, FC
- Padronização de categóricas (SEXO, TARGET, PULSOS, SOPRO, B2)
- Remoção de colunas irrelevantes

**Output gerado:**
```
UCMF_CLEAN.xlsx                         # Dataset limpo e pronto para análise
```

---

### 3. Exploratory Data Analysis → `eda_basic.py`

Análise descritiva detalhada dos dados limpos.

**Outputs gerados:**
```
01_statistics_summary.csv               # Estatísticas completas (média, mediana, std, etc)
02_histograms_density.png               # Histogramas com curvas de densidade
03_boxplots.png                         # Boxplots de todas variáveis numéricas
04_spread_measures.csv                  # Medidas de dispersão (Range, IQR, CV, MAD)
05_target_distribution.png              # Distribuição Normal vs Anormal
06_categorical_distributions.png        # Distribuições de variáveis categóricas
07_normality_tests.csv                  # Testes Shapiro-Wilk e Kolmogorov-Smirnov
```

---

### 4. Bivariate Analysis → `bivariate_analysis.py`

Análise de relações entre pares de variáveis.

**Outputs gerados:**
```
08_correlation_pearson.csv              # Correlações de Pearson
09_correlation_spearman.csv             # Correlações de Spearman
10_correlation_kendall.csv              # Correlações de Kendall
11_correlation_heatmaps.png             # Heatmaps das 3 correlações
12_strong_correlations.csv              # Correlações fortes (|r| > 0.5)
13_simple_regressions.csv               # Regressões lineares simples
14_scatterplots_correlations.png        # Scatterplots com linhas de regressão
15_ttest_normal_vs_anormal.csv          # T-tests comparando grupos
16_boxplots_normal_vs_anormal.png       # Boxplots por grupo (Normal/Anormal)
17_violinplots_normal_vs_anormal.png    # Violinplots por grupo
17b_anova_categorical_vs_numeric.csv    # ANOVA para variáveis categóricas
17c_partial_correlations.csv            # Correlações parciais (controlando IDADE)
17d_pairplot_all_variables.png          # Matriz de scatterplots completa
```

---

### 5. Multivariate Analysis → `multivariate_analysis.py`

Análise de relações complexas entre múltiplas variáveis.

**Outputs gerados:**
```
18_logistic_regression_coefficients.csv # Coeficientes e Odds Ratios
19_logistic_regression_plot.png         # Gráficos de coeficientes e ORs
20_mutual_information.csv               # Scores de Mutual Information
21_mutual_information_plot.png          # Ranking de importância das variáveis
22_pca_variance.csv                     # Variância explicada por componente
23_pca_variance_plots.png               # Scree plot e variância cumulativa
24_pca_loadings.csv                     # Contribuição das variáveis nas PCs
25_pca_biplot.png                       # Biplot PC1 vs PC2
26_cluster_metrics.csv                  # Métricas de clustering (Silhouette, Davies-Bouldin)
27_cluster_evaluation.png               # Elbow method e análise de clusters
28_cluster_vs_target.csv                # Distribuição TARGET por cluster
29_clusters_vs_target_pca.png           # Clusters vs TARGET no espaço PCA
30_cluster_profiles.csv                 # Perfil médio de cada cluster
31_cluster_profiles_heatmap.png         # Heatmap dos perfis de clusters
```

---

### 6. Feature Selection → `feature_selection.py`

Identificação das variáveis mais relevantes para predição.

**Outputs gerados:**
```
31_multicolinearidade_analysis.csv      # Análise de multicolinearidade (VIF)
32_feature_variances.csv                # Variâncias das features
33_univariate_ftest.csv                 # F-test univariado
34_mutual_information_scores.csv        # Scores de MI para classificação
35_random_forest_importance.csv         # Importância via Random Forest
36_gradient_boosting_importance.csv     # Importância via Gradient Boosting
37_rfe_rankings.csv                     # Ranking via Recursive Feature Elimination
38_consolidated_feature_ranking.csv     # Ranking consolidado de todas as técnicas
39_feature_selection_comparison.png     # Comparação visual de métodos
40_feature_selection_heatmap.png        # Heatmap de rankings
41_cv_feature_selection.csv             # Performance com cross-validation
42_cv_performance_vs_features.png       # Curva performance vs número de features
```

---

### 7. Predictive Modeling → `predictive_models.py`

Treino e avaliação de modelos de classificação (Normal vs Anormal).

**Modelos implementados:**
- Decision Tree
- Support Vector Machine (SVM)
- Naive Bayes
- Random Forest
- Neural Network (MLP)

**Outputs gerados:**
```
43_model_comparison.csv                 # Comparação de métricas de todos modelos
44_confusion_matrices.png               # Matrizes de confusão (5 modelos)
45_roc_curves.png                       # Curvas ROC com AUC
46_metrics_comparison.png               # Comparação visual de métricas
47_feature_importance_models.png        # Feature importance (DT e RF)
48_decision_tree_visualization.png      # Visualização completa da árvore
49_classification_reports.csv           # Reports detalhados por classe
50_cross_validation_results.csv         # Resultados de validação cruzada (5-fold)
```

**Total de arquivos gerados:** 50+ (CSVs + PNGs)

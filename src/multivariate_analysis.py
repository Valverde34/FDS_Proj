import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configuracoes de visualizacao
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

# Carregar dados limpos
df = pd.read_excel(r'C:\Uni\1_ano\1_semestre\CD\FDS_Proj\src\UCMF_CLEAN.xlsx')

print("="*80)
print("ANÁLISE MULTIVARIADA")
print("="*80)

# Definir variaveis numericas
numeric_vars = ['IDADE', 'Peso', 'Altura', 'IMC', 'PA SISTOLICA', 'PA DIASTOLICA', 'FC']

# Preparar dados (remover NaN)
df_clean = df[numeric_vars + ['NORMAL X ANORMAL']].dropna()
X = df_clean[numeric_vars].values
y = df_clean['NORMAL X ANORMAL'].values

print(f"\nRegistos válidos: {len(df_clean)}")
print(f"Variáveis numéricas: {len(numeric_vars)}")

# ============================================================================
# 1. REGRESSÃO LOGÍSTICA (TARGET BINÁRIO)
# ============================================================================
print("\n" + "="*80)
print("1. REGRESSÃO LOGÍSTICA MULTIVARIADA")
print("="*80)
print("NOTA: Como o TARGET é DICOTÔMICO (Normal/Anormal), usamos Regressão Logística")
print("      (não Linear, pois esta assume variável contínua)")

# Codificar target (Normal=0, Anormal=1)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Dividir dados (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Normalizar dados (importante para regressão logística)
scaler_logistic = StandardScaler()
X_train_scaled = scaler_logistic.fit_transform(X_train)
X_test_scaled = scaler_logistic.transform(X_test)

# Treinar modelo de regressão logística
model_logistic = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
model_logistic.fit(X_train_scaled, y_train)

# Avaliar modelo
accuracy_train = model_logistic.score(X_train_scaled, y_train)
accuracy_test = model_logistic.score(X_test_scaled, y_test)

print(f"\nRegressão Logística (predição de TARGET):")
print(f"Accuracy (treino): {accuracy_train:.4f}")
print(f"Accuracy (teste): {accuracy_test:.4f}")

# Coeficientes e Odds Ratios
coeficientes_log = model_logistic.coef_[0]
odds_ratios = np.exp(coeficientes_log)

coeficientes_df = pd.DataFrame({
    'Variável': numeric_vars,
    'Coeficiente': coeficientes_log,
    'Odds_Ratio': odds_ratios,
    'Interpretação': ['Aumenta risco' if or_ > 1 else 'Diminui risco' for or_ in odds_ratios],
    'Abs_Coef': np.abs(coeficientes_log)
})
coeficientes_df = coeficientes_df.sort_values('Abs_Coef', ascending=False)

print(f"\nCoeficientes da Regressão Logística:")
print(coeficientes_df[['Variável', 'Coeficiente', 'Odds_Ratio', 'Interpretação']].round(4))

print(f"\n📊 INTERPRETAÇÃO DOS ODDS RATIOS:")
for idx, row in coeficientes_df.iterrows():
    var = row['Variável']
    or_val = row['Odds_Ratio']
    if or_val > 1:
        percentual = (or_val - 1) * 100
        print(f"   • {var}: OR={or_val:.3f} → Aumenta em {percentual:.1f}% a chance de Anormal")
    else:
        percentual = (1 - or_val) * 100
        print(f"   • {var}: OR={or_val:.3f} → Diminui em {percentual:.1f}% a chance de Anormal")

coeficientes_df.to_csv('18_logistic_regression_coefficients.csv', index=False)

# Grafico dos coeficientes
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Gráfico 1: Coeficientes
colors = ['green' if c > 0 else 'red' for c in coeficientes_df['Coeficiente']]
axes[0].barh(coeficientes_df['Variável'], coeficientes_df['Coeficiente'], color=colors, alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Coeficiente (Log-Odds)', fontsize=12, fontweight='bold')
axes[0].set_title('Coeficientes da Regressão Logística\n(Predição: Normal vs Anormal)', 
             fontsize=14, fontweight='bold')
axes[0].axvline(0, color='black', linewidth=1.5)
axes[0].grid(True, alpha=0.3, axis='x')

# Gráfico 2: Odds Ratios
colors_or = ['green' if or_ > 1 else 'red' for or_ in coeficientes_df['Odds_Ratio']]
axes[1].barh(coeficientes_df['Variável'], coeficientes_df['Odds_Ratio'], color=colors_or, alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Odds Ratio', fontsize=12, fontweight='bold')
axes[1].set_title('Odds Ratios (exp(coeficiente))\nOR>1: Aumenta risco | OR<1: Diminui risco', 
             fontsize=14, fontweight='bold')
axes[1].axvline(1, color='black', linewidth=2, linestyle='--', label='OR=1 (sem efeito)')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('19_logistic_regression_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Gráfico salvo: 19_logistic_regression_plot.png")

# ============================================================================
# 2. MUTUAL INFORMATION
# ============================================================================
print("\n" + "="*80)
print("2. MUTUAL INFORMATION")
print("="*80)

# Calcular mutual information (importancia das variaveis para target)
mi_scores = mutual_info_classif(X, y_encoded, random_state=42)

# Criar DataFrame
mi_df = pd.DataFrame({
    'Variável': numeric_vars,
    'MI_Score': mi_scores
})
mi_df = mi_df.sort_values('MI_Score', ascending=False)

print("\nMutual Information Scores:")
print(mi_df.round(4))
mi_df.to_csv('20_mutual_information.csv', index=False)

# Grafico
fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(mi_df['Variável'], mi_df['MI_Score'], color='teal', alpha=0.7, edgecolor='black')
ax.set_xlabel('Mutual Information Score', fontsize=12, fontweight='bold')
ax.set_title('Mutual Information: Importância das Variáveis\n(maior score = mais informativa)', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('21_mutual_information_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Gráfico salvo: 21_mutual_information_plot.png")

# ============================================================================
# 3. PCA (Principal Component Analysis)
# ============================================================================
print("\n" + "="*80)
print("3. PCA (Principal Component Analysis)")
print("="*80)

# Normalizar dados (importante para PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Variancia explicada
var_explicada = pca.explained_variance_ratio_
var_cumulativa = np.cumsum(var_explicada)

print("\nVariância explicada por componente:")
for i, (var, cum) in enumerate(zip(var_explicada, var_cumulativa)):
    print(f"PC{i+1}: {var*100:.2f}% (cumulativa: {cum*100:.2f}%)")

# Salvar resultados
pca_results = pd.DataFrame({
    'Componente': [f'PC{i+1}' for i in range(len(var_explicada))],
    'Variância_Explicada_%': var_explicada * 100,
    'Variância_Cumulativa_%': var_cumulativa * 100
})
pca_results.to_csv('22_pca_variance.csv', index=False)

# Grafico de variancia explicada
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Scree plot
axes[0].bar(range(1, len(var_explicada)+1), var_explicada*100, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].plot(range(1, len(var_explicada)+1), var_explicada*100, 'o-', color='red', linewidth=2, markersize=8)
axes[0].set_xlabel('Componente Principal', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Variância Explicada (%)', fontsize=12, fontweight='bold')
axes[0].set_title('Scree Plot - PCA', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

# Variancia cumulativa
axes[1].plot(range(1, len(var_cumulativa)+1), var_cumulativa*100, 'o-', color='green', linewidth=2, markersize=8)
axes[1].axhline(80, color='red', linestyle='--', linewidth=2, label='80% da variância')
axes[1].axhline(90, color='orange', linestyle='--', linewidth=2, label='90% da variância')
axes[1].set_xlabel('Número de Componentes', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Variância Cumulativa (%)', fontsize=12, fontweight='bold')
axes[1].set_title('Variância Cumulativa - PCA', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('23_pca_variance_plots.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Gráficos salvos: 23_pca_variance_plots.png")

# Loadings (contribuicao de cada variavel original nas PCs)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loadings_df = pd.DataFrame(
    loadings[:, :3],  # Primeiras 3 PCs
    columns=['PC1', 'PC2', 'PC3'],
    index=numeric_vars
)
print("\nLoadings (contribuição das variáveis nas PCs):")
print(loadings_df.round(3))
loadings_df.to_csv('24_pca_loadings.csv')

# Biplot (PC1 vs PC2)
fig, ax = plt.subplots(figsize=(12, 10))

# Pontos (observacoes)
colors_map = {'Normal': '#2ecc71', 'Anormal': '#e74c3c'}
for label in np.unique(y):
    mask = y == label
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
              c=colors_map[label], label=label, alpha=0.6, s=50)

# Vetores (variaveis)
for i, var in enumerate(numeric_vars):
    ax.arrow(0, 0, loadings[i, 0]*3, loadings[i, 1]*3,
            head_width=0.3, head_length=0.3, fc='black', ec='black', linewidth=2)
    ax.text(loadings[i, 0]*3.3, loadings[i, 1]*3.3, var,
           fontsize=11, fontweight='bold', ha='center', va='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

ax.set_xlabel(f'PC1 ({var_explicada[0]*100:.1f}%)', fontsize=12, fontweight='bold')
ax.set_ylabel(f'PC2 ({var_explicada[1]*100:.1f}%)', fontsize=12, fontweight='bold')
ax.set_title('PCA Biplot: PC1 vs PC2\n(Normal vs Anormal)', fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='gray', linewidth=1)
ax.axvline(0, color='gray', linewidth=1)
plt.tight_layout()
plt.savefig('25_pca_biplot.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Biplot salvo: 25_pca_biplot.png")

# ============================================================================
# 4. CLUSTER ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("4. CLUSTER ANALYSIS")
print("="*80)

# Testar diferentes numeros de clusters (K-Means)
inertias = []
silhouette_scores = []
db_scores = []
k_range = range(2, 8)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))
    db_scores.append(davies_bouldin_score(X_scaled, labels))

# Determinar numero otimo de clusters
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\nNúmero ótimo de clusters (Silhouette): {optimal_k}")

# Criar DataFrame com metricas
cluster_metrics = pd.DataFrame({
    'K': list(k_range),
    'Inertia': inertias,
    'Silhouette_Score': silhouette_scores,
    'Davies_Bouldin': db_scores
})
print("\nMétricas de Clustering:")
print(cluster_metrics.round(4))
cluster_metrics.to_csv('26_cluster_metrics.csv', index=False)

# Graficos de avaliacao
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Elbow plot
axes[0].plot(k_range, inertias, 'o-', color='blue', linewidth=2, markersize=8)
axes[0].set_xlabel('Número de Clusters (k)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=11, fontweight='bold')
axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Silhouette score
axes[1].plot(k_range, silhouette_scores, 'o-', color='green', linewidth=2, markersize=8)
axes[1].axvline(optimal_k, color='red', linestyle='--', linewidth=2, label=f'Ótimo: k={optimal_k}')
axes[1].set_xlabel('Número de Clusters (k)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Silhouette Score (maior = melhor)', fontsize=11, fontweight='bold')
axes[1].set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Davies-Bouldin
axes[2].plot(k_range, db_scores, 'o-', color='orange', linewidth=2, markersize=8)
axes[2].set_xlabel('Número de Clusters (k)', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Davies-Bouldin Index (menor = melhor)', fontsize=11, fontweight='bold')
axes[2].set_title('Davies-Bouldin Analysis', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('27_cluster_evaluation.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Gráficos salvos: 27_cluster_evaluation.png")

# Aplicar K-Means com numero otimo
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_scaled)

# Adicionar clusters ao dataframe
df_clean['Cluster'] = clusters

# Analise dos clusters
print(f"\nDistribuição dos Clusters:")
print(df_clean['Cluster'].value_counts().sort_index())

# Cross-tab: Cluster vs TARGET
crosstab = pd.crosstab(df_clean['Cluster'], df_clean['NORMAL X ANORMAL'], normalize='index')
print("\nDistribuição TARGET por Cluster (%):")
print(crosstab.round(3) * 100)
crosstab.to_csv('28_cluster_vs_target.csv')

# Visualizacao dos clusters (PCA space)
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Clusters no espaço PCA
for cluster in range(optimal_k):
    mask = clusters == cluster
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   label=f'Cluster {cluster}', alpha=0.6, s=50)
axes[0].set_xlabel(f'PC1 ({var_explicada[0]*100:.1f}%)', fontsize=12, fontweight='bold')
axes[0].set_ylabel(f'PC2 ({var_explicada[1]*100:.1f}%)', fontsize=12, fontweight='bold')
axes[0].set_title(f'K-Means Clustering (k={optimal_k})\nno espaço PCA', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Target no espaço PCA (comparacao)
for label in np.unique(y):
    mask = y == label
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=colors_map[label], label=label, alpha=0.6, s=50)
axes[1].set_xlabel(f'PC1 ({var_explicada[0]*100:.1f}%)', fontsize=12, fontweight='bold')
axes[1].set_ylabel(f'PC2 ({var_explicada[1]*100:.1f}%)', fontsize=12, fontweight='bold')
axes[1].set_title('TARGET Real\nno espaço PCA', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('29_clusters_vs_target_pca.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Visualização salva: 29_clusters_vs_target_pca.png")

# Perfil dos clusters (medias)
cluster_profile = df_clean.groupby('Cluster')[numeric_vars].mean()
print("\nPerfil dos Clusters (médias):")
print(cluster_profile.round(2))
cluster_profile.to_csv('30_cluster_profiles.csv')

# Heatmap dos perfis
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(cluster_profile.T, annot=True, fmt='.1f', cmap='YlOrRd', 
            cbar_kws={"shrink": 0.8}, linewidths=1, ax=ax)
ax.set_title(f'Perfil dos Clusters (k={optimal_k})\nMédias das Variáveis', 
            fontsize=14, fontweight='bold')
ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
ax.set_ylabel('Variável', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('31_cluster_profiles_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Heatmap salvo: 31_cluster_profiles_heatmap.png")

# ============================================================================
# RESUMO FINAL
# ============================================================================
print("\n" + "="*80)
print("RESUMO DA ANÁLISE MULTIVARIADA")
print("="*80)

print("\nArquivos gerados:")
print("  18_logistic_regression_coefficients.csv")
print("  19_logistic_regression_plot.png")
print("  20_mutual_information.csv")
print("  21_mutual_information_plot.png")
print("  22_pca_variance.csv")
print("  23_pca_variance_plots.png")
print("  24_pca_loadings.csv")
print("  25_pca_biplot.png")
print("  26_cluster_metrics.csv")
print("  27_cluster_evaluation.png")
print("  28_cluster_vs_target.csv")
print("  29_clusters_vs_target_pca.png")
print("  30_cluster_profiles.csv")
print("  31_cluster_profiles_heatmap.png")

print("\n" + "="*80)
print("PRINCIPAIS CONCLUSÕES:")
print("="*80)
print(f"\n1. Regressão Logística: Accuracy = {accuracy_test:.3f}")
print(f"   Variável mais importante: {coeficientes_df.iloc[0]['Variável']} (OR={coeficientes_df.iloc[0]['Odds_Ratio']:.3f})")

print(f"\n2. Mutual Information:")
print(f"   Variável mais informativa: {mi_df.iloc[0]['Variável']} (MI={mi_df.iloc[0]['MI_Score']:.3f})")

print(f"\n3. PCA:")
print(f"   PC1+PC2 explicam {var_cumulativa[1]*100:.1f}% da variância")
print(f"   {np.sum(var_cumulativa >= 0.8)} componentes explicam 80% da variância")

print(f"\n4. Clustering:")
print(f"   Número ótimo de clusters: {optimal_k}")
print(f"   Silhouette Score: {max(silhouette_scores):.3f}")

print("\n✓ Análise Multivariada concluída com sucesso!")

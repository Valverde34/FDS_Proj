import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import (
    SelectKBest, f_classif, chi2, mutual_info_classif,
    RFE, SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Configuracoes
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Carregar dados
df = pd.read_excel(r'./UCMF_CLEAN.xlsx')

print("\n=== FEATURE SELECTION ===")

# ============================================================================
# ANÁLISE DE MULTICOLINEARIDADE (Peso, Altura, IMC)
# ============================================================================
print("\n[1/10] Detectar multicolinearidade...")

all_vars = ['IDADE', 'Peso', 'Altura', 'IMC', 'PA SISTOLICA', 'PA DIASTOLICA', 'FC']
df_all = df[all_vars].dropna()

# Correlação entre Peso, Altura e IMC
corr_matrix = df_all[['Peso', 'Altura', 'IMC']].corr()

print(f"   Peso <-> IMC: r = {corr_matrix.loc['Peso', 'IMC']:.3f}")
print(f"   Altura <-> IMC: r = {corr_matrix.loc['Altura', 'IMC']:.3f}")
print("   Decisao: Remover Peso e Altura (manter IMC)")

# Salvar análise de multicolinearidade
multicol_df = pd.DataFrame({
    'Variável_1': ['Peso', 'Altura', 'Peso'],
    'Variável_2': ['IMC', 'IMC', 'Altura'],
    'Correlação': [corr_matrix.loc['Peso', 'IMC'], 
                   corr_matrix.loc['Altura', 'IMC'],
                   corr_matrix.loc['Peso', 'Altura']],
    'Decisão': ['Remover Peso (manter IMC)', 'Remover Altura (manter IMC)', 'Confirma redundância']
})
multicol_df.to_csv('31_multicolinearidade_analysis.csv', index=False)

# Preparar dados (SEM Peso e Altura)
print("\n[2/10] Preparar dados sem variáveis redundantes...")

numeric_vars = ['IDADE', 'IMC', 'PA SISTOLICA', 'PA DIASTOLICA', 'FC']
df_clean = df[numeric_vars + ['NORMAL X ANORMAL']].dropna()

X = df_clean[numeric_vars].values
y = df_clean['NORMAL X ANORMAL'].values

# Codificar target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"   Registos: {len(df_clean)} | Variáveis: {len(numeric_vars)}")

# ============================================================================
# 1. VARIANCE THRESHOLD (remover features com baixa variância)
# ============================================================================
print("\n[3/10] Variance Threshold...")

# Calcular variancias
variances = np.var(X, axis=0)
var_df = pd.DataFrame({
    'Variável': numeric_vars,
    'Variância': variances
})
var_df = var_df.sort_values('Variância', ascending=False)
var_df.to_csv('32_feature_variances.csv', index=False)

# ============================================================================
# 2. UNIVARIATE FEATURE SELECTION (F-test)
# ============================================================================
print("[4/10] F-test ANOVA...")

# F-test
selector_f = SelectKBest(score_func=f_classif, k='all')
selector_f.fit(X, y_encoded)

f_scores = selector_f.scores_
f_pvalues = selector_f.pvalues_

f_df = pd.DataFrame({
    'Variável': numeric_vars,
    'F-Score': f_scores,
    'p-value': f_pvalues,
    'Significativo': ['Sim' if p < 0.05 else 'Não' for p in f_pvalues]
})
f_df = f_df.sort_values('F-Score', ascending=False)
f_df.to_csv('33_univariate_ftest.csv', index=False)

# ============================================================================
# 3. MUTUAL INFORMATION
# ============================================================================
print("[5/10] Mutual Information...")

mi_scores = mutual_info_classif(X, y_encoded, random_state=42)

mi_df = pd.DataFrame({
    'Variável': numeric_vars,
    'MI_Score': mi_scores
})
mi_df = mi_df.sort_values('MI_Score', ascending=False)
mi_df.to_csv('34_mutual_information_scores.csv', index=False)

# ============================================================================
# 4. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================================
print("[6/10] Random Forest Importance...")

rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
rf.fit(X, y_encoded)

rf_importance = rf.feature_importances_
rf_df = pd.DataFrame({
    'Variável': numeric_vars,
    'Importance': rf_importance
})
rf_df = rf_df.sort_values('Importance', ascending=False)
rf_df.to_csv('35_random_forest_importance.csv', index=False)

# ============================================================================
# 5. GRADIENT BOOSTING FEATURE IMPORTANCE
# ============================================================================
print("[7/10] Gradient Boosting Importance...")

gb = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=3)
gb.fit(X, y_encoded)

gb_importance = gb.feature_importances_
gb_df = pd.DataFrame({
    'Variável': numeric_vars,
    'Importance': gb_importance
})
gb_df = gb_df.sort_values('Importance', ascending=False)
gb_df.to_csv('36_gradient_boosting_importance.csv', index=False)

# ============================================================================
# 6. RECURSIVE FEATURE ELIMINATION (RFE)
# ============================================================================
print("[8/10] Recursive Feature Elimination...")

# Normalizar para RFE
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# RFE com Logistic Regression
estimator = LogisticRegression(random_state=42, max_iter=1000)
rfe = RFE(estimator, n_features_to_select=5)
rfe.fit(X_scaled, y_encoded)

rfe_ranking = rfe.ranking_
rfe_df = pd.DataFrame({
    'Variável': numeric_vars,
    'Ranking': rfe_ranking,
    'Selecionado': rfe.support_
})
rfe_df = rfe_df.sort_values('Ranking')
rfe_df.to_csv('37_rfe_rankings.csv', index=False)

selected_features = rfe_df[rfe_df['Selecionado']]['Variável'].tolist()

# ============================================================================
# 7. CONSOLIDATED FEATURE RANKING
# ============================================================================
print("[9/10] Ranking consolidado...")

# Normalizar todos os scores para [0, 1]
def normalize_scores(scores):
    min_val = np.min(scores)
    max_val = np.max(scores)
    if max_val - min_val == 0:
        return np.ones_like(scores)
    return (scores - min_val) / (max_val - min_val)

# Criar ranking consolidado
consolidated = pd.DataFrame({
    'Variável': numeric_vars,
    'F-Score_norm': normalize_scores(f_scores),
    'MI_norm': normalize_scores(mi_scores),
    'RF_Importance_norm': normalize_scores(rf_importance),
    'GB_Importance_norm': normalize_scores(gb_importance),
    'RFE_Score': 1.0 / rfe_ranking  # Inverter ranking (1 = melhor)
})

# Calcular score medio
consolidated['Score_Médio'] = consolidated[['F-Score_norm', 'MI_norm', 'RF_Importance_norm', 
                                            'GB_Importance_norm', 'RFE_Score']].mean(axis=1)

# Ordenar por score medio
consolidated = consolidated.sort_values('Score_Médio', ascending=False)
consolidated.to_csv('38_consolidated_feature_ranking.csv', index=False)

# Top features (score > 0.6)
top_features = consolidated[consolidated['Score_Médio'] > 0.6]['Variável'].tolist()

# ============================================================================
# 8. VISUALIZAÇÕES
# ============================================================================
print("[10/10] Gerar visualizações...")

# Grafico 1: Comparacao de todos os metodos
fig, axes = plt.subplots(2, 3, figsize=(22, 12))

# F-test
axes[0, 0].barh(f_df['Variável'], f_df['F-Score'], color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('F-Score', fontweight='bold')
axes[0, 0].set_title('F-test (ANOVA)', fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='x')

# Mutual Information
axes[0, 1].barh(mi_df['Variável'], mi_df['MI_Score'], color='teal', alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('MI Score', fontweight='bold')
axes[0, 1].set_title('Mutual Information', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='x')

# Random Forest
axes[0, 2].barh(rf_df['Variável'], rf_df['Importance'], color='green', alpha=0.7, edgecolor='black')
axes[0, 2].set_xlabel('Importance', fontweight='bold')
axes[0, 2].set_title('Random Forest', fontsize=13, fontweight='bold')
axes[0, 2].grid(True, alpha=0.3, axis='x')

# Gradient Boosting
axes[1, 0].barh(gb_df['Variável'], gb_df['Importance'], color='orange', alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Importance', fontweight='bold')
axes[1, 0].set_title('Gradient Boosting', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='x')

# RFE
colors_rfe = ['green' if s else 'red' for s in rfe_df['Selecionado']]
axes[1, 1].barh(rfe_df['Variável'], 1.0/rfe_df['Ranking'], color=colors_rfe, alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('RFE Score (1/Ranking)', fontweight='bold')
axes[1, 1].set_title('RFE (verde=selecionado)', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='x')

# Consolidated
colors_cons = ['darkgreen' if v in top_features else 'lightblue' for v in consolidated['Variável']]
axes[1, 2].barh(consolidated['Variável'], consolidated['Score_Médio'], color=colors_cons, alpha=0.7, edgecolor='black')
axes[1, 2].set_xlabel('Score Médio (normalizado)', fontweight='bold')
axes[1, 2].set_title('RANKING CONSOLIDADO\n(verde=top features)', fontsize=13, fontweight='bold')
axes[1, 2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('39_feature_selection_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Grafico 2: Heatmap de rankings normalizados
fig, ax = plt.subplots(figsize=(12, 8))

heatmap_data = consolidated[['F-Score_norm', 'MI_norm', 'RF_Importance_norm', 
                             'GB_Importance_norm', 'RFE_Score']].T
heatmap_data.columns = consolidated['Variável']

sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd', 
            cbar_kws={"shrink": 0.8}, linewidths=1, ax=ax)
ax.set_title('Heatmap: Scores Normalizados por Método', fontsize=14, fontweight='bold')
ax.set_ylabel('Método', fontsize=12, fontweight='bold')
ax.set_xlabel('Variável', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('40_feature_selection_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 9. VALIDAÇÃO COM CROSS-VALIDATION
# ============================================================================

# Testar diferentes numeros de features
cv_results = []

for k in range(1, len(numeric_vars) + 1):
    # Selecionar top k features
    top_k_features = consolidated.head(k)['Variável'].tolist()
    top_k_indices = [numeric_vars.index(f) for f in top_k_features]
    X_subset = X[:, top_k_indices]
    
    # Cross-validation com Random Forest
    rf_cv = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    scores = cross_val_score(rf_cv, X_subset, y_encoded, cv=5, scoring='accuracy')
    
    cv_results.append({
        'N_Features': k,
        'Features': ', '.join(top_k_features),
        'CV_Accuracy_Mean': scores.mean(),
        'CV_Accuracy_Std': scores.std()
    })

cv_df = pd.DataFrame(cv_results)
cv_df.to_csv('41_cv_feature_selection.csv', index=False)

# Grafico de CV accuracy
fig, ax = plt.subplots(figsize=(12, 6))
ax.errorbar(cv_df['N_Features'], cv_df['CV_Accuracy_Mean'], 
           yerr=cv_df['CV_Accuracy_Std'], 
           marker='o', markersize=8, capsize=5, linewidth=2, color='blue')
ax.set_xlabel('Número de Features', fontsize=12, fontweight='bold')
ax.set_ylabel('Cross-Validation Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Desempenho vs Número de Features\n(Random Forest, 5-fold CV)', 
            fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xticks(range(1, len(numeric_vars) + 1))

# Marcar o melhor
best_idx = cv_df['CV_Accuracy_Mean'].idxmax()
best_k = cv_df.loc[best_idx, 'N_Features']
best_acc = cv_df.loc[best_idx, 'CV_Accuracy_Mean']
ax.axvline(best_k, color='red', linestyle='--', linewidth=2, 
          label=f'Ótimo: {int(best_k)} features (Acc={best_acc:.3f})')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('42_cv_performance_vs_features.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# RESUMO FINAL
# ============================================================================
print("\nFeature Selection concluída")

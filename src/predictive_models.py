import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# Configuracoes
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

print('\n=== MODELAÇÃO PREDITIVA & AVALIAÇÃO ===')

# Carregar dados
df = pd.read_excel(r'./UCMF_CLEAN.xlsx')

print('\n[1/8] Preparar dados e features...')

# Preparar dados (SEM Peso e Altura - apenas IMC)
numeric_vars = ['IDADE', 'IMC', 'PA SISTOLICA', 'PA DIASTOLICA', 'FC']
df_clean = df[numeric_vars + ['NORMAL X ANORMAL']].dropna()

X = df_clean[numeric_vars].values
y = df_clean['NORMAL X ANORMAL'].values

# Codificar target
le = LabelEncoder()
y_encoded = le.fit_transform(y)



# Dividir dados (70% treino, 30% teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)



# Normalizar dados (importante para SVM e Neural Networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {}
predictions = {}
predictions_proba = {}

print('\n[2/8] Treinar Decision Tree...')
dt = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=20, min_samples_leaf=10)
dt.fit(X_train, y_train)
models['Decision Tree'] = dt
predictions['Decision Tree'] = dt.predict(X_test)
predictions_proba['Decision Tree'] = dt.predict_proba(X_test)[:, 1]

print('[3/8] Treinar SVM...')
svm = SVC(kernel='rbf', probability=True, random_state=42, C=1.0, gamma='scale')
svm.fit(X_train_scaled, y_train)
models['SVM'] = svm
predictions['SVM'] = svm.predict(X_test_scaled)
predictions_proba['SVM'] = svm.predict_proba(X_test_scaled)[:, 1]

print('[4/8] Treinar Naive Bayes...')
nb = GaussianNB()
nb.fit(X_train, y_train)
models['Naive Bayes'] = nb
predictions['Naive Bayes'] = nb.predict(X_test)
predictions_proba['Naive Bayes'] = nb.predict_proba(X_test)[:, 1]

print('[5/8] Treinar Random Forest...')
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5, min_samples_split=20)
rf.fit(X_train, y_train)
models['Random Forest'] = rf
predictions['Random Forest'] = rf.predict(X_test)
predictions_proba['Random Forest'] = rf.predict_proba(X_test)[:, 1]

print('[6/8] Treinar Neural Network...')
nn = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42, early_stopping=True)
nn.fit(X_train_scaled, y_train)
models['Neural Network'] = nn
predictions['Neural Network'] = nn.predict(X_test_scaled)
predictions_proba['Neural Network'] = nn.predict_proba(X_test_scaled)[:, 1]

print('\n[7/8] Avaliar modelos...')

results = []

for model_name in models.keys():
    y_pred = predictions[model_name]
    
    # Calcular metricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # AUC-ROC
    try:
        auc_score = roc_auc_score(y_test, predictions_proba[model_name])
    except:
        auc_score = 0.0
    
    results.append({
        'Modelo': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'AUC-ROC': auc_score
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('F1-Score', ascending=False)
results_df.to_csv('43_model_comparison.csv', index=False)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.ravel()

for idx, model_name in enumerate(models.keys()):
    y_pred = predictions[model_name]
    cm = confusion_matrix(y_test, y_pred)
    
    # Plotar
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[idx],
                xticklabels=le.classes_, yticklabels=le.classes_)
    axes[idx].set_title(f'{model_name}\nAccuracy: {accuracy_score(y_test, y_pred):.3f}', 
                       fontsize=13, fontweight='bold')
    axes[idx].set_xlabel('Predição', fontweight='bold')
    axes[idx].set_ylabel('Real', fontweight='bold')

# Remover eixo vazio
fig.delaxes(axes[5])

plt.suptitle('Matrizes de Confusão - Todos os Modelos', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('44_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(12, 10))

colors = ['blue', 'red', 'green', 'orange', 'purple']

for idx, model_name in enumerate(models.keys()):
    y_proba = predictions_proba[model_name]
    
    # Calcular ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plotar
    ax.plot(fpr, tpr, color=colors[idx], linewidth=2.5, 
           label=f'{model_name} (AUC = {roc_auc:.3f})')

# Linha diagonal (classificador aleatorio)
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Aleatório (AUC = 0.500)')

ax.set_xlabel('False Positive Rate (1 - Especificidade)', fontsize=13, fontweight='bold')
ax.set_ylabel('True Positive Rate (Sensibilidade)', fontsize=13, fontweight='bold')
ax.set_title('Curvas ROC - Comparação de Modelos', fontsize=15, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('45_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# COMPARAÇÃO DE MÉTRICAS
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors_bar = ['steelblue', 'coral', 'mediumseagreen', 'gold']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    
    data = results_df.sort_values(metric, ascending=True)
    ax.barh(data['Modelo'], data[metric], color=colors_bar[idx], alpha=0.8, edgecolor='black')
    ax.set_xlabel(metric, fontsize=12, fontweight='bold')
    ax.set_title(f'Comparação: {metric}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim([0, 1])
    
    # Adicionar valores
    for i, v in enumerate(data[metric]):
        ax.text(v + 0.02, i, f'{v:.3f}', va='center', fontweight='bold')

plt.suptitle('Comparação de Métricas - Todos os Modelos', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('46_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Decision Tree
dt_importance = models['Decision Tree'].feature_importances_
dt_imp_df = pd.DataFrame({'Feature': numeric_vars, 'Importance': dt_importance})
dt_imp_df = dt_imp_df.sort_values('Importance', ascending=True)

axes[0].barh(dt_imp_df['Feature'], dt_imp_df['Importance'], color='skyblue', alpha=0.8, edgecolor='black')
axes[0].set_xlabel('Importance', fontsize=12, fontweight='bold')
axes[0].set_title('Feature Importance - Decision Tree', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')

# Random Forest
rf_importance = models['Random Forest'].feature_importances_
rf_imp_df = pd.DataFrame({'Feature': numeric_vars, 'Importance': rf_importance})
rf_imp_df = rf_imp_df.sort_values('Importance', ascending=True)

axes[1].barh(rf_imp_df['Feature'], rf_imp_df['Importance'], color='lightgreen', alpha=0.8, edgecolor='black')
axes[1].set_xlabel('Importance', fontsize=12, fontweight='bold')
axes[1].set_title('Feature Importance - Random Forest', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('47_feature_importance_models.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# VISUALIZAR ÁRVORE DE DECISÃO
# ============================================================================
fig, ax = plt.subplots(figsize=(20, 12))
plot_tree(models['Decision Tree'], filled=True, feature_names=numeric_vars, 
         class_names=le.classes_, rounded=True, fontsize=10, ax=ax)
ax.set_title('Árvore de Decisão Completa', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('48_decision_tree_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

reports = []

for model_name in models.keys():
    y_pred = predictions[model_name]
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    for label in le.classes_:
        reports.append({
            'Modelo': model_name,
            'Classe': label,
            'Precision': report[label]['precision'],
            'Recall': report[label]['recall'],
            'F1-Score': report[label]['f1-score'],
            'Support': report[label]['support']
        })

reports_df = pd.DataFrame(reports)
reports_df.to_csv('49_classification_reports.csv', index=False)

cv_results = []

for model_name, model in models.items():
    # Selecionar dados corretos (scaled vs nao-scaled)
    if model_name in ['SVM', 'Neural Network']:
        X_cv = X_train_scaled
    else:
        X_cv = X_train
    
    scores = cross_val_score(model, X_cv, y_train, cv=5, scoring='f1_weighted')
    
    cv_results.append({
        'Modelo': model_name,
        'CV_F1_Mean': scores.mean(),
        'CV_F1_Std': scores.std()
    })

cv_df = pd.DataFrame(cv_results)
cv_df = cv_df.sort_values('CV_F1_Mean', ascending=False)
cv_df.to_csv('50_cross_validation_results.csv', index=False)

print('\n[8/8] Gerar visualizações e relatórios...')
print('\nModelação Preditiva concluída')

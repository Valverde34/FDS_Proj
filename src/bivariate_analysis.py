import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau, ttest_ind, f_oneway, kruskal
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Configuracoes de visualizacao
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

# Carregar dados limpos
df = pd.read_excel(r'./UCMF_CLEAN.xlsx')

print("\n=== ANÁLISE BIVARIADA ===")
print(f"Registos carregados: {len(df)}")

# Definir variaveis numericas
numeric_vars = ['IDADE', 'Peso', 'Altura', 'IMC', 'PA SISTOLICA', 'PA DIASTOLICA', 'FC']

# Remover valores em falta para analise de correlacao
df_numeric = df[numeric_vars].dropna()
print(f"Registos válidos para correlação: {len(df_numeric)}")

# ============================================================================
# 1. CORRELAÇÕES (Pearson, Spearman, Kendall)
# ============================================================================
print("\n[1/11] Calcular correlações...")
# Pearson - mede relacao linear
pearson_corr = df_numeric.corr(method='pearson')
pearson_corr.to_csv('08_correlation_pearson.csv')

# Spearman - mede relacao monotonica (nao precisa ser linear)
spearman_corr = df_numeric.corr(method='spearman')
spearman_corr.to_csv('09_correlation_spearman.csv')

# Kendall - outra medida de correlacao de ranking
kendall_corr = df_numeric.corr(method='kendall')
kendall_corr.to_csv('10_correlation_kendall.csv')

# ============================================================================
# 2. HEATMAPS DAS CORRELAÇÕES
# ============================================================================
print("[2/11] Gerar heatmaps de correlação...")
fig, axes = plt.subplots(1, 3, figsize=(24, 7))

# Heatmap Pearson
sns.heatmap(pearson_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=axes[0],
            vmin=-1, vmax=1)
axes[0].set_title('Correlação de Pearson (Linear)', fontsize=14, fontweight='bold')

# Heatmap Spearman
sns.heatmap(spearman_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=axes[1],
            vmin=-1, vmax=1)
axes[1].set_title('Correlação de Spearman (Monotônica)', fontsize=14, fontweight='bold')

# Heatmap Kendall
sns.heatmap(kendall_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=axes[2],
            vmin=-1, vmax=1)
axes[2].set_title('Correlação de Kendall (Rank)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('11_correlation_heatmaps.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 3. ANÁLISE DE CORRELAÇÕES SIGNIFICATIVAS
# ============================================================================
print("[3/11] Identificar correlações significativas...")
# Encontrar correlacoes fortes (Pearson)
correlacoes_fortes = []
for i in range(len(pearson_corr.columns)):
    for j in range(i+1, len(pearson_corr.columns)):
        var1 = pearson_corr.columns[i]
        var2 = pearson_corr.columns[j]
        r = pearson_corr.iloc[i, j]
        
        if abs(r) > 0.3:  # Correlacao moderada ou forte
            correlacoes_fortes.append({
                'Var1': var1,
                'Var2': var2,
                'Pearson': r,
                'Spearman': spearman_corr.iloc[i, j],
                'Kendall': kendall_corr.iloc[i, j]
            })

if correlacoes_fortes:
    df_corr_fortes = pd.DataFrame(correlacoes_fortes)
    df_corr_fortes = df_corr_fortes.sort_values('Pearson', ascending=False, key=abs)
    df_corr_fortes.to_csv('12_strong_correlations.csv', index=False)
    print(f"   Correlações fortes encontradas: {len(correlacoes_fortes)}")

# ============================================================================
# 4. REGRESSÕES SIMPLES
# ============================================================================
print("[4/11] Calcular regressões lineares...")
# Vamos fazer regressoes das variaveis mais correlacionadas
regressoes = []

for corr in correlacoes_fortes[:5]:  # Top 5 correlacoes
    var1 = corr['Var1']
    var2 = corr['Var2']
    
    # Preparar dados (remover NaN)
    data_temp = df[[var1, var2]].dropna()
    X = data_temp[[var1]].values
    y = data_temp[var2].values
    
    if len(X) > 10:  # Minimo de pontos
        # Treinar modelo
        model = LinearRegression()
        model.fit(X, y)
        
        # R² score
        r2 = model.score(X, y)
        
        regressoes.append({
            'X (independente)': var1,
            'Y (dependente)': var2,
            'Coeficiente': model.coef_[0],
            'Intercepto': model.intercept_,
            'R²': r2,
            'N': len(X)
        })

if regressoes:
    df_regressoes = pd.DataFrame(regressoes)
    df_regressoes.to_csv('13_simple_regressions.csv', index=False)

# ============================================================================
# 5. GRÁFICOS DE DISPERSÃO (SCATTERPLOTS)
# ============================================================================
print("[5/11] Gerar scatterplots...")
if correlacoes_fortes:
    n_plots = min(6, len(correlacoes_fortes))
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.ravel()
    
    for idx, corr in enumerate(correlacoes_fortes[:n_plots]):
        var1 = corr['Var1']
        var2 = corr['Var2']
        r = corr['Pearson']
        
        # Preparar dados
        data_temp = df[[var1, var2]].dropna()
        
        # Scatterplot
        axes[idx].scatter(data_temp[var1], data_temp[var2], alpha=0.5, color='steelblue')
        
        # Linha de regressao
        X = data_temp[[var1]].values
        y = data_temp[var2].values
        model = LinearRegression()
        model.fit(X, y)
        axes[idx].plot(X, model.predict(X), color='red', linewidth=2, label='Regressão Linear')
        
        # Configuracoes
        axes[idx].set_xlabel(var1, fontsize=11, fontweight='bold')
        axes[idx].set_ylabel(var2, fontsize=11, fontweight='bold')
        axes[idx].set_title(f'{var1} vs {var2}\nr = {r:.3f}, R² = {model.score(X, y):.3f}',
                           fontsize=12, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('14_scatterplots_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# 6. COMPARAÇÃO NORMAL vs ANORMAL (t-tests)
# ============================================================================
print("[6/11] Realizar t-tests (Normal vs Anormal)...")
# Separar grupos
df_normal = df[df['NORMAL X ANORMAL'] == 'Normal']
df_anormal = df[df['NORMAL X ANORMAL'] == 'Anormal']
print(f"   Normal: {len(df_normal)} | Anormal: {len(df_anormal)}")

# T-tests para cada variavel numerica
ttest_results = []

for var in numeric_vars:
    # Remover NaN
    normal_data = df_normal[var].dropna()
    anormal_data = df_anormal[var].dropna()
    
    if len(normal_data) > 0 and len(anormal_data) > 0:
        # T-test independente
        t_stat, p_value = ttest_ind(normal_data, anormal_data)
        
        # Calcular medias e desvios
        mean_normal = normal_data.mean()
        mean_anormal = anormal_data.mean()
        std_normal = normal_data.std()
        std_anormal = anormal_data.std()
        
        # Diferenca entre medias
        diff = mean_anormal - mean_normal
        
        # Significancia estatistica
        significativo = 'Sim' if p_value < 0.05 else 'Não'
        
        ttest_results.append({
            'Variável': var,
            'Média_Normal': mean_normal,
            'Std_Normal': std_normal,
            'Média_Anormal': mean_anormal,
            'Std_Anormal': std_anormal,
            'Diferença': diff,
            't-statistic': t_stat,
            'p-value': p_value,
            'Significativo (α=0.05)': significativo
        })

# Salvar resultados
df_ttests = pd.DataFrame(ttest_results)
df_ttests = df_ttests.sort_values('p-value')
df_ttests.to_csv('15_ttest_normal_vs_anormal.csv', index=False)

# ============================================================================
# 7. BOXPLOTS COMPARATIVOS (Normal vs Anormal)
# ============================================================================
print("[7/11] Gerar boxplots comparativos...")
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes = axes.ravel()

for i, var in enumerate(numeric_vars):
    if var in df.columns:
        # Preparar dados
        data_plot = df[[var, 'NORMAL X ANORMAL']].dropna()
        
        # Boxplot
        data_plot.boxplot(column=var, by='NORMAL X ANORMAL', ax=axes[i],
                         patch_artist=True)
        
        # Obter o p-value do t-test
        p_val = df_ttests[df_ttests['Variável'] == var]['p-value'].values[0] if var in df_ttests['Variável'].values else 1.0
        sig_text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        
        # Configuracoes
        axes[i].set_title(f'{var}\n(p = {p_val:.4f} {sig_text})', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel(var, fontsize=11)
        plt.sca(axes[i])
        plt.xticks(rotation=0)

# Remover eixo vazio
fig.delaxes(axes[7])

plt.suptitle('Comparação Normal vs Anormal (* p<0.05, ** p<0.01, *** p<0.001)', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('16_boxplots_normal_vs_anormal.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 8. VIOLINPLOTS (alternativa aos boxplots)
# ============================================================================
print("[8/11] Gerar violinplots...")
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes = axes.ravel()

for i, var in enumerate(numeric_vars):
    if var in df.columns:
        # Preparar dados
        data_plot = df[[var, 'NORMAL X ANORMAL']].dropna()
        
        # Violinplot
        sns.violinplot(data=data_plot, x='NORMAL X ANORMAL', y=var, ax=axes[i],
                      palette=['#2ecc71', '#e74c3c'], inner='box')
        
        # Obter o p-value
        p_val = df_ttests[df_ttests['Variável'] == var]['p-value'].values[0] if var in df_ttests['Variável'].values else 1.0
        sig_text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        
        # Configuracoes
        axes[i].set_title(f'{var}\n(p = {p_val:.4f} {sig_text})', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel(var, fontsize=11)
        axes[i].grid(True, alpha=0.3, axis='y')

# Remover eixo vazio
fig.delaxes(axes[7])

plt.suptitle('Distribuições: Normal vs Anormal (* p<0.05, ** p<0.01, *** p<0.001)', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('17_violinplots_normal_vs_anormal.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 9. ANÁLISE NUMÉRICA vs CATEGÓRICA (ANOVA / Kruskal-Wallis)
# ============================================================================
print("[9/11] Executar ANOVA e Kruskal-Wallis...")
categorical_vars = ['SEXO', 'PULSOS', 'SOPRO', 'B2']
anova_results = []

for cat_var in categorical_vars:
    if cat_var in df.columns:
        # Obter categorias válidas (remover NaN)
        df_temp = df[[cat_var] + numeric_vars].dropna(subset=[cat_var])
        categories = df_temp[cat_var].unique()
        
        for num_var in numeric_vars:
            # Preparar grupos
            groups = []
            for cat in categories:
                group_data = df_temp[df_temp[cat_var] == cat][num_var].dropna()
                if len(group_data) > 0:
                    groups.append(group_data)
            
            if len(groups) >= 2:  # Precisa de pelo menos 2 grupos
                # ANOVA (paramétrico - assume normalidade)
                f_stat, p_anova = f_oneway(*groups)
                
                # Kruskal-Wallis (não-paramétrico - não assume normalidade)
                h_stat, p_kruskal = kruskal(*groups)
                
                # Médias por grupo
                means = {cat: df_temp[df_temp[cat_var] == cat][num_var].mean() 
                        for cat in categories}
                
                # Significância
                sig_anova = 'Sim' if p_anova < 0.05 else 'Não'
                sig_kruskal = 'Sim' if p_kruskal < 0.05 else 'Não'
                
                anova_results.append({
                    'Variável_Categórica': cat_var,
                    'Variável_Numérica': num_var,
                    'N_Grupos': len(groups),
                    'F_statistic': f_stat,
                    'p_ANOVA': p_anova,
                    'H_statistic': h_stat,
                    'p_Kruskal': p_kruskal,
                    'Sig_ANOVA': sig_anova,
                    'Sig_Kruskal': sig_kruskal
                })

# Salvar resultados
if anova_results:
    df_anova = pd.DataFrame(anova_results)
    df_anova = df_anova.sort_values('p_ANOVA')
    df_anova.to_csv('17b_anova_categorical_vs_numeric.csv', index=False)

# ============================================================================
# 10. CORRELAÇÃO PARCIAL (Controlando IDADE)
# ============================================================================
print("[10/11] Calcular correlações parciais...")
# Correlação parcial: controlar idade para ver correlações "puras"
df_partial = df[numeric_vars].dropna()

partial_corr_results = []
for i, var1 in enumerate(numeric_vars):
    for var2 in numeric_vars[i+1:]:
        if var1 != 'IDADE' and var2 != 'IDADE':
            # Correlação parcial controlando IDADE
            try:
                # Calcular manualmente correlação parcial
                # r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz²)(1 - r_yz²))
                r_xy = df_partial[[var1, var2]].corr().iloc[0, 1]
                r_x_idade = df_partial[[var1, 'IDADE']].corr().iloc[0, 1]
                r_y_idade = df_partial[[var2, 'IDADE']].corr().iloc[0, 1]
                
                numerator = r_xy - (r_x_idade * r_y_idade)
                denominator = np.sqrt((1 - r_x_idade**2) * (1 - r_y_idade**2))
                
                if denominator != 0:
                    r_partial = numerator / denominator
                else:
                    r_partial = np.nan
                
                partial_corr_results.append({
                    'Var1': var1,
                    'Var2': var2,
                    'Corr_Simples': r_xy,
                    'Corr_Parcial_sem_IDADE': r_partial,
                    'Diferença': r_partial - r_xy
                })
                    
            except Exception as e:
                continue

# Salvar resultados
if partial_corr_results:
    df_partial_corr = pd.DataFrame(partial_corr_results)
    df_partial_corr = df_partial_corr.sort_values('Corr_Parcial_sem_IDADE', ascending=False, key=abs)
    df_partial_corr.to_csv('17c_partial_correlations.csv', index=False)

# ============================================================================
# 11. PAIRPLOT (Visão Geral de Todas as Relações)
# ============================================================================
print("[11/11] Gerar pairplot (pode demorar)...")
# Pairplot com todas as variáveis numéricas, colorido por TARGET
df_pairplot = df[numeric_vars + ['NORMAL X ANORMAL']].dropna()

# Limitar amostra se muito grande (para performance)
if len(df_pairplot) > 1000:
    df_pairplot = df_pairplot.sample(n=1000, random_state=42)
    print(f"   Usando amostra de 1000 registos")

pairplot = sns.pairplot(df_pairplot, hue='NORMAL X ANORMAL', 
                        palette={'Normal': '#2ecc71', 'Anormal': '#e74c3c'},
                        diag_kind='kde', plot_kws={'alpha': 0.6, 's': 20},
                        height=2.5, aspect=1.0)

pairplot.fig.suptitle('Pairplot: Todas as Variáveis Numéricas (Normal vs Anormal)', 
                      fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('17d_pairplot_all_variables.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nAnálise bivariada concluída")

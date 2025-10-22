import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configuracoes de visualizacao
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

# Carregar dados limpos
df = pd.read_csv('UCMF_CLEAN.csv')

# Definir variaveis numericas para analise
numeric_vars = ['IDADE', 'Peso', 'Altura', 'IMC', 'PA SISTOLICA', 'PA DIASTOLICA', 'FC']

# Summaries - calcular estatisticas resumo (media, mediana, quartis, desvio padrao, variancia, amplitude, IQR, assimetria, curtose)
# Criar tabela vazia para guardar estatisticas
stats_summary = pd.DataFrame()

# Calcular estatisticas para cada variavel numerica
for var in numeric_vars:
    if var in df.columns:
        # Remover valores em falta para calculos
        data = df[var].dropna()
        
        # Calcular todas as medidas estatisticas
        stats_summary[var] = {
            'Count': len(data),                              # Numero de valores validos
            'Missing': df[var].isna().sum(),                 # Numero de valores em falta
            'Mean': data.mean(),                             # Media
            'Std': data.std(),                               # Desvio padrao
            'Min': data.min(),                               # Minimo
            '25%': data.quantile(0.25),                      # 1º Quartil
            'Median': data.median(),                         # Mediana (2º Quartil)
            '75%': data.quantile(0.75),                      # 3º Quartil
            'Max': data.max(),                               # Maximo
            'Range': data.max() - data.min(),                # Amplitude (max - min)
            'IQR': data.quantile(0.75) - data.quantile(0.25), # Amplitude interquartil
            'Variance': data.var(),                          # Variancia
            'Skewness': data.skew(),                         # Assimetria (simetria da distribuicao)
            'Kurtosis': data.kurtosis()                      # Curtose (achatamento da distribuicao)
        }

# Trocar as colunas pelas linhas
stats_summary = stats_summary.T
print("Estatísticas descritivas completas:")
print(stats_summary.round(2))

# Guardar estatisticas em ficheiro CSV
stats_summary.to_csv('01_statistics_summary.csv')

# Histograms - Distribuicoes com densidade
# Criar figura com subplots (3 linhas x 3 colunas)
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.ravel()  # Transformar matriz de eixos em lista

# Criar histograma para cada variavel
for i, var in enumerate(numeric_vars):
    if var in df.columns and i < len(axes):
        data = df[var].dropna()
        
        # Histograma (barras) com densidade normalizada
        axes[i].hist(data, bins=50, alpha=0.7, color='steelblue', edgecolor='black', density=True)
        
        # Curva de densidade (KDE - Kernel Density Estimation)
        data.plot(kind='density', ax=axes[i], color='red', linewidth=2)
        
        # Configurar titulos e labels
        axes[i].set_title(f'Distribuição: {var}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel(var)
        axes[i].set_ylabel('Densidade')
        axes[i].grid(True, alpha=0.3)
        
        # Adicionar linhas verticais para media e mediana
        mean_val = data.mean()
        median_val = data.median()
        axes[i].axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Media: {mean_val:.1f}')
        axes[i].axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Mediana: {median_val:.1f}')
        axes[i].legend()

# Remover eixos vazios
for i in range(len(numeric_vars), len(axes)):
    fig.delaxes(axes[i])

# Guardar grafico
plt.tight_layout()
plt.savefig('02_histograms_density.png', dpi=300, bbox_inches='tight')
plt.close()

# Boxplots - Identificar valores fora do padrao (outliers)
# Criar figura com subplots (2 linhas x 4 colunas)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()

# Criar boxplot para cada variavel
for i, var in enumerate(numeric_vars):
    if var in df.columns:
        data = df[var].dropna()
        
        # Criar boxplot
        bp = axes[i].boxplot(data, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')      # Cor da caixa
        bp['boxes'][0].set_edgecolor('black')          # Contorno da caixa
        bp['medians'][0].set_color('red')              # Linha da mediana em vermelho
        bp['medians'][0].set_linewidth(2)
        
        # Configurar titulos
        axes[i].set_title(f'Boxplot: {var}', fontsize=12, fontweight='bold')
        axes[i].set_ylabel(var)
        axes[i].grid(True, alpha=0.3, axis='y')
        
        # Calcular e mostrar numeros fora do padrao
        # Outliers: valores fora de [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        # Adicionar texto com numero de outliers
        axes[i].text(0.5, 0.95, f'Outliers: {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)',
                    transform=axes[i].transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Remover eixo vazio
fig.delaxes(axes[7])

# Guardar grafico
plt.tight_layout()
plt.savefig('03_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()

# Spread Measures - Medidas de dispersao
# Criar tabela vazia para medidas de dispersao
spread_measures = pd.DataFrame()

# Calcular medidas de dispersao para cada variavel
for var in numeric_vars:
    if var in df.columns:
        data = df[var].dropna()
        
        spread_measures[var] = {
            'Range': data.max() - data.min(),                # Amplitude (max - min)
            'IQR': data.quantile(0.75) - data.quantile(0.25), # Amplitude interquartil (Q3 - Q1)
            'Variance': data.var(),                          # Variancia
            'Std_Dev': data.std(),                           # Desvio padrao
            'CV': (data.std() / data.mean()) * 100,          # Coeficiente de variacao (%)
            'MAD': np.mean(np.abs(data - data.mean()))       # Desvio absoluto medio
        }

# Trocar linhas por colunas
spread_measures = spread_measures.T
print("\nMedidas de Dispersão:")
print(spread_measures.round(2))

# Guardar em CSV
spread_measures.to_csv('04_spread_measures.csv')

# Distribuicao da Variavel TARGET
# Criar figura com 2 subplots (grafico de barras + pizza)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Contar numero de Normal vs Anormal
target_counts = df['NORMAL X ANORMAL'].value_counts()
colors = ['#2ecc71', '#e74c3c']  # Verde para Normal, Vermelho para Anormal

# Grafico de barras
axes[0].bar(target_counts.index, target_counts.values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
axes[0].set_title('Distribuição da Variável TARGET', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Número de Pacientes', fontsize=12)
axes[0].set_xlabel('Classe', fontsize=12)
axes[0].grid(True, alpha=0.3, axis='y')

# Adicionar valores em cima das barras
for i, v in enumerate(target_counts.values):
    axes[0].text(i, v + 100, f'{v}\n({v/len(df)*100:.1f}%)', 
                ha='center', va='bottom', fontweight='bold', fontsize=11)

# Grafico pizza (proporcao)
axes[1].pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90, explode=(0.05, 0.05),
            textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[1].set_title('Proporção Normal vs Anormal', fontsize=14, fontweight='bold')

# Guardar grafico
plt.tight_layout()
plt.savefig('05_target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Distribuições categóricas
# Definir variaveis categoricas para analisar
categorical_vars = ['SEXO', 'PULSOS', 'SOPRO', 'B2', 'PPA']

# Criar figura com subplots (2 linhas x 3 colunas)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

# Criar grafico de barras para cada variavel categorica
for i, var in enumerate(categorical_vars):
    if var in df.columns and i < len(axes):
        # Contar frequencia de cada categoria (top 10)
        counts = df[var].value_counts().head(10)
        
        # Grafico de barras
        axes[i].bar(range(len(counts)), counts.values, color='teal', alpha=0.8, edgecolor='black')
        axes[i].set_title(f'Distribuição: {var}', fontsize=12, fontweight='bold')
        axes[i].set_xticks(range(len(counts)))
        axes[i].set_xticklabels(counts.index, rotation=45, ha='right')
        axes[i].set_ylabel('Frequencia', fontsize=10)
        axes[i].grid(True, alpha=0.3, axis='y')
        
        # Adicionar valores em cima das barras
        for j, v in enumerate(counts.values):
            axes[i].text(j, v + max(counts.values)*0.02, str(v), 
                        ha='center', va='bottom', fontsize=9)

# Remover eixo vazio
fig.delaxes(axes[5])

# Salvar grafico
plt.tight_layout()
plt.savefig('06_categorical_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

# Teste de normalidade
# Testar se os dados seguem uma distribuicao normal
normality_tests = pd.DataFrame()

for var in numeric_vars:
    if var in df.columns:
        data = df[var].dropna()
        
        # Shapiro-Wilk test (para amostras pequenas < 5000)
        # Kolmogorov-Smirnov test (para amostras grandes)
        if len(data) < 5000:
            stat, p_value = stats.shapiro(data[:5000])
        else:
            stat, p_value = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
        
        normality_tests[var] = {
            'Statistic': stat,
            'P-value': p_value,
            'Normal?': 'Yes' if p_value > 0.05 else 'No'  # Se p > 0.05, aceita normalidade
        }

# Trocar as linhas com as colunas
normality_tests = normality_tests.T
print("\nTestes de Normalidade (alpha=0.05):")
print(normality_tests.round(4))

# Guardar em CSV
normality_tests.to_csv('07_normality_tests.csv')
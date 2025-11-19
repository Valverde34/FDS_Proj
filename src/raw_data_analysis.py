import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# Configuracoes de visualizacao
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

# Carregar dados brutos (originais)
df = pd.read_excel(r'C:\Uni\1_ano\1_semestre\CD\FDS_Proj\UCMF.xls')

print("Análise dos dados brutos (RAW DATA)\n")
print(f"Total de registos: {len(df)}")
print(f"Total de variáveis: {len(df.columns)}")

# DESCRICAO DOS TIPOS DE VARIAVEIS
tipos_variaveis = {
    'ID': 'Nominal (identificador unico)',
    'Peso': 'Ratio (peso em kg, tem zero absoluto)',
    'Altura': 'Ratio (altura em cm, tem zero absoluto)',
    'IMC': 'Ratio (indice de massa corporal)',
    'Atendimento': 'Intervalar (data/timestamp - eventos no tempo)',
    'DN': 'Intervalar (data de nascimento - eventos no tempo)',
    'IDADE': 'Ratio (idade em anos)',
    'Convenio': 'Nominal (tipo de seguro de saude)',
    'PULSOS': 'Nominal (tipo de pulso)',
    'PA SISTOLICA': 'Ratio (pressao arterial sistolica em mmHg)',
    'PA DIASTOLICA': 'Ratio (pressao arterial diastolica em mmHg)',
    'PPA': 'Ratio (indice calculado - pressao de pulso arterial)',
    'NORMAL X ANORMAL': 'Binary (variavel target: Normal ou Anormal)',
    'B2': 'Nominal (tipo do segundo som cardiaco)',
    'SOPRO': 'Nominal (tipo de sopro cardiaco)',
    'FC': 'Ratio (frequencia cardiaca em bpm)',
    'HDA 1': 'Nominal (historia da doenca 1)',
    'HDA2': 'Nominal (historia da doenca 2)',
    'SEXO': 'Nominal (genero do paciente)',
    'MOTIVO1': 'Nominal (primeiro motivo de encaminhamento)',
    'MOTIVO2': 'Nominal (segundo motivo de encaminhamento)'
}

for var, tipo in tipos_variaveis.items():
    print(f"{var:20s} -> {tipo}")

# Guardar tipos de variaveis em CSV
tipos_df = pd.DataFrame(list(tipos_variaveis.items()), columns=['Variavel', 'Tipo'])
tipos_df.to_csv('raw_01_tipos_variaveis.csv', index=False)

# IDENTIFICAR PROBLEMAS NOS DADOS BRUTOS
problemas = []

# Converter colunas para numerico ANTES de qualquer analise
df['IDADE_num'] = pd.to_numeric(df['IDADE'], errors='coerce')
df['Peso_num'] = pd.to_numeric(df['Peso'], errors='coerce')
df['Altura_num'] = pd.to_numeric(df['Altura'], errors='coerce')
df['IMC_num'] = pd.to_numeric(df['IMC'], errors='coerce')
df['PA_SIST_num'] = pd.to_numeric(df['PA SISTOLICA'], errors='coerce')
df['PA_DIAST_num'] = pd.to_numeric(df['PA DIASTOLICA'], errors='coerce')
df['FC_num'] = pd.to_numeric(df['FC'], errors='coerce')

# Duplicados
duplicados = df.duplicated(subset=['ID']).sum()
print(f"\nDuplicados (por ID): {duplicados}")
problemas.append(f"Duplicados: {duplicados} registos")

# Idades problematicas
idades_negativas = (df['IDADE_num'] < 0).sum()
idades_maiores = (df['IDADE_num'] > 19).sum()
print(f"Idades negativas: {idades_negativas}")
print(f"Idades > 19 anos: {idades_maiores}")
problemas.append(f"Idades negativas: {idades_negativas}")
problemas.append(f"Idades > 19 anos: {idades_maiores}")

# Pesos e alturas invalidos
pesos_zero = (df['Peso_num'] == 0).sum()
alturas_zero = (df['Altura_num'] == 0).sum()
pesos_negativos = (df['Peso_num'] < 0).sum()
print(f"\nPeso = 0: {pesos_zero}")
print(f"Altura = 0: {alturas_zero}")
print(f"Peso negativo: {pesos_negativos}")
problemas.append(f"Peso = 0: {pesos_zero}")
problemas.append(f"Altura = 0: {alturas_zero}")
problemas.append(f"Peso negativo: {pesos_negativos}")

# Missing values significativos
print(f"\nMissing values por coluna (> 5%):")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_significativo = missing_pct[missing_pct > 5].sort_values(ascending=False)
for col, pct in missing_significativo.items():
    print(f"  {col}: {missing[col]} ({pct:.1f}%)")
    problemas.append(f"{col}: {missing[col]} missing ({pct:.1f}%)")

# Salvar missing values completo em CSV
missing_df = pd.DataFrame({
    'Variavel': df.columns,
    'Missing_Count': missing.values,
    'Missing_Percentage': missing_pct.values
}).sort_values('Missing_Percentage', ascending=False)
missing_df.to_csv('raw_02b_missing_values.csv', index=False)
print(f"\n✓ Missing values salvos em raw_02b_missing_values.csv")

# Inconsistencias categoricas
print(f"\nInconsistências em variaveis categóricas:")

# SEXO
sexo_unique = df['SEXO'].value_counts()
print(f"  SEXO - valores únicos: {len(sexo_unique)}")
print(f"    {dict(sexo_unique.head(5))}")
problemas.append(f"SEXO: {len(sexo_unique)} valores diferentes (precisa padronização)")

# TARGET
target_unique = df['NORMAL X ANORMAL'].value_counts()
print(f"  NORMAL X ANORMAL - valores únicos: {len(target_unique)}")
print(f"    {dict(target_unique)}")
problemas.append(f"TARGET: {len(target_unique)} valores diferentes (precisa padronização)")

# Identificar valores fisiologicamente impossiveis
print(f"\n" + "="*80)
print("VALORES FISIOLOGICAMENTE IMPOSSÍVEIS (Pediatria: 0-19 anos)")
print("="*80)

impossible_values = []

# FC > 300 bpm (impossivel mesmo em neonatos)
fc_impossible = df[df['FC_num'] > 300]
if len(fc_impossible) > 0:
    print(f"\n FC > 300 bpm: {len(fc_impossible)} casos")
    print(f"   Valores: {sorted(fc_impossible['FC_num'].dropna().unique())[:10]}")
    impossible_values.append({'Variavel': 'FC', 'Condicao': '> 300 bpm', 'Casos': len(fc_impossible)})

# PA Sistolica > 300 mmHg (impossivel)
pa_sist_impossible = df[df['PA_SIST_num'] > 300]
if len(pa_sist_impossible) > 0:
    print(f"\n PA SISTOLICA > 300 mmHg: {len(pa_sist_impossible)} casos")
    print(f"   Valores: {sorted(pa_sist_impossible['PA_SIST_num'].dropna().unique())[:10]}")
    impossible_values.append({'Variavel': 'PA_SISTOLICA', 'Condicao': '> 300 mmHg', 'Casos': len(pa_sist_impossible)})

# PA Diastolica > 200 mmHg (impossivel)
pa_diast_impossible = df[df['PA_DIAST_num'] > 200]
if len(pa_diast_impossible) > 0:
    print(f"\n PA DIASTOLICA > 200 mmHg: {len(pa_diast_impossible)} casos")
    print(f"   Valores: {sorted(pa_diast_impossible['PA_DIAST_num'].dropna().unique())[:10]}")
    impossible_values.append({'Variavel': 'PA_DIASTOLICA', 'Condicao': '> 200 mmHg', 'Casos': len(pa_diast_impossible)})

# IMC > 60 (extremamente raro, suspeito)
imc_impossible = df[df['IMC_num'] > 60]
if len(imc_impossible) > 0:
    print(f"\n IMC > 60: {len(imc_impossible)} casos (provavelmente erro)")
    print(f"   Valores: {sorted(imc_impossible['IMC_num'].dropna().unique())[:10]}")
    impossible_values.append({'Variavel': 'IMC', 'Condicao': '> 60', 'Casos': len(imc_impossible)})

# Altura < 30 cm (impossivel para crianca viva)
altura_impossible = df[df['Altura_num'] < 30]
if len(altura_impossible) > 0:
    print(f"\n Altura < 30 cm: {len(altura_impossible)} casos")
    print(f"   Valores: {sorted(altura_impossible['Altura_num'].dropna().unique())[:10]}")
    impossible_values.append({'Variavel': 'Altura', 'Condicao': '< 30 cm', 'Casos': len(altura_impossible)})

# Peso < 1 kg (improvavel exceto prematuros extremos)
peso_impossible = df[df['Peso_num'] < 1]
if len(peso_impossible) > 0:
    print(f"\n Peso < 1 kg: {len(peso_impossible)} casos")
    print(f"   Valores: {sorted(peso_impossible['Peso_num'].dropna().unique())[:10]}")
    impossible_values.append({'Variavel': 'Peso', 'Condicao': '< 1 kg', 'Casos': len(peso_impossible)})

# Idade < 0 ou > 19
idade_impossible = df[(df['IDADE_num'] < 0) | (df['IDADE_num'] > 19)]
if len(idade_impossible) > 0:
    print(f"\n IDADE fora do intervalo [0-19]: {len(idade_impossible)} casos")
    impossible_values.append({'Variavel': 'IDADE', 'Condicao': '< 0 ou > 19 anos', 'Casos': len(idade_impossible)})

# Salvar valores impossiveis
if impossible_values:
    impossible_df = pd.DataFrame(impossible_values)
    impossible_df.to_csv('raw_02c_impossible_physiological_values.csv', index=False)
    print(f"\n✓ Valores impossíveis salvos em raw_02c_impossible_physiological_values.csv")
else:
    print(f"\n✓ Nenhum valor fisiologicamente impossível detectado.")

# Guardar lista de problemas
problemas_df = pd.DataFrame(problemas, columns=['Problema'])
problemas_df.to_csv('raw_02_problemas_identificados.csv', index=False)

# ESTATISTICAS DESCRITIVAS DOS DADOS BRUTOS (COMPLETO)
print(f"\n" + "="*80)
print("ESTATÍSTICAS DESCRITIVAS COMPLETAS")
print("="*80)

# Variaveis numericas ja convertidas anteriormente
stats_raw = pd.DataFrame()

var_mapping = {
    'IDADE_num': 'IDADE',
    'Peso_num': 'Peso',
    'Altura_num': 'Altura',
    'IMC_num': 'IMC',
    'PA_SIST_num': 'PA SISTOLICA',
    'PA_DIAST_num': 'PA DIASTOLICA',
    'FC_num': 'FC'
}

for var_code, var_name in var_mapping.items():
    if var_code in df.columns:
        data = df[var_code].dropna()
        
        if len(data) > 0:
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            
            stats_raw[var_name] = {
                'Count': len(data),
                'Missing': df[var_code].isna().sum(),
                'Mean': data.mean(),
                'Median': data.median(),
                'Std': data.std(),
                'Min': data.min(),
                'Q1': q1,
                'Q3': q3,
                'Max': data.max(),
                'IQR': iqr,
                'P5': data.quantile(0.05),
                'P95': data.quantile(0.95),
                'P1': data.quantile(0.01),
                'P99': data.quantile(0.99),
                'Outliers_Extremos': ((data < data.quantile(0.01)) | (data > data.quantile(0.99))).sum()
            }

stats_raw = stats_raw.T
print(stats_raw.round(2))

stats_raw.to_csv('raw_03_statistics_raw.csv')
print(f"\n✓ Estatísticas completas salvas em raw_03_statistics_raw.csv")

# HISTOGRAMAS DOS DADOS BRUTOS
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.ravel()

var_list = ['IDADE_num', 'Peso_num', 'Altura_num', 'IMC_num', 'PA_SIST_num', 'PA_DIAST_num', 'FC_num']
var_names = ['IDADE', 'Peso', 'Altura', 'IMC', 'PA SISTOLICA', 'PA DIASTOLICA', 'FC']

for i, (var, name) in enumerate(zip(var_list, var_names)):
    if var in df.columns and i < len(axes):
        data = df[var].dropna()
        
        if len(data) > 0:
            axes[i].hist(data, bins=50, alpha=0.7, color='coral', edgecolor='black')
            axes[i].set_title(f'BRUTO: {name}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel(name)
            axes[i].set_ylabel('Frequência')
            axes[i].grid(True, alpha=0.3)
            
            # Marcar valores problematicos
            mean_val = data.mean()
            axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_val:.1f}')
            axes[i].legend()

for i in range(len(var_list), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig('raw_04_histograms_raw.png', dpi=300, bbox_inches='tight')

plt.close()

# BOXPLOTS DOS DADOS BRUTOS
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()

for i, (var, name) in enumerate(zip(var_list, var_names)):
    if var in df.columns:
        data = df[var].dropna()
        
        if len(data) > 0:
            bp = axes[i].boxplot(data, vert=True, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightsalmon')
            bp['boxes'][0].set_edgecolor('black')
            bp['medians'][0].set_color('red')
            bp['medians'][0].set_linewidth(2)
            
            axes[i].set_title(f'BRUTO: {name}', fontsize=12, fontweight='bold')
            axes[i].set_ylabel(name)
            axes[i].grid(True, alpha=0.3, axis='y')
            
            # Mostrar numero de outliers
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = data[(data < lower) | (data > upper)]
            
            axes[i].text(0.5, 0.95, f'Outliers: {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)',
                        transform=axes[i].transAxes, ha='center', va='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.delaxes(axes[7])

plt.tight_layout()
plt.savefig('raw_05_boxplots_raw.png', dpi=300, bbox_inches='tight')
plt.close()

# DISTRIBUICAO TARGET (BRUTO)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

target_counts = df['NORMAL X ANORMAL'].value_counts()
colors = ['#e74c3c', '#3498db', '#95a5a6']

axes[0].bar(range(len(target_counts)), target_counts.values, color=colors[:len(target_counts)], 
            alpha=0.8, edgecolor='black', linewidth=2)
axes[0].set_title('BRUTO: Distribuição TARGET', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Número de Pacientes', fontsize=12)
axes[0].set_xlabel('Classe', fontsize=12)
axes[0].set_xticks(range(len(target_counts)))
axes[0].set_xticklabels(target_counts.index, rotation=45, ha='right')
axes[0].grid(True, alpha=0.3, axis='y')

for i, v in enumerate(target_counts.values):
    axes[0].text(i, v + 100, f'{v}\n({v/len(df)*100:.1f}%)', 
                ha='center', va='bottom', fontweight='bold', fontsize=11)

axes[1].pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%', 
            colors=colors[:len(target_counts)], startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[1].set_title('BRUTO: Proporção TARGET', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('raw_06_target_distribution_raw.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "="*80)
print("ANÁLISE DE DENSIDADE (KDE) - VARIÁVEIS NUMÉRICAS")
print("="*80)

# KDE Plots
fig, axes = plt.subplots(3, 3, figsize=(20, 14))
axes = axes.ravel()

var_list = ['IDADE_num', 'Peso_num', 'Altura_num', 'IMC_num', 'PA_SIST_num', 'PA_DIAST_num', 'FC_num']
var_names = ['IDADE', 'Peso', 'Altura', 'IMC', 'PA SISTOLICA', 'PA DIASTOLICA', 'FC']

for i, (var, name) in enumerate(zip(var_list, var_names)):
    if var in df.columns and i < len(axes):
        data = df[var].dropna()
        
        if len(data) > 0:
            # Histograma + KDE
            axes[i].hist(data, bins=50, alpha=0.5, color='skyblue', edgecolor='black', density=True, label='Histograma')
            
            # KDE
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            axes[i].plot(x_range, kde(x_range), color='red', linewidth=2.5, label='KDE')
            
            axes[i].set_title(f'Densidade: {name}', fontsize=13, fontweight='bold')
            axes[i].set_xlabel(name, fontsize=11)
            axes[i].set_ylabel('Densidade', fontsize=11)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

for i in range(len(var_list), len(axes)):
    fig.delaxes(axes[i])

plt.suptitle('Análise de Densidade (KDE) - Dados Brutos', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('raw_07_density_kde.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Análise de densidade (KDE) salva: raw_07_density_kde.png")

# ============================================================================
# ANÁLISE DE TODAS AS VARIÁVEIS CATEGÓRICAS
# ============================================================================
print("\n" + "="*80)
print("ANÁLISE COMPLETA DE VARIÁVEIS CATEGÓRICAS")
print("="*80)

categorical_vars = ['SEXO', 'NORMAL X ANORMAL', 'Convenio', 'PULSOS', 'B2', 'SOPRO', 
                   'MOTIVO1', 'MOTIVO2', 'HDA 1', 'HDA2']

categorical_analysis = []
rare_categories = []  # Para categorias com frequência < 1%

for var in categorical_vars:
    if var in df.columns:
        print(f"\n{var}:")
        counts = df[var].value_counts(dropna=False)
        percentages = (counts / len(df)) * 100
        
        print(f"  Valores únicos: {len(counts)}")
        print(f"  Top 10 categorias:")
        
        for idx, (cat, count) in enumerate(counts.head(10).items()):
            pct = percentages[cat]
            print(f"    {idx+1}. '{cat}': {count} ({pct:.2f}%)")
            
            categorical_analysis.append({
                'Variavel': var,
                'Categoria': str(cat),
                'Frequencia': count,
                'Percentagem': pct
            })
            
            # Identificar categorias raras (< 1%)
            if pct < 1.0 and pct > 0:
                rare_categories.append({
                    'Variavel': var,
                    'Categoria': str(cat),
                    'Frequencia': count,
                    'Percentagem': pct
                })

# Salvar análise categórica completa
cat_df = pd.DataFrame(categorical_analysis)
cat_df.to_csv('raw_08_categorical_analysis.csv', index=False)
print(f"\n✓ Análise categórica completa salva: raw_08_categorical_analysis.csv")

# Salvar categorias raras
if rare_categories:
    rare_df = pd.DataFrame(rare_categories)
    rare_df = rare_df.sort_values('Percentagem', ascending=True)
    rare_df.to_csv('raw_09_rare_categories.csv', index=False)
    print(f"✓ Categorias raras (<1%) salvas: raw_09_rare_categories.csv")
    print(f"  Total de categorias raras: {len(rare_categories)}")
else:
    print(f"  Nenhuma categoria rara (<1%) detectada.")

# Gráficos de barras para variáveis categóricas principais
print("\n" + "="*80)
print("GRÁFICOS DE FREQUÊNCIA - VARIÁVEIS CATEGÓRICAS")
print("="*80)

fig, axes = plt.subplots(3, 3, figsize=(22, 16))
axes = axes.ravel()

for idx, var in enumerate(categorical_vars[:9]):
    if var in df.columns:
        counts = df[var].value_counts().head(15)  # Top 15 categorias
        
        axes[idx].barh(range(len(counts)), counts.values, color='teal', alpha=0.7, edgecolor='black')
        axes[idx].set_yticks(range(len(counts)))
        axes[idx].set_yticklabels([str(x)[:30] for x in counts.index], fontsize=9)  # Truncar labels longos
        axes[idx].set_xlabel('Frequência', fontsize=10, fontweight='bold')
        axes[idx].set_title(f'{var} (Top 15)', fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='x')
        
        # Adicionar percentagens
        for i, v in enumerate(counts.values):
            pct = (v / len(df)) * 100
            axes[idx].text(v + max(counts.values)*0.01, i, f'{pct:.1f}%', 
                          va='center', fontsize=8, fontweight='bold')

for idx in range(len(categorical_vars[:9]), 9):
    fig.delaxes(axes[idx])

plt.suptitle('Distribuição de Variáveis Categóricas - Dados Brutos', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('raw_10_categorical_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

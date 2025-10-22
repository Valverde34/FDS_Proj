import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    'Atendimento': 'Ordinal (data de atendimento)',
    'DN': 'Ordinal (data de nascimento)',
    'IDADE': 'Ratio (idade em anos)',
    'Convenio': 'Nominal (tipo de seguro de saude)',
    'PULSOS': 'Nominal (tipo de pulso)',
    'PA SISTOLICA': 'Ratio (pressao arterial sistolica em mmHg)',
    'PA DIASTOLICA': 'Ratio (pressao arterial diastolica em mmHg)',
    'PPA': 'Ordinal (resultado calculado de PA)',
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

# Duplicados
duplicados = df.duplicated(subset=['ID']).sum()
print(f"\nDuplicados (por ID): {duplicados}")
problemas.append(f"Duplicados: {duplicados} registos")

# Idades problematicas
df['IDADE_num'] = pd.to_numeric(df['IDADE'], errors='coerce')
idades_negativas = (df['IDADE_num'] < 0).sum()
idades_maiores = (df['IDADE_num'] > 19).sum()
print(f"Idades negativas: {idades_negativas}")
print(f"Idades > 19 anos: {idades_maiores}")
problemas.append(f"Idades negativas: {idades_negativas}")
problemas.append(f"Idades > 19 anos: {idades_maiores}")

# Pesos e alturas invalidos
df['Peso_num'] = pd.to_numeric(df['Peso'], errors='coerce')
df['Altura_num'] = pd.to_numeric(df['Altura'], errors='coerce')
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

# Guardar lista de problemas
problemas_df = pd.DataFrame(problemas, columns=['Problema'])
problemas_df.to_csv('raw_02_problemas_identificados.csv', index=False)

# ESTATISTICAS DESCRITIVAS DOS DADOS BRUTOS
numeric_vars = ['IDADE_num', 'Peso_num', 'Altura_num', 'IMC', 'PA SISTOLICA', 'PA DIASTOLICA', 'FC']
df['IMC_num'] = pd.to_numeric(df['IMC'], errors='coerce')
df['PA_SIST_num'] = pd.to_numeric(df['PA SISTOLICA'], errors='coerce')
df['PA_DIAST_num'] = pd.to_numeric(df['PA DIASTOLICA'], errors='coerce')
df['FC_num'] = pd.to_numeric(df['FC'], errors='coerce')

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
            stats_raw[var_name] = {
                'Count': len(data),
                'Missing': df[var_code].isna().sum(),
                'Mean': data.mean(),
                'Std': data.std(),
                'Min': data.min(),
                'Max': data.max(),
                'Outliers_Extremos': ((data < data.quantile(0.01)) | (data > data.quantile(0.99))).sum()
            }

stats_raw = stats_raw.T
print(stats_raw.round(2))

stats_raw.to_csv('raw_03_statistics_raw.csv')

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

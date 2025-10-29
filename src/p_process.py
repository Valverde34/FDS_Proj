#Usar o outro pre process, depois eliminamos isto


import pandas as pd
import numpy as np

# Carregar dados
df = pd.read_excel(r'C:\Uni\1_ano\1_semestre\CD\FDS_Proj\UCMF.xls')

# Remover repetidos
inicial = len(df)
df = df.drop_duplicates(subset=['ID'], keep='first')
removidos = inicial - len(df)

# Limpar idade
# Converter para numérico
df['IDADE'] = pd.to_numeric(df['IDADE'], errors='coerce')

# Contar problemas antes de remover
idades_negativas = (df['IDADE'] < 0).sum()
idades_maiores = (df['IDADE'] > 19).sum()
print(f"   - Idades negativas encontradas: {idades_negativas}")
print(f"   - Idades > 19 anos encontradas: {idades_maiores}")

# Filtrar apenas idades válidas (0-19 anos)
df = df[(df['IDADE'] >= 0) & (df['IDADE'] <= 19)]

# Limpar peso
df['Peso'] = pd.to_numeric(df['Peso'], errors='coerce')

# Valores inválidos: <= 0, < 0.5 kg (muito baixo), > 150 kg (muito alto para crianças)
pesos_invalidos = ((df['Peso'] <= 0) | (df['Peso'] < 0.5) | (df['Peso'] > 150)).sum()
print(f"   - Pesos inválidos encontrados: {pesos_invalidos}")

df.loc[(df['Peso'] <= 0) | (df['Peso'] < 0.5) | (df['Peso'] > 150), 'Peso'] = np.nan

# Limpar altura
df['Altura'] = pd.to_numeric(df['Altura'], errors='coerce')

# Valores inválidos: <= 0, < 30 cm (muito baixo), > 200 cm (muito alto)
alturas_invalidas = ((df['Altura'] <= 0) | (df['Altura'] < 30) | (df['Altura'] > 200)).sum()
print(f"   - Alturas inválidas encontradas: {alturas_invalidas}")

df.loc[(df['Altura'] <= 0) | (df['Altura'] < 30) | (df['Altura'] > 200), 'Altura'] = np.nan

# Recalcular IMC
# IMC = Peso(kg) / (Altura(m))^2
df['IMC'] = np.where(
    (df['Peso'].notna()) & (df['Altura'].notna()) & (df['Altura'] > 0),
    df['Peso'] / ((df['Altura'] / 100) ** 2),
    np.nan
)

# Validar IMC: deve estar entre 10 e 40
imc_calculados = df['IMC'].notna().sum()
print(f"   - IMC calculados: {imc_calculados}")

# Remover IMCs impossíveis
df.loc[(df['IMC'] < 10) | (df['IMC'] > 40), 'IMC'] = np.nan

# Limpar Pressão Arterial
df['PA SISTOLICA'] = pd.to_numeric(df['PA SISTOLICA'], errors='coerce')
df['PA DIASTOLICA'] = pd.to_numeric(df['PA DIASTOLICA'], errors='coerce')

# Valores normais em crianças: PA Sistólica 60-180, PA Diastólica 30-120
sistolicas_invalidas = ((df['PA SISTOLICA'] < 60) | (df['PA SISTOLICA'] > 180)).sum()
diastolicas_invalidas = ((df['PA DIASTOLICA'] < 30) | (df['PA DIASTOLICA'] > 120)).sum()

print(f"   - PA Sistólicas inválidas: {sistolicas_invalidas}")
print(f"   - PA Diastólicas inválidas: {diastolicas_invalidas}")

df.loc[(df['PA SISTOLICA'] < 60) | (df['PA SISTOLICA'] > 180), 'PA SISTOLICA'] = np.nan
df.loc[(df['PA DIASTOLICA'] < 30) | (df['PA DIASTOLICA'] > 120), 'PA DIASTOLICA'] = np.nan

# Limpar Frequência Cardíaca
df['FC'] = pd.to_numeric(df['FC'], errors='coerce')

# Valores normais em crianças: 40-200 bpm
fc_invalidas = ((df['FC'] < 40) | (df['FC'] > 200)).sum()
print(f"   - FC inválidas encontradas: {fc_invalidas}")

df.loc[(df['FC'] < 40) | (df['FC'] > 200), 'FC'] = np.nan

# Sexo
df['SEXO'] = df['SEXO'].astype(str).str.upper().str.strip()
df['SEXO'] = df['SEXO'].replace({
    'MASCULINO': 'M',
    'FEMININO': 'F',
    'INDETERMINADO': 'I',
    'NAN': np.nan,
    'NONE': np.nan
})
sexo_antes = len(df)
df = df[df['SEXO'].notna()]
print(f"   - SEXO: padronizado (M/F/I), removidos {sexo_antes - len(df)} registos com sexo nulo")

# Normal x Anormal
df['NORMAL X ANORMAL'] = df['NORMAL X ANORMAL'].astype(str).str.capitalize().str.strip()
df['NORMAL X ANORMAL'] = df['NORMAL X ANORMAL'].replace({
    'Normais': 'Normal',
    'Nan': np.nan,
    'None': np.nan
})
target_antes = len(df)
df = df[df['NORMAL X ANORMAL'].notna()]
df = df[df['NORMAL X ANORMAL'].isin(['Normal', 'Anormal'])]
print(f"   - TARGET: padronizado (Normal/Anormal), removidos {target_antes - len(df)} registos inválidos")

# Pulsos
df['PULSOS'] = df['PULSOS'].astype(str).str.capitalize().str.strip()
df['PULSOS'] = df['PULSOS'].replace({
    'Normais': 'Normal',
    'Amplos': 'Amplo',
    'Femorais diminuidos': 'Diminuido',
    'Diminuídos': 'Diminuido',
    'Diminuídos ': 'Diminuido',
    'Nan': np.nan,
    'None': np.nan
})
print(f"   - PULSOS: padronizado")

# Sopro
df['SOPRO'] = df['SOPRO'].astype(str).str.capitalize().str.strip()
df['SOPRO'] = df['SOPRO'].replace({
    'Sistólico': 'Sistolico',
    'Contínuo': 'Continuo',
    'Diastólico': 'Diastolico',
    'Sistolico e diastólico': 'Sistolico_Diastolico',
    'Nan': np.nan,
    'None': np.nan
})
print(f"   - SOPRO: padronizado")

# B2
df['B2'] = df['B2'].astype(str).str.capitalize().str.strip()
df['B2'] = df['B2'].replace({
    'Nan': np.nan,
    'None': np.nan
})

# Remover colunas irrelevantes
# Remover HDA2 
colunas_remover = ['HDA2']
colunas_removidas = [col for col in colunas_remover if col in df.columns]
df = df.drop(columns=colunas_removidas, errors='ignore')
print(f"   - Removidas: {', '.join(colunas_removidas)}")

# Remover colunas de data 
df = df.drop(columns=['Atendimento', 'DN'], errors='ignore')
print(f"   - Removidas: Atendimento, DN (datas não necessárias)")

# Resumo
print("Resumo da Limpeza")
print(f"Registos finais: {len(df)}")
print(f"Colunas finais: {len(df.columns)}")

print(f"\n Distribuição TARGET:")
print(df['NORMAL X ANORMAL'].value_counts())

print(f"\n Valores em falta por coluna:")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
for col, count in missing.items():
    pct = (count/len(df))*100
    print(f"   {col}: {count} ({pct:.1f}%)")

print(f"\n Estatísticas descritivas (variáveis numéricas):")
numeric_cols = ['IDADE', 'Peso', 'Altura', 'IMC', 'PA SISTOLICA', 'PA DIASTOLICA', 'FC']
print(df[numeric_cols].describe().round(2))

# Guardar o dataset
output_file = 'UCMF_CLEAN.csv'
df.to_csv(output_file, index=False)
print(f"\nDataset limpo guardado em: {output_file}")

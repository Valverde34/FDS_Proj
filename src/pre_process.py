import pandas as pd
import numpy as np

# Carregar dados
df = pd.read_excel(r'C:\Users\diogo\OneDrive - Universidade de Aveiro\Documents\M_ERSI\1ano\Int_Ciencia_Dados\FDS_Proj\UCMF.xls')

print("="*80)
print("ANÁLISE DOS DADOS BRUTOS")
print("="*80)

# Informações básicas
print(f"\nDimensões iniciais: {df.shape}")
print(f"Colunas: {list(df.columns)}")

# Verificar colunas com valor constante
print("\nVerificação de colunas com valor único:")
for col in df.columns:
    unique_vals = df[col].nunique()
    if unique_vals == 1:
        print(f"   - {col}: apenas 1 valor único → REMOVER")
    elif unique_vals < 5:
        print(f"   - {col}: {unique_vals} valores únicos")

# Tipos de dados
print("\nTipos de dados:")
print(df.dtypes)

# Missing values nos dados brutos
print("\nValores em falta (dados brutos):")
missing_raw = df.isnull().sum()
missing_raw = missing_raw[missing_raw > 0].sort_values(ascending=False)
for col, count in missing_raw.items():
    pct = (count/len(df))*100
    print(f"   {col}: {count} ({pct:.1f}%)")

print("\n" + "="*80)
print("PRÉ-PROCESSAMENTO")
print("="*80)

# 1. REMOVER DUPLICADOS
inicial = len(df)
df = df.drop_duplicates(subset=['ID'], keep='first')
removidos = inicial - len(df)
print(f"\n1. Duplicados removidos: {removidos}")

# 2. IDADE
print("\n2. Limpeza IDADE:")
df['IDADE'] = pd.to_numeric(df['IDADE'], errors='coerce')
idades_negativas = (df['IDADE'] < 0).sum()
idades_maiores = (df['IDADE'] > 19).sum()
print(f"   - Idades negativas: {idades_negativas}")
print(f"   - Idades > 19 anos: {idades_maiores}")
# Substituir por NaN 
df.loc[(df['IDADE'] < 0) | (df['IDADE'] > 19), 'IDADE'] = np.nan
df['IDADE'] = df['IDADE'].round(2)  

# 3. PESO
print("\n3. Limpeza PESO:")
df['Peso'] = pd.to_numeric(df['Peso'], errors='coerce')
pesos_invalidos = ((df['Peso'] <= 0) | (df['Peso'] < 0.5) | (df['Peso'] > 150)).sum()
print(f"   - Pesos inválidos: {pesos_invalidos}")
df.loc[(df['Peso'] <= 0) | (df['Peso'] < 0.5) | (df['Peso'] > 150), 'Peso'] = np.nan

# 4. ALTURA
print("\n4. Limpeza ALTURA:")
df['Altura'] = pd.to_numeric(df['Altura'], errors='coerce')
alturas_invalidas = ((df['Altura'] <= 0) | (df['Altura'] < 30) | (df['Altura'] > 200)).sum()
print(f"   - Alturas inválidas: {alturas_invalidas}")
df.loc[(df['Altura'] <= 0) | (df['Altura'] < 30) | (df['Altura'] > 200), 'Altura'] = np.nan

# 5. IMC
# Guardar IMC original para comparação
df['IMC_original'] = df['IMC'].copy()

# Recalcular IMC onde está em falta OU onde está inválido
df['IMC_calculado'] = np.where(
    (df['Peso'].notna()) & (df['Altura'].notna()) & (df['Altura'] > 0),
    df['Peso'] / ((df['Altura'] / 100) ** 2),
    np.nan
)

# Usar IMC calculado se o original não existir
df['IMC'] = df['IMC'].fillna(df['IMC_calculado'])

# Remover IMCs impossíveis
df.loc[(df['IMC'] < 10) | (df['IMC'] > 40), 'IMC'] = np.nan

# Limpar colunas temporárias
df = df.drop(columns=['IMC_original', 'IMC_calculado'], errors='ignore')

# 6. PRESSÃO ARTERIAL
print("\n6. Limpeza PRESSÃO ARTERIAL:")
df['PA SISTOLICA'] = pd.to_numeric(df['PA SISTOLICA'], errors='coerce')
df['PA DIASTOLICA'] = pd.to_numeric(df['PA DIASTOLICA'], errors='coerce')
sistolicas_invalidas = ((df['PA SISTOLICA'] < 60) | (df['PA SISTOLICA'] > 180)).sum()
diastolicas_invalidas = ((df['PA DIASTOLICA'] < 30) | (df['PA DIASTOLICA'] > 120)).sum()
print(f"   - PA Sistólicas inválidas: {sistolicas_invalidas}")
print(f"   - PA Diastólicas inválidas: {diastolicas_invalidas}")
df.loc[(df['PA SISTOLICA'] < 60) | (df['PA SISTOLICA'] > 180), 'PA SISTOLICA'] = np.nan
df.loc[(df['PA DIASTOLICA'] < 30) | (df['PA DIASTOLICA'] > 120), 'PA DIASTOLICA'] = np.nan

# 7. FREQUÊNCIA CARDÍACA
print("\n7. Limpeza FC:")
df['FC'] = pd.to_numeric(df['FC'], errors='coerce')
fc_invalidas = ((df['FC'] < 40) | (df['FC'] > 200)).sum()
print(f"   - FC inválidas: {fc_invalidas}")
df.loc[(df['FC'] < 40) | (df['FC'] > 200), 'FC'] = np.nan

# 8. SEXO - padronização
print("\n8. Padronização SEXO:")
df['SEXO'] = df['SEXO'].astype(str).str.upper().str.strip()
df['SEXO'] = df['SEXO'].replace({
    'MASCULINO': 'M',
    'FEMININO': 'F',
    'INDETERMINADO': 'I',
    'NAN': np.nan,
    'NONE': np.nan
})
print(f"   - Valores únicos: {df['SEXO'].value_counts().to_dict()}")

# 9. TARGET - padronização
print("\n9. Padronização TARGET:")
df['NORMAL X ANORMAL'] = df['NORMAL X ANORMAL'].astype(str).str.capitalize().str.strip()
df['NORMAL X ANORMAL'] = df['NORMAL X ANORMAL'].replace({
    'Normais': 'Normal',
    'Nan': np.nan,
    'None': np.nan
})
print(f"   - Valores únicos: {df['NORMAL X ANORMAL'].value_counts().to_dict()}")

# 10. PULSOS - padronização
print("\n10. Padronização PULSOS:")
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

# 11. SOPRO - padronização
print("\n11. Padronização SOPRO:")
df['SOPRO'] = df['SOPRO'].astype(str).str.capitalize().str.strip()
df['SOPRO'] = df['SOPRO'].replace({
    'Sistólico': 'Sistolico',
    'Contínuo': 'Continuo',
    'Diastólico': 'Diastolico',
    'Sistolico e diastólico': 'Sistolico_Diastolico',
    'Nan': np.nan,
    'None': np.nan
})

# 12. B2 - padronização
df['B2'] = df['B2'].astype(str).str.capitalize().str.strip()
df['B2'] = df['B2'].replace({'Nan': np.nan, 'None': np.nan})

# 13. REMOVER COLUNAS IRRELEVANTES
print("\n12. Remoção de colunas:")
colunas_remover = ['HDA2', 'Atendimento', 'DN']
colunas_removidas = [col for col in colunas_remover if col in df.columns]
df = df.drop(columns=colunas_removidas, errors='ignore')
print(f"   - Removidas: {', '.join(colunas_removidas)}")

# 14. ELIMINAR LINHAS COM >50% MISSING (exceto SEXO e TARGET)
print("\n13. Eliminação de linhas:")
antes_threshold = len(df)
threshold = len(df.columns) * 0.5
df = df.dropna(thresh=threshold)
print(f"   - Linhas com >50% missing: {antes_threshold - len(df)}")

# SEXO e TARGET são obrigatórios
antes_sexo = len(df)
df = df[df['SEXO'].notna()]
print(f"   - Linhas sem SEXO: {antes_sexo - len(df)}")

antes_target = len(df)
df = df[df['NORMAL X ANORMAL'].notna()]
df = df[df['NORMAL X ANORMAL'].isin(['Normal', 'Anormal'])]
print(f"   - Linhas sem TARGET válido: {antes_target - len(df)}")

# RESUMO FINAL
print("\n" + "="*80)
print("RESUMO FINAL")
print("="*80)
print(f"\nRegistos finais: {len(df)} (de {inicial} iniciais)")
print(f"Colunas finais: {len(df.columns)}")

print(f"\nDistribuição TARGET:")
print(df['NORMAL X ANORMAL'].value_counts())
print(f"Balanceamento: {df['NORMAL X ANORMAL'].value_counts(normalize=True).round(3).to_dict()}")

print(f"\nValores em falta por coluna (após limpeza):")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
for col, count in missing.items():
    pct = (count/len(df))*100
    print(f"   {col}: {count} ({pct:.1f}%)")

print(f"\nEstatísticas descritivas:")
numeric_cols = ['IDADE', 'Peso', 'Altura', 'IMC', 'PA SISTOLICA', 'PA DIASTOLICA', 'FC']
print(df[numeric_cols].describe().round(2))

# Guardar dataset
output_file = 'UCMF_CLEAN.xls'
df.to_csv(output_file, index=False, sep=';', encoding='utf-8-sig', quotechar='"', quoting=1)
print(f"\nDataset limpo guardado em: {output_file}")
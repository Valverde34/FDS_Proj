import pandas as pd
import numpy as np

# Ler o arquivo Excel
df = pd.read_excel(r'C:\Users\diogo\OneDrive - Universidade de Aveiro\Documents\M_ERSI\1ano\Int_Ciencia_Dados\FDS_Proj\UCMF.xls')

# Remover duplicados de ID
df = df.drop_duplicates(subset=['ID'], keep='first')

# IDADE - Converter para numérico e filtrar
df['IDADE'] = pd.to_numeric(df['IDADE'], errors='coerce')
df['Idade'] = df['IDADE'].round(0).astype('Int64')
df = df[(df['Idade'] >= 0) & (df['Idade'] <= 19)]

# PESO - Limpar valores inválidos
df['Peso_clean'] = pd.to_numeric(df['Peso'], errors='coerce')
df.loc[(df['Peso_clean'] <= 0) | (df['Peso_clean'] < 0.5) | (df['Peso_clean'] > 150), 'Peso_clean'] = np.nan

# ALTURA - Limpar valores inválidos
df['Altura_clean'] = pd.to_numeric(df['Altura'], errors='coerce')
df.loc[(df['Altura_clean'] <= 0) | (df['Altura_clean'] < 30) | (df['Altura_clean'] > 200), 'Altura_clean'] = np.nan

# IMC - Recalcular
df['IMC_clean'] = np.where(
    (df['Peso_clean'].notna()) & (df['Altura_clean'].notna()) & (df['Altura_clean'] > 0),
    df['Peso_clean'] / ((df['Altura_clean'] / 100) ** 2),
    np.nan
)

print(f"Registos após limpeza: {len(df)}")
print(f"Missing - Peso_clean: {df['Peso_clean'].isna().sum()}, Altura_clean: {df['Altura_clean'].isna().sum()}, IMC_clean: {df['IMC_clean'].isna().sum()}")

print(df[['ID', 'Peso_clean', 'Altura_clean', 'IMC_clean', 'Idade']].head(20))
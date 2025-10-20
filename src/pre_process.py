import pandas as pd

# Ler o arquivo Excel
df = pd.read_excel(r'C:\Users\diogo\OneDrive - Universidade de Aveiro\Documents\M_ERSI\1ano\Int_Ciencia_Dados\Projeto\UCMF.xls')

# Converter IDADE para numérico (transforma erros como '#VALUE!' em NaN)
df['IDADE'] = pd.to_numeric(df['IDADE'], errors='coerce')

# Criar nova variável Idade como inteiro, mantendo NA onde existe NA
df['Idade'] = df['IDADE'].round(0).astype('Int64')  # Int64 suporta NA

# Excluir registros com idade negativa ou maior que 19
df = df[(df['Idade'] >= 0) & (df['Idade'] <= 19)]

print(df[['ID', 'DN', 'Atendimento', 'Idade']].head(20))
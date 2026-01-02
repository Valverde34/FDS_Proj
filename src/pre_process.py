import pandas as pd
import numpy as np

# Carregar dados
df = pd.read_excel(r'../UCMF.xls')

print("\n=== PRÉ-PROCESSAMENTO DE DADOS ===")
print(f"Dimensões iniciais: {df.shape}")

# Verificar colunas com valor constante
for col in df.columns:
    unique_vals = df[col].nunique()

# Missing values nos dados brutos
missing_raw = df.isnull().sum()
missing_raw = missing_raw[missing_raw > 0].sort_values(ascending=False)

# 1. REMOVER DUPLICADOS
inicial = len(df)
df = df.drop_duplicates(subset=['ID'], keep='first')
removidos = inicial - len(df)

# 2. IDADE
df['IDADE'] = pd.to_numeric(df['IDADE'], errors='coerce')
idades_negativas = (df['IDADE'] < 0).sum()
idades_maiores = (df['IDADE'] > 19).sum()
# Substituir por NaN 
df.loc[(df['IDADE'] < 0) | (df['IDADE'] > 19), 'IDADE'] = np.nan
df['IDADE'] = df['IDADE'].round(2)  

# 3. PESO
df['Peso'] = pd.to_numeric(df['Peso'], errors='coerce')
pesos_invalidos = ((df['Peso'] < 0.5) | (df['Peso'] > 150)).sum()
df.loc[(df['Peso'] < 0.5) | (df['Peso'] > 150), 'Peso'] = np.nan

# 4. ALTURA
df['Altura'] = pd.to_numeric(df['Altura'], errors='coerce')
alturas_invalidas = ((df['Altura'] < 30) | (df['Altura'] > 200)).sum()
df.loc[(df['Altura'] < 30) | (df['Altura'] > 200), 'Altura'] = np.nan

# 4.1. VERIFICAÇÃO ADICIONAL - Outliers Extremos vs Valores Impossíveis
# Filtros de segurança (valores clinicamente impossíveis)
# IDADE: 0-19 anos (pediatria)
# PESO: 0.5-150 kg (recém-nascido a adolescente obeso)  
# ALTURA: 30-200 cm (prematuro a adolescente alto)

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

# 5.1 FEATURE ENGINEERING - Criar categorias clinicas para IDADE e IMC
# Criar categorias de IDADE baseadas em cardiologia pediátrica
# Fonte: American Heart Association - Pediatric Cardiology Guidelines
df['Categoria_Idade'] = pd.cut(df['IDADE'], 
                               bins=[-0.1, 1, 5, 12, 19],
                               labels=['Lactente (0-1)', 'Crianca (1-5)', 
                                      'Escolar (5-12)', 'Adolescente (12-19)'],
                               include_lowest=True)

# Criar categorias de IMC para crianças/adolescentes
# Fonte: WHO Growth Standards e CDC BMI-for-age percentiles
df['Categoria_IMC'] = pd.cut(df['IMC'],
                             bins=[0, 14, 18, 23, 30, 100],
                             labels=['Baixo_Peso', 'Normal', 'Sobrepeso', 
                                    'Obesidade', 'Obesidade_Grave'],
                             include_lowest=True)

# 6. PRESSÃO ARTERIAL
# PA Sistólica: 60-180 mmHg | PA Diastólica: 30-120 mmHg
df['PA SISTOLICA'] = pd.to_numeric(df['PA SISTOLICA'], errors='coerce')
df['PA DIASTOLICA'] = pd.to_numeric(df['PA DIASTOLICA'], errors='coerce')
sistolicas_invalidas = ((df['PA SISTOLICA'] < 60) | (df['PA SISTOLICA'] > 180)).sum()
diastolicas_invalidas = ((df['PA DIASTOLICA'] < 30) | (df['PA DIASTOLICA'] > 120)).sum()
df.loc[(df['PA SISTOLICA'] < 60) | (df['PA SISTOLICA'] > 180), 'PA SISTOLICA'] = np.nan
df.loc[(df['PA DIASTOLICA'] < 30) | (df['PA DIASTOLICA'] > 120), 'PA DIASTOLICA'] = np.nan

# 6.1. CALCULAR PPA (Pressão Arterial Média)
df['PPA'] = np.where(
    (df['PA SISTOLICA'].notna()) & (df['PA DIASTOLICA'].notna()),
    df['PA DIASTOLICA'] + (df['PA SISTOLICA'] - df['PA DIASTOLICA']) / 3,
    np.nan
)

# 7. FREQUÊNCIA CARDÍACA
# FC: 40-200 bpm (intervalo conservador para pediatria)
df['FC'] = pd.to_numeric(df['FC'], errors='coerce')
fc_invalidas = ((df['FC'] < 40) | (df['FC'] > 200)).sum()
df.loc[(df['FC'] < 40) | (df['FC'] > 200), 'FC'] = np.nan

# 8. SEXO - padronização
df['SEXO'] = df['SEXO'].astype(str).str.upper().str.strip()
df['SEXO'] = df['SEXO'].replace({
    'MASCULINO': 'M',
    'FEMININO': 'F',
    'INDETERMINADO': 'I',
    'NAN': np.nan,
    'NONE': np.nan
})

# 9. TARGET - padronização
df['NORMAL X ANORMAL'] = df['NORMAL X ANORMAL'].astype(str).str.capitalize().str.strip()
df['NORMAL X ANORMAL'] = df['NORMAL X ANORMAL'].replace({
    'Normais': 'Normal',
    'Nan': np.nan,
    'None': np.nan
})

# 10. PULSOS - padronização
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

# 12.1 CONVENIO - padronização
df['Convenio'] = df['Convenio'].astype(str).str.strip().str.upper()
df['Convenio'] = df['Convenio'].replace({
    'CAMED': 'CAMED',
    'MEDIAL': 'MEDIAL',
    'REAL S.': 'REAL_SAUDE',
    'GRUPO': 'GRUPO_SAUDE',
    'NAN': np.nan,
    'NONE': np.nan
})

# 12.2 MOTIVO1 - padronização
df['MOTIVO1'] = df['MOTIVO1'].astype(str).str.strip()
df['MOTIVO1'] = df['MOTIVO1'].replace({'nan': np.nan, 'None': np.nan})

# 12.3 MOTIVO2 - padronização
df['MOTIVO2'] = df['MOTIVO2'].astype(str).str.strip()
df['MOTIVO2'] = df['MOTIVO2'].replace({
    'Outro': 'OUTRO',
    'nan': np.nan,
    'None': np.nan
})

# 13. ELIMINAR LINHAS COM >50% MISSING
antes_threshold = len(df)
threshold = int(len(df.columns) * 0.5)
df = df.dropna(thresh=threshold)
removidas = antes_threshold - len(df)

# 14. REMOVER COLUNAS COM >50% MISSING
missing_threshold = len(df) * 0.5
colunas_com_muitos_missing = []
for col in df.columns:
    missing_count = df[col].isnull().sum()
    if missing_count > missing_threshold:
        colunas_com_muitos_missing.append(col)
        
if colunas_com_muitos_missing:
    df = df.drop(columns=colunas_com_muitos_missing)

# 15. REMOVER COLUNAS CONSTANTES
colunas_constantes = []
for col in df.columns:
    unique_count = df[col].nunique(dropna=True)
    if unique_count <= 1:
        colunas_constantes.append(col)
        
if colunas_constantes:
    df = df.drop(columns=colunas_constantes)

# 16. REMOVER COLUNA ID E OUTRAS IRRELEVANTES
colunas_remover = ['ID', 'HDA2', 'Atendimento', 'DN']
colunas_removidas = [col for col in colunas_remover if col in df.columns]
if colunas_removidas:
    df = df.drop(columns=colunas_removidas, errors='ignore')

# 17. VALIDAÇÃO DE CAMPOS OBRIGATÓRIOS (SEXO e TARGET)
# SEXO é obrigatório
antes_sexo = len(df)
df = df[df['SEXO'].notna()]

# TARGET é obrigatório e deve ser Normal ou Anormal
antes_target = len(df)
df = df[df['NORMAL X ANORMAL'].notna()]
df = df[df['NORMAL X ANORMAL'].isin(['Normal', 'Anormal'])]

# RESUMO FINAL
print("\n=== RESUMO FINAL ===")
print(f"Registos: {len(df)} (inicial: {inicial})")
print(f"Colunas: {len(df.columns)}")
print(f"\nDistribuição TARGET:")
print(df['NORMAL X ANORMAL'].value_counts())

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    print(f"\nMissing values:")
    for col, count in missing.items():
        pct = (count/len(df))*100
        print(f"  {col}: {count} ({pct:.1f}%)")

# Guardar dataset limpo
output_file = './UCMF_CLEAN2.xlsx'
df.to_excel(output_file, index=False, engine='openpyxl')
print(f"\nDataset salvo: {output_file}")
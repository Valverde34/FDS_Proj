import pandas as pd
import numpy as np

# Carregar dados
df = pd.read_excel(r'C:\Uni\1_ano\1_semestre\CD\FDS_Proj\UCMF.xls')

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
pesos_invalidos = ((df['Peso'] < 0.5) | (df['Peso'] > 150)).sum()
print(f"   - Pesos inválidos (<0.5 kg ou >150 kg): {pesos_invalidos}")
df.loc[(df['Peso'] < 0.5) | (df['Peso'] > 150), 'Peso'] = np.nan

# 4. ALTURA
print("\n4. Limpeza ALTURA:")
df['Altura'] = pd.to_numeric(df['Altura'], errors='coerce')
alturas_invalidas = ((df['Altura'] < 30) | (df['Altura'] > 200)).sum()
print(f"   - Alturas inválidas (<30 cm ou >200 cm): {alturas_invalidas}")
df.loc[(df['Altura'] < 30) | (df['Altura'] > 200), 'Altura'] = np.nan

# 4.1. VERIFICAÇÃO ADICIONAL - Outliers Extremos vs Valores Impossíveis
print("\n4.1. Verificação de Outliers Extremos:")
# Para garantir que não há erros de digitação (ex: 900 kg, 150 anos)
# Já fazemos isso acima, mas vamos documentar melhor
antes_outliers = len(df)

# Filtros de segurança (valores clinicamente impossíveis)
# IDADE: Já filtrado (0-19 anos)
# PESO: Já filtrado (0.5-150 kg)  
# ALTURA: Já filtrado (30-200 cm)
# Estes limites protegem contra erros de digitação mantendo valores válidos

print(f"   ✓ Filtros de valores impossíveis aplicados")
print(f"   ✓ IDADE: 0-19 anos (pediatria)")
print(f"   ✓ PESO: 0.5-150 kg (recém-nascido a adolescente obeso)")
print(f"   ✓ ALTURA: 30-200 cm (prematuro a adolescente alto)")

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
print("\n5.1. Feature Engineering (Intervalos Clínicos):")

# Criar categorias de IDADE baseadas em cardiologia pediátrica
# Fonte: American Heart Association - Pediatric Cardiology Guidelines
df['Categoria_Idade'] = pd.cut(df['IDADE'], 
                               bins=[-0.1, 1, 5, 12, 19],
                               labels=['Lactente (0-1)', 'Crianca (1-5)', 
                                      'Escolar (5-12)', 'Adolescente (12-19)'],
                               include_lowest=True)

# Criar categorias de IMC para crianças/adolescentes
# Fonte: WHO Growth Standards e CDC BMI-for-age percentiles
# Valores aproximados (devem ser ajustados por percentis de idade, mas usamos médias)
df['Categoria_IMC'] = pd.cut(df['IMC'],
                             bins=[0, 14, 18, 23, 30, 100],
                             labels=['Baixo_Peso', 'Normal', 'Sobrepeso', 
                                    'Obesidade', 'Obesidade_Grave'],
                             include_lowest=True)

print(f"   - Categorias de Idade criadas:")
print(f"     {df['Categoria_Idade'].value_counts().to_dict()}")
print(f"   - Categorias de IMC criadas:")
print(f"     {df['Categoria_IMC'].value_counts().to_dict()}")

# 6. PRESSÃO ARTERIAL
print("\n6. Limpeza PRESSÃO ARTERIAL:")
print("   JUSTIFICATIVA CLÍNICA: Utilizamos limites fisiológicos recomendados para")
print("   pediatria (0-19 anos) baseados em guidelines da American Academy of Pediatrics.")
print("   PA Sistólica: 60-180 mmHg | PA Diastólica: 30-120 mmHg")
print("   Valores fora destes intervalos são considerados erros de medição ou digitação.")
df['PA SISTOLICA'] = pd.to_numeric(df['PA SISTOLICA'], errors='coerce')
df['PA DIASTOLICA'] = pd.to_numeric(df['PA DIASTOLICA'], errors='coerce')
sistolicas_invalidas = ((df['PA SISTOLICA'] < 60) | (df['PA SISTOLICA'] > 180)).sum()
diastolicas_invalidas = ((df['PA DIASTOLICA'] < 30) | (df['PA DIASTOLICA'] > 120)).sum()
print(f"   - PA Sistólicas fora do intervalo: {sistolicas_invalidas}")
print(f"   - PA Diastólicas fora do intervalo: {diastolicas_invalidas}")
df.loc[(df['PA SISTOLICA'] < 60) | (df['PA SISTOLICA'] > 180), 'PA SISTOLICA'] = np.nan
df.loc[(df['PA DIASTOLICA'] < 30) | (df['PA DIASTOLICA'] > 120), 'PA DIASTOLICA'] = np.nan

# 7. FREQUÊNCIA CARDÍACA
print("\n7. Limpeza FC:")
print("   JUSTIFICATIVA CLÍNICA: FC em pediatria varia com idade, mas mantemos intervalo")
print("   conservador (40-200 bpm) para incluir bradicardia em atletas e taquicardia febril.")
print("   Valores extremos (FC>300) são considerados erros de digitação ou medição.")
df['FC'] = pd.to_numeric(df['FC'], errors='coerce')
fc_invalidas = ((df['FC'] < 40) | (df['FC'] > 200)).sum()
print(f"   - FC fora do intervalo (40-200 bpm): {fc_invalidas}")
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
print("\n12. Padronização B2:")
df['B2'] = df['B2'].astype(str).str.capitalize().str.strip()
df['B2'] = df['B2'].replace({'Nan': np.nan, 'None': np.nan})

# 12.1 CONVENIO - padronização (normalizar variações)
print("\n12.1. Padronização CONVENIO:")
df['Convenio'] = df['Convenio'].astype(str).str.strip().str.upper()
df['Convenio'] = df['Convenio'].replace({
    'CAMED': 'CAMED',
    'MEDIAL': 'MEDIAL',
    'REAL S.': 'REAL_SAUDE',
    'GRUPO': 'GRUPO_SAUDE',
    'NAN': np.nan,
    'NONE': np.nan
})
print(f"   - Valores únicos após padronização: {df['Convenio'].nunique()}")

# 12.2 MOTIVO1 - padronização
print("\n12.2. Padronização MOTIVO1:")
df['MOTIVO1'] = df['MOTIVO1'].astype(str).str.strip()
df['MOTIVO1'] = df['MOTIVO1'].replace({'nan': np.nan, 'None': np.nan})
print(f"   - Valores únicos: {df['MOTIVO1'].nunique()}")

# 12.3 MOTIVO2 - padronização
print("\n12.3. Padronização MOTIVO2:")
df['MOTIVO2'] = df['MOTIVO2'].astype(str).str.strip()
df['MOTIVO2'] = df['MOTIVO2'].replace({
    'Outro': 'OUTRO',
    'nan': np.nan,
    'None': np.nan
})
print(f"   - Valores únicos: {df['MOTIVO2'].nunique()}")

# 13. REMOVER COLUNAS IRRELEVANTES
print("\n13. Remoção de colunas:")
colunas_remover = ['HDA2', 'Atendimento', 'DN']
colunas_removidas = [col for col in colunas_remover if col in df.columns]
df = df.drop(columns=colunas_removidas, errors='ignore')
print(f"   - Removidas: {', '.join(colunas_removidas)}")

# 14. ELIMINAR LINHAS COM >50% MISSING (exceto SEXO e TARGET)
print("\n14. Eliminação de linhas:")
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

# Guardar dataset limpo como Excel (.xlsx)
output_file = 'UCMF_CLEAN.xlsx'
df.to_excel(output_file, index=False, engine='openpyxl')
print(f"\n✓ Dataset limpo salvo em: {output_file}")
"""
Script de Verificação - Projeto FDS
Verifica se todos os outputs esperados foram gerados
"""

import os
from pathlib import Path

# Diretório src
src_dir = Path(__file__).parent

print("="*80)
print("VERIFICAÇÃO DE OUTPUTS - Projeto FDS")
print("="*80)

# Lista de arquivos esperados
expected_files = {
    'Dados Limpos': [
        'UCMF_CLEAN.xlsx'
    ],
    
    'Raw Data Analysis (raw_*)': [
        'raw_01_tipos_variaveis.csv',
        'raw_02_problemas_identificados.csv',
        'raw_03_statistics_raw.csv',
        'raw_04_histograms_raw.png',
        'raw_05_boxplots_raw.png',
        'raw_06_target_distribution_raw.png'
    ],
    
    'EDA Básico (01-07)': [
        '01_statistics_summary.csv',
        '02_histograms_density.png',
        '03_boxplots.png',
        '04_spread_measures.csv',
        '05_target_distribution.png',
        '06_categorical_distributions.png',
        '07_normality_tests.csv'
    ],
    
    'Análise Bivariada (08-17)': [
        '08_correlation_pearson.csv',
        '09_correlation_spearman.csv',
        '10_correlation_kendall.csv',
        '11_correlation_heatmaps.png',
        '12_strong_correlations.csv',
        '13_simple_regressions.csv',
        '14_scatterplots_correlations.png',
        '15_ttest_normal_vs_anormal.csv',
        '16_boxplots_normal_vs_anormal.png',
        '17_violinplots_normal_vs_anormal.png'
    ],
    
    'Análise Multivariada (18-31)': [
        '18_multiple_regression_coefficients.csv',
        '19_multiple_regression_plot.png',
        '20_mutual_information.csv',
        '21_mutual_information_plot.png',
        '22_pca_variance.csv',
        '23_pca_variance_plots.png',
        '24_pca_loadings.csv',
        '25_pca_biplot.png',
        '26_cluster_metrics.csv',
        '27_cluster_evaluation.png',
        '28_cluster_vs_target.csv',
        '29_clusters_vs_target_pca.png',
        '30_cluster_profiles.csv',
        '31_cluster_profiles_heatmap.png'
    ],
    
    'Feature Selection (32-42)': [
        '32_feature_variances.csv',
        '33_univariate_ftest.csv',
        '34_mutual_information_scores.csv',
        '35_random_forest_importance.csv',
        '36_gradient_boosting_importance.csv',
        '37_rfe_rankings.csv',
        '38_consolidated_feature_ranking.csv',
        '39_feature_selection_comparison.png',
        '40_feature_selection_heatmap.png',
        '41_cv_feature_selection.csv',
        '42_cv_performance_vs_features.png'
    ],
    
    'Modelos & Avaliação (43-50)': [
        '43_model_comparison.csv',
        '44_confusion_matrices.png',
        '45_roc_curves.png',
        '46_metrics_comparison.png',
        '47_feature_importance_models.png',
        '48_decision_tree_visualization.png',
        '49_classification_reports.csv',
        '50_cross_validation_results.csv'
    ]
}

# Verificar cada categoria
total_files = 0
missing_files = 0
found_files = 0

for category, files in expected_files.items():
    print(f"\n{category}")
    print("-" * 60)
    
    category_missing = 0
    category_found = 0
    
    for filename in files:
        total_files += 1
        filepath = src_dir / filename
        
        if filepath.exists():
            print(f"  ✓ {filename}")
            found_files += 1
            category_found += 1
        else:
            print(f"  ✗ {filename} [FALTANDO]")
            missing_files += 1
            category_missing += 1
    
    print(f"  Subtotal: {category_found}/{len(files)} encontrados")

# Resumo final
print("\n" + "="*80)
print("RESUMO FINAL")
print("="*80)
print(f"\nTotal de arquivos esperados: {total_files}")
print(f"Encontrados: {found_files} ({found_files/total_files*100:.1f}%)")
print(f"Faltando: {missing_files} ({missing_files/total_files*100:.1f}%)")

if missing_files == 0:
    print("\n🎉 PARABÉNS! Todos os outputs foram gerados com sucesso!")
    print("\nPróximos passos:")
    print("  1. Consulte README.md para visão geral")
    print("  2. Veja 43_model_comparison.csv para comparar modelos")
    print("  3. Analise os gráficos PNG para visualizações")
elif missing_files < total_files * 0.2:  # Menos de 20% faltando
    print("\n⚠️ Quase completo! Alguns arquivos estão faltando.")
    print("\nSugestão:")
    print("  Execute novamente os scripts correspondentes aos arquivos faltando")
else:
    print("\n❌ Muitos arquivos faltando.")
    print("\nSugestão:")
    print("  Execute: python run_all.py")
    print("  Isso irá gerar todos os outputs automaticamente")

# Verificar scripts
print("\n" + "="*80)
print("SCRIPTS DISPONÍVEIS")
print("="*80)

scripts = [
    'run_all.py',
    'pre_process.py',
    'raw_data_analysis.py',
    'eda_basic.py',
    'bivariate_analysis.py',
    'multivariate_analysis.py',
    'feature_selection.py',
    'predictive_models.py'
]

print("\nScripts Python:")
for script in scripts:
    if (src_dir / script).exists():
        print(f"  ✓ {script}")
    else:
        print(f"  ✗ {script} [FALTANDO]")

print("\n" + "="*80)

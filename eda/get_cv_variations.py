import os
import glob
import pandas as pd

# =====================================================================
# CONFIGURAÇÕES ACADÊMICAS
# =====================================================================
results_config = {
    "Default": "paper_results",
    "Samples 64": "paper_results/variations/samples_64",
    "Samples 256": "paper_results/variations/samples_256",
    "Without tta": "paper_results/variations/without_tta",
    "Mean Pooling Only": "paper_results/variations/mean_pooling_only",
    "Without Threshold Optim": "paper_results/variations/without_threshold_optim"
}

# =====================================================================
# EXTRAÇÃO DOS DADOS
# =====================================================================
data_list = []

for display_name, folder_path in results_config.items():
    pattern = os.path.join("results", "recall_free", folder_path, "classification_reports", "test_year_*.csv")
    files = glob.glob(pattern)

    for file_path in files:
        try:
            filename = os.path.basename(file_path)
            parts = filename.replace(".csv", "").split("_")
            year = int(parts[2])
            seed = int(parts[4])

            df_report = pd.read_csv(file_path, index_col=0)
            df_report.index = df_report.index.astype(str)

            for label in ['1', '1.0', 'Class 1', 'Citrus', 'citrus']:
                if label in df_report.index:
                    f1 = float(df_report.loc[label, 'f1-score'])
                    data_list.append({
                        "Model": display_name,
                        "Year": year,
                        "Seed": seed,
                        "F1-Score": f1
                    })
                    break
        except Exception as e:
            pass # Ignora erros silenciosamente para continuar a extração

df = pd.DataFrame(data_list)

if df.empty:
    print("ERRO: Nenhum dado foi extraído. Verifique os caminhos dos arquivos.")
    exit()

# =====================================================================
# CÁLCULO DO COEFICIENTE DE VARIAÇÃO (CV)
# =====================================================================

# 1. CV Anual (Agrupado por Modelo e Ano)
yearly_stats = df.groupby(['Model', 'Year'])['F1-Score'].agg(['mean', 'std']).reset_index()
yearly_stats['CV_str'] = ((yearly_stats['std'] / yearly_stats['mean']) * 100).apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "-")

# Pivot para colocar os Anos nas colunas
df_cv_yearly = yearly_stats.pivot(index='Model', columns='Year', values='CV_str')

# 2. CV Unificado (Agrupado apenas por Modelo, ignorando o Ano)
unified_stats = df.groupby('Model')['F1-Score'].agg(['mean', 'std']).reset_index()
unified_stats['Unified'] = ((unified_stats['std'] / unified_stats['mean']) * 100).apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "-")
unified_stats.set_index('Model', inplace=True)

# 3. Juntando Tudo
df_final = df_cv_yearly.join(unified_stats['Unified']).fillna("-")

# Ordenar para garantir que o modelo proposto fique no topo
ordered_models = list(results_config.keys())
df_final = df_final.reindex([m for m in ordered_models if m in df_final.index])

# =====================================================================
# IMPRESSÃO NO TERMINAL
# =====================================================================
print("\n" + "="*80)
print(" TABELA: COEFICIENTE DE VARIAÇÃO (CV) - ESTABILIDADE DO F1-SCORE")
print("="*80)
print(df_final.to_string())
print("="*80 + "\n")

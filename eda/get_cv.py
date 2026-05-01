import os
import glob
import pandas as pd

# =====================================================================
# CONFIGURAÇÕES ACADÊMICAS
# =====================================================================
results_config = {
    "PSE-CNN (Proposed)": "paper_results",
    "PSE-TAE": "paper_results_pse_tae",
    "PSE-Transformer": "paper_results_pse_transformer",
    "RF (Bands)": "paper_results_rf",
    "RF (Bands+Idx)": "paper_results_rf_bands_indexes",
    "XGB (Bands)": "paper_results_xgb",
    "XGB (Bands+Idx)": "paper_results_xgb_bands_indexes"
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
yearly_stats['CV_str'] = ((yearly_stats['std'] / yearly_stats['mean']) * 100).apply(lambda x: f"{x:.2f}\\%" if pd.notna(x) else "-")

# Pivot para colocar os Anos (2023, 2024) nas colunas
df_cv_yearly = yearly_stats.pivot(index='Model', columns='Year', values='CV_str')

# 2. CV Unificado (Agrupado apenas por Modelo, ignorando o Ano)
# Isso avalia a robustez do modelo tanto às sementes quanto à variação temporal
unified_stats = df.groupby('Model')['F1-Score'].agg(['mean', 'std']).reset_index()
unified_stats['Unified'] = ((unified_stats['std'] / unified_stats['mean']) * 100).apply(lambda x: f"{x:.2f}\\%" if pd.notna(x) else "-")
unified_stats.set_index('Model', inplace=True)

# 3. Juntando Tudo
df_final = df_cv_yearly.join(unified_stats['Unified']).fillna("-")

# Ordenar para garantir que o modelo proposto fique no topo
ordered_models = list(results_config.keys())
df_final = df_final.reindex([m for m in ordered_models if m in df_final.index])

# =====================================================================
# GERAÇÃO DO CÓDIGO LATEX
# =====================================================================
print("\n" + "="*60)
print(" CÓDIGO LATEX GERADO PARA O ARTIGO")
print("="*60 + "\n")

# Pega os anos disponíveis (ex: [2023, 2024]) de forma dinâmica
years = [col for col in df_final.columns if col != 'Unified']

latex = "% --- TABELA: CV (2023, 2024 e UNIFIED) ---\n"
latex += "\\begin{table}[ht!]\n\\centering\n"
latex += "\\caption{Coefficient of Variation (CV) for the F1-Score. The Unified column evaluates the overall structural stability across both random weight initializations and temporal domains.}\n"
latex += "\\label{tab:cv_stability}\n"
latex += "\\begin{tabular}{lccc}\n\\toprule\n"

# Monta o cabeçalho
header_years = " & ".join([f"\\textbf{{{y}}}" for y in years])
latex += f"\\textbf{{Model}} & {header_years} & \\textbf{{Unified}} \\\\\n\\midrule\n"

for model in df_final.index:
    # Coleta os valores da linha
    vals = []
    for y in years:
        vals.append(str(df_final.loc[model, y]))
    vals.append(str(df_final.loc[model, 'Unified']))
    
    # Destaca o modelo proposto em negrito
    if "Proposed" in model:
        latex += f"\\textbf{{{model}}} & " + " & ".join([f"\\textbf{{{v}}}" for v in vals]) + " \\\\\n"
    else:
        latex += f"{model} & " + " & ".join(vals) + " \\\\\n"
        
latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

print(latex)

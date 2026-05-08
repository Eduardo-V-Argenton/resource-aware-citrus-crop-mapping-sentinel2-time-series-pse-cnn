import os
import pandas as pd
import re

BASE_DIR = "results/recall_free"
MODELS_MAP = {
    "Default": "paper_results",
    "Samples 64": "paper_results/variations/samples_64",
    "Samples 256": "paper_results/variations/samples_256",
    "Without tta": "paper_results/variations/without_tta",
    "Mean Pooling Only": "paper_results/variations/mean_pooling_only",
    "Without Threshold Optim": "paper_results/variations/without_threshold_optim"
}

def extract_metrics_from_reports(phase_type="test"):
    all_data = []
    
    for model_display_name, folder_name in MODELS_MAP.items():
        report_path = os.path.join(BASE_DIR, folder_name, "classification_reports")
        
        if not os.path.exists(report_path):
            continue
            
        files = [f for f in os.listdir(report_path) if f.startswith(f"{phase_type}_") and f.endswith(".csv")]
        
        for file in files:
            match = re.search(r"year_(\d+)_seed_(\d+)", file)
            if not match:
                continue
                
            year = match.group(1)
            seed = match.group(2)
            
            file_full_path = os.path.join(report_path, file)
            
            try:
                df = pd.read_csv(file_full_path, index_col=0)
                
                possiveis_nomes = ['1', 1, '1.0', 1.0, 'Citrus', 'citrus']
                row = None
                
                for nome in possiveis_nomes:
                    if nome in df.index:
                        row = df.loc[nome]
                        break
                        
                if row is None:
                    print(f"[{model_display_name}] Aviso: Classe Citrus não encontrada no arquivo {file}.")
                    continue
                
                all_data.append({
                    "Model": model_display_name,
                    "Year": year,
                    "Seed": seed,
                    "Precision": float(row['precision']),
                    "Recall": float(row['recall']),
                    "F1-Score": float(row['f1-score'])
                })
            except Exception as e:
                print(f"Erro ao processar {file}: {e}")
                
    return pd.DataFrame(all_data)

def print_terminal_table(df, phase_label):
    if df.empty:
        print(f"\nNenhum dado encontrado para a fase: {phase_label}")
        return

    # 1. Estatísticas agrupadas por Ano
    stats_years = df.groupby(["Model", "Year"]).agg({
        "Precision": ["mean", "std"],
        "Recall": ["mean", "std"],
        "F1-Score": ["mean", "std"]
    })
    
    # 2. Estatísticas Unificadas (Ignorando o Ano)
    stats_unified = df.groupby(["Model"]).agg({
        "Precision": ["mean", "std"],
        "Recall": ["mean", "std"],
        "F1-Score": ["mean", "std"]
    })
    
    # Formatando as strings de (mean ± std) para os Anos
    formatted_years = pd.DataFrame(index=stats_years.index)
    for metric in ["Precision", "Recall", "F1-Score"]:
        formatted_years[metric] = stats_years.apply(
            lambda x: f"{x[(metric, 'mean')]:.3f} ± {x[(metric, 'std')]:.3f}", axis=1
        )
    
    df_pivot = formatted_years.unstack(level=1)
    df_pivot = df_pivot.swaplevel(0, 1, axis=1)
    
    # Formatando as strings unificadas com uma MultiIndex compátivel
    formatted_unified = pd.DataFrame(index=stats_unified.index)
    for metric in ["Precision", "Recall", "F1-Score"]:
        formatted_unified[("Unified", metric)] = stats_unified.apply(
            lambda x: f"{x[(metric, 'mean')]:.3f} ± {x[(metric, 'std')]:.3f}", axis=1
        )
        
    # Juntando os Anos com a coluna Unified
    df_final = pd.concat([df_pivot, formatted_unified], axis=1)
    
    # Ordenando as colunas (Anos primeiro, depois Unified)
    years = sorted(df["Year"].unique())
    metrics = ["Precision", "Recall", "F1-Score"]
    
    col_order = [(y, m) for y in years for m in metrics] + [("Unified", m) for m in metrics]
    df_final = df_final.reindex(columns=col_order)
    
    # Ordenando os modelos conforme o dicionário MODELS_MAP
    ordered_models = list(MODELS_MAP.keys())
    df_final = df_final.reindex([m for m in ordered_models if m in df_final.index])
    
    # Imprimindo no terminal de forma elegante
    print(f"\n{'='*100}")
    print(f" TABELA: {phase_label.upper()} (Class: Citrus)")
    print(f"{'='*100}")
    print(df_final.to_string())
    print("="*100 + "\n")

# Execução
print("Processando dados de Validação e Teste...")
df_test = extract_metrics_from_reports("test")
df_val = extract_metrics_from_reports("val")

print_terminal_table(df_test, "Test")
print_terminal_table(df_val, "Validation")